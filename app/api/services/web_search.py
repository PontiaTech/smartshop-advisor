import os
import httpx
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from sentence_transformers import SentenceTransformer
from chromadb import HttpClient
import uuid
import logging

from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products_all")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


logger = logging.getLogger("smartshop.websearch")

# Si no tienes logging_config global, al menos esto:
# (si ya lo configuras en otro sitio, borra estas 2 líneas)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


# -------------------------
# Helpers
# -------------------------


def _safe_url(u: str) -> str:
    """
    Hace la URL clicable:
    - Escapa espacios y caracteres raros.
    - Mantiene intactos : / ? & = y demás separadores.
    """
    u = _safe_str(u)
    if not u:
        return ""
    return quote(u, safe=":/?&=#%+-_.~")

def _add_api_key(url: str, api_key: str) -> str:
    u = urlparse(url)
    q = parse_qs(u.query)
    q["api_key"] = [api_key]
    new_query = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


def _safe_str(x) -> str:
    return (x if isinstance(x, str) else str(x or "")).strip()


def _normalize_url(u: str) -> str:
    try:
        p = urlparse(u)
        path = p.path.rstrip("/")
        return urlunparse((p.scheme, p.netloc.lower(), path, p.params, p.query, ""))
    except Exception:
        return (u or "").strip()


def _guess_color_from_title(title: str) -> str:
    t = (title or "").lower()
    for c in ["black", "white", "red", "blue", "green", "beige", "grey", "gray", "brown", "pink", "orange", "yellow"]:
        if f" {c} " in f" {t} ":
            return c
    return ""


def pick_best(items: list[dict], k: int = 3) -> list[dict]:
    def r(x): return float(x.get("rating") or 0.0)
    def n(x): return int(x.get("reviews") or 0)

    qualified = [x for x in items if r(x) > 0 and n(x) >= 10]
    qualified.sort(key=lambda x: (r(x), n(x)), reverse=True)

    logger.debug(
        "pick_best: total=%d qualified=%d k=%d",
        len(items), len(qualified), k
    )

    if len(qualified) >= k:
        return qualified[:k]

    rest = [x for x in items if x not in qualified]
    rest.sort(key=lambda x: (r(x), n(x)), reverse=True)
    return (qualified + rest)[:k]


def _as_vector_record(base: dict, description: str) -> dict:
    title = _safe_str(base.get("title"))
    # url = _safe_str(base.get("url"))
    # thumbnail = _safe_str(base.get("thumbnail"))
    raw_url = _safe_str(base.get("product_link") or base.get("url"))
    
    url = _safe_url(raw_url)

    thumbnail = _safe_str(base.get("thumbnail"))
    thumbnail = _safe_url(thumbnail)
    source = _safe_str(base.get("source"))
    
    logger.info(
    "as_vector_record urls",
    extra={"raw_url": raw_url, "safe_url": url, "raw_thumb": base.get("thumbnail"), "safe_thumb": thumbnail}
    )

    color = _safe_str(base.get("color")) or _guess_color_from_title(title)
    family = _safe_str(base.get("product_family"))  # normalmente vacío en SerpAPI shopping

    return {
        "product_name": title,
        "description": _safe_str(description),
        "product_family": family,
        "image": thumbnail,
        "url": url,
        "color": color,
        "source": source,
    }


# -------------------------
# Main
# -------------------------

async def web_search_products(
    query: str,
    k: int = 3,
    lang: str = "es",
    pool_size: int = 20,
    location: str | None = None,
) -> list[dict]:
    if not SERPAPI_API_KEY:
        logger.warning("SERPAPI_API_KEY not set -> returning []")
        return []

    q = (query or "").strip()
    if not q or k <= 0:
        logger.info("Empty query or k<=0 (q=%r, k=%s) -> returning []", q, k)
        return []

    # Forzamos estos valores por el tema de la descripción
    hl = "en"
    gl = "us"
    loc = location or "Austin, Texas, United States"
    web_query = f"{q} buy"

    params = {
        "engine": "google_shopping",
        "q": web_query,
        "hl": hl,
        "gl": gl,
        "location": loc,
        "api_key": SERPAPI_API_KEY,
        "no_cache": "false",
        "output": "json",
    }

    logger.info(
        "web_search_products: q=%r web_query=%r k=%d pool_size=%d hl=%s gl=%s location=%r",
        q, web_query, k, pool_size, hl, gl, loc
    )
    logger.debug("SerpAPI params=%s", params)

    url = "https://serpapi.com/search.json"

    async with httpx.AsyncClient(timeout=20) as client:
        # 1) shopping search
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.exception("SerpAPI shopping request failed: %s", e)
            return []

        shopping = data.get("shopping_results") or []
        logger.info("SerpAPI shopping_results=%d", len(shopping))
        
        if shopping:
            logger.info("shopping[0] keys", extra={"keys": list(shopping[0].keys())})

        if not shopping:
            logger.warning("No shopping_results for query=%r", web_query)
            return []

        # 2) global desc (fallback)
        global_desc = ""
        first_imm = next(
            (x.get("serpapi_immersive_product_api") for x in shopping if x.get("serpapi_immersive_product_api")),
            None
        )

        if first_imm:
            try:
                imm_url = _add_api_key(first_imm, SERPAPI_API_KEY)
                logger.debug("Fetching global immersive: %s", imm_url)
                r2 = await client.get(imm_url)
                r2.raise_for_status()
                imm = r2.json()
                pr = imm.get("product_results") or imm.get("product_result") or {}
                global_desc = _safe_str((pr.get("about_the_product") or {}).get("description") or "")
                logger.info("global_desc loaded? %s (len=%d)", bool(global_desc), len(global_desc))
            except Exception as e:
                logger.exception("Global immersive fetch failed: %s", e)
                global_desc = ""
        else:
            logger.info("No serpapi_immersive_product_api found -> global_desc empty")

        # 3) build deduped pool
        seen_product_ids: set[str] = set()
        seen_urls: set[str] = set()
        pool: list[dict] = []

        scan_limit = max(pool_size, k * 6)
        logger.debug("Scanning up to %d items to build pool", scan_limit)

        skipped_pid = 0
        skipped_url = 0
        skipped_missing = 0

        for it in shopping[:scan_limit]:
            product_url = it.get("product_link") or it.get("link") or it.get("url") or ""
            title = it.get("title") or ""
            if not title or not product_url:
                skipped_missing += 1
                continue

            pid = _safe_str(it.get("product_id"))
            norm_url = _normalize_url(product_url)

            if pid and pid in seen_product_ids:
                skipped_pid += 1
                continue
            if norm_url and norm_url in seen_urls:
                skipped_url += 1
                continue

            if pid:
                seen_product_ids.add(pid)
            if norm_url:
                seen_urls.add(norm_url)

            domain = urlparse(product_url).netloc

            pool.append({
                "title": title,
                "url": product_url,
                "product_link": it.get("product_link") or it.get("link") or "",
                "source": it.get("source") or domain,
                "price": it.get("price"),
                "extracted_price": it.get("extracted_price"),
                "rating": it.get("rating"),
                "reviews": it.get("reviews"),
                "thumbnail": it.get("thumbnail") or it.get("serpapi_thumbnail"),
                "position": it.get("position"),
                "serpapi_immersive_product_api": it.get("serpapi_immersive_product_api"),
                "product_id": it.get("product_id"),
                "snippet": it.get("snippet"),
            })

            if len(pool) >= pool_size:
                break

        logger.info(
            "Pool built: size=%d (skipped missing=%d pid_dup=%d url_dup=%d)",
            len(pool), skipped_missing, skipped_pid, skipped_url
        )

        if not pool:
            logger.warning("Pool empty after dedupe -> returning []")
            return []

        # 4) pick best
        best_count = min(k, len(pool))
        best = pick_best(pool, k=best_count)
        logger.info("Best selected: %d", len(best))
        logger.debug("Best titles=%s", [b.get("title") for b in best])

        # 5) per-item enrich
        enriched: list[dict] = []
        for idx, base in enumerate(best, start=1):
            title = _safe_str(base.get("title"))
            pid = _safe_str(base.get("product_id"))
            logger.info("Enriching item %d/%d title=%r pid=%r", idx, len(best), title, pid)

            # prefer snippet
            desc = _safe_str(base.get("snippet") or base.get("description") or "")
            if desc:
                logger.debug("Desc from snippet/inline (len=%d)", len(desc))

            # try immersive if still empty
            if not desc:
                imm_url = base.get("serpapi_immersive_product_api")
                if imm_url:
                    try:
                        full_imm = _add_api_key(imm_url, SERPAPI_API_KEY)
                        logger.debug("Fetching immersive for item: %s", full_imm)
                        r3 = await client.get(full_imm)
                        r3.raise_for_status()
                        imm2 = r3.json()
                        pr2 = imm2.get("product_results") or imm2.get("product_result") or {}
                        desc = _safe_str((pr2.get("about_the_product") or {}).get("description") or "")
                        logger.debug("Desc from immersive (len=%d)", len(desc))
                    except Exception as e:
                        logger.exception("Immersive fetch failed for title=%r: %s", title, e)
                        desc = ""
                else:
                    logger.debug("No immersive url for title=%r", title)

            # fallback: global_desc
            if not desc and global_desc:
                desc = global_desc
                logger.debug("Desc fallback -> global_desc (len=%d)", len(desc))

            # fallback: last resort
            if not desc:
                desc = "Product found on the web. Open the link for full details."
                logger.debug("Desc fallback -> generic")

            enriched.append(_as_vector_record(base, description=desc))

        logger.info("web_search_products done -> returning %d items", len(enriched))
        if enriched:
            logger.info("web_search sample", extra={"sample": enriched[0]})

        return enriched