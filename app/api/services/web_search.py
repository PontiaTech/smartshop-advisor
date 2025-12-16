import os
import httpx
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from dotenv import load_dotenv

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


def _country_from_lang(lang: str) -> str:
    lang = (lang or "").lower()
    if lang.startswith("es"):
        return "es"
    if lang.startswith("en"):
        return "us"
    if lang.startswith("fr"):
        return "fr"
    if lang.startswith("de"):
        return "de"
    if lang.startswith("it"):
        return "it"
    return "us"

def _default_location(lang: str) -> str:
    lang = (lang or "").lower()
    if lang.startswith("es"):
        return "Madrid, Spain"
    if lang.startswith("en"):
        return "Austin, Texas, United States"
    if lang.startswith("fr"):
        return "Paris, France"
    if lang.startswith("de"):
        return "Berlin, Germany"
    return "Austin, Texas, United States"

def pick_best(items: list[dict], k: int = 3) -> list[dict]:
    def r(x): return float(x.get("rating") or 0.0)
    def n(x): return int(x.get("reviews") or 0)

    qualified = [x for x in items if r(x) > 0 and n(x) >= 10]
    qualified.sort(key=lambda x: (r(x), n(x)), reverse=True)

    if len(qualified) >= k:
        return qualified[:k]

    rest = [x for x in items if x not in qualified]
    rest.sort(key=lambda x: (r(x), n(x)), reverse=True)

    return (qualified + rest)[:k]


def _add_api_key(url: str, api_key: str) -> str:
    """
    SerpApi devuelve serpapi_immersive_product_api sin api_key.
    Esto lo añade sin romper el resto de parámetros.
    """
    u = urlparse(url)
    q = parse_qs(u.query)
    q["api_key"] = [api_key]
    new_query = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


async def web_search_products(query: str, k: int = 3, lang: str = "es", pool_size: int = 20, location: str | None = None) -> list[dict]:
    """
    Devuelve hasta k resultados de Google Shopping (SerpApi).
    Cada item incluye: title, url, source, price, extracted_price, rating, reviews, thumbnail, position.
    """
    if not SERPAPI_API_KEY:
        return []

    q = (query or "").strip()
    if not q or k <= 0:
        return []

    lang = (lang or "en").lower()

    if lang.startswith("es"):
        web_query = f"{q} comprar"
        hl = "es"
    elif lang.startswith("en"):
        web_query = f"{q} buy"
        hl = "en"
    else:
        web_query = q
        hl = lang[:2] if len(lang) >= 2 else "en"

    gl = _country_from_lang(lang)
    loc = location or _default_location(lang)

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

    url = "https://serpapi.com/search.json"
    

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        shopping = data.get("shopping_results") or []

        pool: list[dict] = []
        for it in shopping[: max(pool_size, k)]:
            product_url = it.get("link") or it.get("product_link")
            title = it.get("title") or ""
            if not title or not product_url:
                continue

            domain = urlparse(product_url).netloc

            pool.append({
                "title": title,
                "url": product_url,
                "source": it.get("source") or domain,
                "price": it.get("price"),
                "extracted_price": it.get("extracted_price"),
                "rating": it.get("rating"),
                "reviews": it.get("reviews"),
                "thumbnail": it.get("thumbnail") or it.get("serpapi_thumbnail"),
                "position": it.get("position"),
                "serpapi_immersive_product_api": it.get("serpapi_immersive_product_api"),
                "product_id": it.get("product_id"),
            })

        best = pick_best(pool, k=k)

        enriched: list[dict] = []
        for base in best:
            immersive_url = base.get("serpapi_immersive_product_api")
            product_results = None

            if immersive_url:
                immersive_url = _add_api_key(immersive_url, SERPAPI_API_KEY)
                try:
                    r2 = await client.get(immersive_url)
                    r2.raise_for_status()
                    immersive = r2.json()
                    product_results = immersive.get("product_results") or immersive.get("product_result")
                except httpx.HTTPError:
                    product_results = None

            # enriched.append(_as_vector_record(base, product_results))

    return enriched