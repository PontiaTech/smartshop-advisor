import os
import httpx
from urllib.parse import urlparse

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

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    shopping = data.get("shopping_results") or []

    # Parseamos un pool más grande para poder elegir los mejores
    out: list[dict] = []
    for it in shopping[: max(pool_size, k)]:
        product_url = it.get("link") or it.get("product_link")
        title = it.get("title") or ""
        if not title or not product_url:
            continue

        domain = urlparse(product_url).netloc

        out.append({
            "title": title,
            "url": product_url,
            "source": it.get("source") or domain,
            "price": it.get("price"),
            "extracted_price": it.get("extracted_price"),
            "rating": it.get("rating"),
            "reviews": it.get("reviews"),
            "thumbnail": it.get("thumbnail") or it.get("serpapi_thumbnail"),
            "position": it.get("position"),
        })
        
    return pick_best(out, k=k)