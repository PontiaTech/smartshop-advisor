import os
import httpx
from urllib.parse import urlparse, urlparse, parse_qs, urlencode, urlunparse
from dotenv import load_dotenv
import json

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
key = SERPAPI_API_KEY
print(
    "SERPAPI_API_KEY loaded?", bool(key),
    "len:", len(key) if key else None,
    "preview:", (key[:4] + "..." + key[-4:]) if key else None
)

def add_api_key(url: str, api_key: str) -> str:
    u = urlparse(url)
    q = parse_qs(u.query)
    q["api_key"] = [api_key]  # sobreescribe si ya existiera
    new_query = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))

async def test_web_search():
    params = {
        "engine": "google_shopping",
        "q": "black winter sneakers",
        "google_domain": "google.com",
        "hl": "en",
        "gl": "us",
        "location": "Austin, Texas, United States",
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

    out: list[dict] = []
    for it in shopping[: max(10, 3)]:
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
            "serpapi_immersive_product_api": it.get("serpapi_immersive_product_api"),
            "product_id": it.get("product_id"),
        })

    print("\n=== TOP 3 (compact) ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # primer item con immersive
    immersive_url = None
    for it in shopping:
        immersive_url = it.get("serpapi_immersive_product_api")
        if immersive_url:
            break

    if not immersive_url:
        print("\nNo hay serpapi_immersive_product_api")
        return

    if not SERPAPI_API_KEY:
        print("\nFalta SERPAPI_API_KEY en el entorno (.env)")
        return

    immersive_url_with_key = add_api_key(immersive_url, SERPAPI_API_KEY)

    print("\nImmersive URL (original):")
    print(immersive_url)
    print("\nImmersive URL (with key):")
    print(immersive_url_with_key)

    print("\n=== PARAMS SENT ===")
    print(json.dumps(params, ensure_ascii=False, indent=2))

    async with httpx.AsyncClient(timeout=20) as client:
        r2 = await client.get(immersive_url_with_key)
        print("\nIMMERSIVE HTTP:", r2.status_code)
        if r2.status_code >= 400:
            print("\n=== IMMERSIVE ERROR BODY (first 1200 chars) ===")
            print((r2.text or "")[:1200])
        r2.raise_for_status()
        immersive = r2.json()

    print("\n=== RESPONSE DEBUG ===")
    print("HTTP:", r.status_code)
    print("Top-level keys:", list(data.keys())[:30])

    meta = data.get("search_metadata") or {}
    print("search_metadata.status:", meta.get("status"))
    print("search_metadata.id:", meta.get("id"))
    print("search_metadata.json_endpoint:", meta.get("json_endpoint"))

    if "error" in data:
        print("ERROR:", data["error"])
    if "message" in data:
        print("MESSAGE:", data["message"])

    print("Has shopping_results?:", "shopping_results" in data, "len:", len(data.get("shopping_results") or []))
    print("Has product_results?:", "product_results" in data)
    print("search_parameters:", data.get("search_parameters"))

    # guarda el JSON crudo de immersive
    with open("serpapi_immersive_full.json", "w", encoding="utf-8") as f:
        json.dump(immersive, f, ensure_ascii=False, indent=2)

    print("\nGuardado a disco:")
    print("- serpapi_immersive_full.json")
    print("\nIMMERSIVE top-level keys:", list(immersive.keys())[:40])

    pr = immersive.get("product_results") or immersive.get("product_result")
    with open("serpapi_immersive_product_results.json", "w", encoding="utf-8") as f:
        json.dump(pr, f, ensure_ascii=False, indent=2)
    print("- serpapi_immersive_product_results.json")

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(test_web_search())
    print(result)
