import os
from dotenv import load_dotenv
from chromadb import HttpClient

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products_all")

def main():
    print(f"Conectando a Chroma en {CHROMA_HOST}:{CHROMA_PORT}...")
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(COLLECTION_NAME)

    total = collection.count()
    print(f"Total documentos en colección: {total}")

    # Traemos todos los metadatos
    data = collection.get(
        include=["metadatas"],
        limit=total  # como son ~2600, cabe sin problema
    )

    metadatas = data["metadatas"]

    missing_source = 0
    missing_url = 0

    ejemplos_sin_source = []
    ejemplos_sin_url = []

    for meta in metadatas:
        src = meta.get("source")
        url = meta.get("url")

        if not src or str(src).strip() == "":
            missing_source += 1
            if len(ejemplos_sin_source) < 5:
                ejemplos_sin_source.append(meta)

        if not url or str(url).strip() == "":
            missing_url += 1
            if len(ejemplos_sin_url) < 5:
                ejemplos_sin_url.append(meta)

    print(f"\nPuntos sin 'source': {missing_source}/{total}")
    print(f"Puntos sin 'url':    {missing_url}/{total}")

    if ejemplos_sin_source:
        print("\nEjemplos sin 'source':")
        for i, meta in enumerate(ejemplos_sin_source, start=1):
            print(f"\n--- Ejemplo #{i} ---")
            print("product_name:", meta.get("product_name"))
            print("product_family:", meta.get("product_family"))
            print("url:", meta.get("url"))

    if ejemplos_sin_url:
        print("\nEjemplos sin 'url':")
        for i, meta in enumerate(ejemplos_sin_url, start=1):
            print(f"\n--- Ejemplo #{i} ---")
            print("source:", meta.get("source"))
            print("product_name:", meta.get("product_name"))
            print("product_family:", meta.get("product_family"))

if __name__ == "__main__":
    main()
