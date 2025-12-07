import os
from dotenv import load_dotenv
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products_all")
EMB_MODEL = os.getenv(
    "EMB_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def main():
    print(f"Conectando a Chroma en {CHROMA_HOST}:{CHROMA_PORT}...")
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(COLLECTION_NAME)
    print("Total documentos en colección:", collection.count())

    embedder = SentenceTransformer(EMB_MODEL)

    query = "zapatillas Vans negras"
    print("\nQuery:", query)
    q_emb = embedder.encode([query], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]),
        start=1
    ):
        print(f"\nResultado #{i}")
        print(f"  distancia: {dist:.4f}")
        print(f"  source: {meta.get('source')}")
        print(f"  product_name: {meta.get('product_name')}")
        print(f"  product_family: {meta.get('product_family')}")
        print(f"  url: {meta.get('url')}")
        print("  doc:", doc[:150], "...")

if __name__ == "__main__":
    main()
