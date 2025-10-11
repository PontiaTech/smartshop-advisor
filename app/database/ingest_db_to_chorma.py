import os
import uuid
import pandas as pd
import kagglehub
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer

# --- Descargar dataset de Kaggle ---
dataset_path = kagglehub.dataset_download("bhavikjikadara/e-commerce-products-images")
print(f"✅ Dataset descargado en: {dataset_path}")

# --- Buscar CSV ---
csv_path = None
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(root, file)
            break
if not csv_path:
    raise FileNotFoundError("❌ No se encontró ningún archivo CSV en el dataset.")
print("✅ CSV encontrado en:", csv_path)

# --- Leer CSV y limpiar ---
df = pd.read_csv(csv_path, usecols=["productDisplayName", "articleType"]).fillna("")
df = df[df["articleType"] != ""].reset_index(drop=True)
print(f"📦 Total de registros cargados: {len(df)}")

# --- Configuración ---
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

# --- Conectar a Chroma ---
client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# --- Cargar modelo de embeddings ---
embedder = SentenceTransformer(EMB_MODEL)
print(f"🧠 Modelo de embeddings cargado: {EMB_MODEL}")

# --- Crear o cargar colección ---
try:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"✅ Colección '{COLLECTION_NAME}' encontrada con {collection.count()} items.")
except Exception:
    collection = client.create_collection(COLLECTION_NAME)
    print(f"🆕 Colección '{COLLECTION_NAME}' creada desde cero.")

# --- Preparar datos ---
texts = df.apply(lambda r: f"{r['articleType']} {r['productDisplayName']}", axis=1).tolist()
ids = [str(uuid.uuid4()) for _ in range(len(texts))]
metas = [{"articleType": str(r["articleType"]).strip()} for _, r in df.iterrows()]

# --- Generar embeddings ---
print("🔹 Generando embeddings (puede tardar un poco)...")
embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True).tolist()

# --- Insertar en lotes ---
BATCH_SIZE = 2000
print("🚀 Iniciando inserción en ChromaDB...")
for i in range(0, len(ids), BATCH_SIZE):
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_embs = embeddings[i:i+BATCH_SIZE]
    batch_metas = metas[i:i+BATCH_SIZE]

    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        embeddings=batch_embs,
        metadatas=batch_metas
    )

    print(f"✅ Ingresados {len(batch_ids)} items (hasta {i + len(batch_ids)}/{len(ids)})")

print(f"🎉 Inserción completada. Total: {collection.count()} items en la colección.")
