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
df = pd.read_csv(csv_path, usecols=["productDisplayName", "articleType"]).dropna()
df["productDisplayName"] = df["productDisplayName"].astype(str).str.strip()
df["articleType"] = df["articleType"].astype(str).str.strip()
df = df[df["productDisplayName"] != ""]
print(f"📦 Total de registros cargados: {len(df)}")

# --- Configuración de modelo y DB ---
EMB_MODEL = os.getenv("EMB_MODEL", "clip-ViT-B-32")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

# --- Conectar a Chroma ---
client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# --- Cargar modelo CLIP multilingüe ---
embedder = SentenceTransformer(EMB_MODEL)
print(f"🧠 Modelo CLIP multi-idioma cargado: {EMB_MODEL}")

# --- Verificar la dimensión del embedding ---
test_emb = embedder.encode(["dimension test"], convert_to_numpy=True)
embedding_dim = test_emb.shape[1]
print(f"📏 Dimensión detectada del embedding: {embedding_dim}")

# --- Reset colección si existe ---
try:
    client.delete_collection(COLLECTION_NAME)
    print("🗑️ Colección anterior eliminada")
except:
    print("ℹ️ No existía colección previa")

# --- Crear colección en Chroma (sin dimension param para HTTP client) ---
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # similitud coseno
)

print(f"🆕 Colección '{COLLECTION_NAME}' creada para embeddings de dimensión {embedding_dim}")


# --- Preparar datos ---
ids = [str(uuid.uuid4()) for _ in range(len(df))]
documents = df["productDisplayName"].tolist()
metas = [{"articleType": a} for a in df["articleType"].tolist()]

# --- Generar embeddings ---
print("🔹 Generando embeddings CLIP...")
embeddings = embedder.encode(
    documents,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
).tolist()

print(f"✅ Ejemplo de vector embedding: {embeddings[0][:5]} ...")
print(f"📐 Dimensión validada: {len(embeddings[0])}")

# --- Insertar en lotes ---
BATCH_SIZE = 2000
print("🚀 Comenzando inserción a ChromaDB...")
for i in range(0, len(ids), BATCH_SIZE):
    collection.add(
        ids=ids[i:i+BATCH_SIZE],
        documents=documents[i:i+BATCH_SIZE],
        embeddings=embeddings[i:i+BATCH_SIZE],
        metadatas=metas[i:i+BATCH_SIZE]
    )
    print(f"✅ Insertados {min(i+BATCH_SIZE, len(ids))}/{len(ids)} productos")

print(f"🎉 Carga completada. Total en colección: {collection.count()} items")
