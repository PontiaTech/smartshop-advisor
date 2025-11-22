import os
import uuid
import pandas as pd
import kagglehub
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import json
import ast

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



def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def infer_product_family_row(row: pd.Series) -> str:
    # 1) root_category si está relleno
    root_cat = safe_str(row.get("root_category", ""))
    if root_cat:
        return root_cat
    
    # 2) Intentar sacarlo del nombre
    name = safe_str(row.get("product_name", "")) or safe_str(row.get("name", ""))
    if name:
        # Ej: "OLD SKOOL UNISEX - Trainers - black/multi-coloured"
        parts = [p.strip() for p in name.split("-")]
        if len(parts) >= 3:
            # la parte central suele ser la categoría ("Trainers", "T-shirt", "Jeans", etc.)
            return parts[1]
    
    # 3) (Opcional) usar other_attributes para intentar deducir
    attrs_str = safe_str(row.get("other_attributes", ""))
    if attrs_str:
        try:
            try:
                attrs = json.loads(attrs_str)
            except json.JSONDecodeError:
                attrs = ast.literal_eval(attrs_str)
            text = " ".join(
                safe_str(item.get("name", "")).lower() + " " +
                safe_str(item.get("value", "")).lower()
                for item in attrs
            )
            if any(x in text for x in ["shoe tip", "heel height", "insole", "sole"]):
                return "Trainers"
            if any(x in text for x in ["waist rise", "leg length", "inseam"]):
                return "Trousers"
            # aquí podrías añadir más reglas…
        except Exception:
            pass
    
    # 4) fallback vacío
    return ""

def ingest_chroma_zalando_mango_zara():
    
    MANGO_CSV = os.getenv("MANGO_CSV", "./data/Mango Products Prepared.csv")
    ZALANDO_CSV = os.getenv("ZALANDO_CSV", "./data/Zalando products Cleaned.csv")
    ZARA_CSV = os.getenv("ZARA_CSV", "./data/Zara - Products Cleaned.csv")

    EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    df_mango = pd.read_csv(MANGO_CSV, encoding="utf-8")
    df_mango = df.fillna("")
    
    df_zara = pd.read_csv(ZARA_CSV, encoding="utf-8")
    df_zara = df.fillna("")
    
    df_zalando = pd.read_csv(ZALANDO_CSV, encoding="utf-8")
    df_zalando = df.fillna("")
    
    mango = pd.DataFrame()
    mango["source"] = "mango"
    mango["product_name"] = df_mango.get("product_name", "")
    mango["image"] = df_mango.get("image", "")
    mango["description"] = df_mango.get("description", "")
    mango["url"] = df_mango.get("url", "")
    mango["product_family"] = df_mango.get("product_family", "")
    
    zara = pd.DataFrame()
    zara["source"] = "zara"
    zara["product_name"] = df_zara.get("product_name", "")
    zara["description"] = df_zara.get("description", "")
    zara["product_family"] = df_zara.get("product_family", "")
    zara["image"] = df_zara.get("image", "")
    zara["url"] = df_zara.get("url", "")
    
    zalando = pd.DataFrame()
    zalando["source"] = "zalando"
    zalando["name"] = df_zara.get("name", "")
    zalando["description"] = df_zara.get("description", "")
    df_zalando["product_family"] = df_zalando.apply(infer_product_family_row, axis=1)
    zalando["main_image"] = df_zara.get("main_image", "")
    zalando["url"] = df_zara.get("url", "")
    
    all_products = pd.concat([df_mango, df_zalando, df_zara], ignore_index=True)
    
    df = all_products.copy()
    # Normalizamos strings
    for col in df.columns:
        df[col] = df[col].apply(safe_str)

    # Documento de texto para embeddings
    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        name = row.get("name", "")
        desc = row.get("description", "")
        text = (name + ". " + desc).strip()

        # Si no hay nada de texto, mejor saltar
        if not text:
            continue

        doc_id = str(uuid.uuid4())

        ids.append(doc_id)
        documents.append(text)

        meta = row.to_dict()
        meta["id"] = doc_id  # por si luego quieres recuperar
        metadatas.append(meta)

    embedder = SentenceTransformer(EMB_MODEL)
    
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]

        embeddings = embedder.encode(
            batch_docs,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta
        )
        print(f"   ✅ Insertados {min(end, len(ids))}/{len(ids)}")
    
    
    
    
    


print(f"🎉 Carga completada. Total en colección: {collection.count()} items")
