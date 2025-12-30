import os
import uuid
import pandas as pd
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import json
import ast
from dotenv import load_dotenv

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products_all")

BATCH_SIZE = 2000

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

def extract_brand_name(val):
    if not isinstance(val, str) or not val.strip():
        return ""
    try:
        data = json.loads(val)
        return data.get("name", "")
    except json.JSONDecodeError:
        return ""

def ingest_chroma_zalando_mango_zara():
    
    MANGO_CSV = os.getenv("MANGO_CSV", "./data/Mango Products Prepared.csv")
    ZALANDO_CSV = os.getenv("ZALANDO_CSV", "./data/Zalando products Cleaned.csv")
    ZARA_CSV = os.getenv("ZARA_CSV", "./data/Zara - Products Cleaned.csv")

    EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️ Colección anterior eliminada")
    except:
        print("ℹ️ No existía colección previa")
    
    collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # similitud coseno
)
    
    df_mango = pd.read_csv(MANGO_CSV, encoding="utf-8")
    df_mango = df_mango.fillna("")
    
    df_zara = pd.read_csv(ZARA_CSV, encoding="utf-8")
    df_zara = df_zara.fillna("")
    
    df_zalando = pd.read_csv(ZALANDO_CSV, encoding="utf-8")
    df_zalando = df_zalando.fillna("")
    
    # mango = pd.DataFrame()
    # mango["source"] = "mango"
    # mango["product_name"] = df_mango.get("product_name", "")
    # mango["image"] = df_mango.get("image", "")
    # mango["description"] = df_mango.get("description", "")
    # mango["url"] = df_mango.get("url", "")
    # mango["product_family"] = df_mango.get("product_family", "")
    
    # zara = pd.DataFrame()
    # zara["source"] = "zara"
    # zara["product_name"] = df_zara.get("product_name", "")
    # zara["description"] = df_zara.get("description", "")
    # zara["product_family"] = df_zara.get("product_family", "")
    # zara["image"] = df_zara.get("image", "")
    # zara["url"] = df_zara.get("url", "")
    
    # zalando = pd.DataFrame()
    # zalando["source"] = "zalando"
    # zalando["name"] = df_zalando.get("name", "")
    # zalando["description"] = df_zalando.get("description", "")
    # zalando["product_family"] = df_zalando.apply(infer_product_family_row, axis=1)
    # zalando["main_image"] = df_zalando.get("main_image", "")
    # zalando["url"] = df_zalando.get("url", "")
    
    # --- Mango ---
    mango = pd.DataFrame({
        "product_name": df_mango.get("product_name", ""),
        "description": df_mango.get("description", ""),
        "product_family": df_mango.get("canonical_family", ""),
        "image": df_mango.get("image", ""),
        "url": df_mango.get("url", ""),
        # "color": ""
    })
    mango["source"] = "mango"

    # --- Zara ---
    zara = pd.DataFrame({
        "product_name": df_zara.get("product_name", ""),
        "description": df_zara.get("description", ""),
        "product_family": df_zara.get("canonical_family", ""),
        "image": df_zara.get("image", ""),
        "url": df_zara.get("url", ""),
        "color": df_zara.get("colour", ""),
    })
    zara["source"] = "zara"

    # --- Zalando ---
    # df_zalando["product_family"] = df_zalando.apply(infer_product_family_row, axis=1)

    brand_names = df_zalando.get("brand", "").apply(extract_brand_name)
    zalando = pd.DataFrame({
        "product_name": df_zalando.get("product_name", df_zalando.get("name", "")),
        "description": df_zalando.get("description", ""),
        "product_family": df_zalando.get("canonical_family", ""),
        "image": df_zalando.get("main_image", ""),
        "url": df_zalando.get("product_url", df_zalando.get("url", "")),
        "color": df_zalando.get("color", ""),
        "source": brand_names
    })
    
    
    all_products = pd.concat([mango, zalando, zara], ignore_index=True)
    
    df = all_products.copy()
    
    print("Columnas DF final:", df.columns.tolist())
    print(df[["source", "product_name", "url"]].head())
    print("Nulos en 'source':", df["source"].isna().sum())
    print("Vacíos en 'source':", (df["source"].astype(str).str.strip() == "").sum())
    
    # Normalizamos strings
    for col in df.columns:
        df[col] = df[col].apply(safe_str)
        
    print("Columnas DF final:", df.columns.tolist())
    print(df[["source", "product_name", "url"]].head())

    # Documento de texto para embeddings
    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        name = row.get("product_name", "")
        desc = row.get("description", "")
        text = (name + ". " + desc).strip()

        # Si no hay nada de texto, mejor saltar
        if not text:
            continue

        doc_id = str(uuid.uuid4())

        ids.append(doc_id)
        documents.append(text)

        meta = row.to_dict()
        meta["id"] = doc_id
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
        print(f"✅ Insertados {min(end, len(ids))}/{len(ids)}")
        

if "__main__" == __name__:
    
    
    ingest_chroma_zalando_mango_zara()
        

