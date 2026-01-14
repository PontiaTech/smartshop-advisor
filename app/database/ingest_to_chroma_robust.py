import os
import uuid
import pandas as pd
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import json
import re
from dotenv import load_dotenv

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products_all")

BATCH_SIZE = 2000


def safe_str(x):
    if x is None:
        return ""
    # pandas NA / numpy nan
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    # listas/dicts -> serializa simple
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


# funcion para arreglar formatos raros y caractreses especiales.
def fix_mojibake(s: str) -> str:
    s = safe_str(s)
    if not s:
        return ""
    try:
        repaired = s.encode("latin1").decode("utf-8")
        def letters_count(t: str) -> int:
            return sum(ch.isalpha() for ch in t)
        return repaired if letters_count(repaired) >= letters_count(s) else s
    except Exception:
        return s


def norm_text(s: str) -> str:
    s = safe_str(s).lower()
    s = re.sub(r"[^a-zà-ÿ0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# diccionarios de colores
COLOR_MAP = {
    "black": ["black", "negro", "noir", "schwarz", "nero", "preto", "antracita", "anthracite"],
    "white": ["white", "blanco", "blanc", "weiss", "weiß", "bianco", "branco", "ivory", "marfil", "crema"],
    "beige": ["beige", "camel", "sand", "arena", "ecru", "écru", "crudo", "cream", "taupe"],
    "blue":  ["blue", "azul", "bleu", "blau", "navy", "marino"],
    "red":   ["red", "rojo", "rouge", "rot"],
    "green": ["green", "verde", "vert", "grün"],
    "grey":  ["grey", "gray", "gris", "grau"],
    "brown": ["brown", "marron", "marrón", "braun", "chocolate"],
    "pink":  ["pink", "rosa", "rose", "rosé"],
    "yellow":["yellow", "amarillo", "jaune", "gelb"],
    "orange":["orange", "naranja"],
}

# muchos prodictos vienen con coloes en varios idiomas y esta funcion ayuda a pasrlos todos al mismo idioma
def canonicalize_color(raw_color: str) -> tuple[str, str]:
    raw_clean = norm_text(fix_mojibake(raw_color))
    if not raw_clean:
        return "", ""

    # Match estricto por variantes
    for canon, variants in COLOR_MAP.items():
        for v in variants:
            vv = norm_text(v)
            if vv and re.search(rf"\b{re.escape(vv)}\b", raw_clean):
                return canon, raw_clean

    return "", raw_clean


FAMILY_RULES = {
    "coat":   ["abrigo", "coat", "overcoat", "manteau", "mantel", "cappotto", "parka", "anorak"],
    "jacket": ["chaqueta", "jacket", "jacke", "veste", "blazer"],
    "sweater":["jersey", "sweater", "pullover", "pull", "strick", "knit", "cardigan"],
    "dress":  ["vestido", "dress", "robe", "kleid"],
    "pants":  ["pantalon", "pantalón", "pants", "trousers", "jeans", "hose", "denim"],
    "skirt":  ["falda", "skirt", "jupe", "rock"],
    "shoes":  ["zapato", "zapatos", "shoes", "schuhe", "chaussures", "sneaker", "trainers", "boots", "botas", "stiefel"],
    "scarf":  ["bufanda", "scarf", "echarpe", "écharpe", "schal"],
    "bag":    ["bolso", "bag", "handbag", "sac", "tasche"],
}

def canonicalize_family(name: str, desc: str, raw_family: str) -> str:
    hay = norm_text(f"{fix_mojibake(name)} {fix_mojibake(desc)} {fix_mojibake(raw_family)}")
    if not hay:
        return ""
    for canon, terms in FAMILY_RULES.items():
        for t in terms:
            if re.search(rf"\b{re.escape(t)}\b", hay):
                return canon
    return ""


def zalando_family_from_name(name: str) -> str:
    """
    Intenta sacar la 'familia' del patrón típico:
    "SOMETHING - Trainers - black"
    Devuelve la parte central (Trainers, T-Shirt print, Winterjacke, etc.)
    """
    s = fix_mojibake(safe_str(name))
    if not s:
        return ""

    parts = [p.strip() for p in s.split("-") if p.strip()]
    if len(parts) >= 3:
        return parts[-2]
    if len(parts) == 2:
        return parts[-1]
    return ""


def ingest_chroma_zalando_mango_zara():
    MANGO_CSV = os.getenv("MANGO_CSV", "./data/Mango Products Cleaned.csv")
    ZALANDO_CSV = os.getenv("ZALANDO_CSV", "./data/Zalando products Cleaned.csv")
    ZARA_CSV = os.getenv("ZARA_CSV", "./data/Zara - Products Cleaned.csv")

    EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️ Colección anterior eliminada")
    except Exception:
        print("ℹ️ No existía colección previa")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


    df_mango = pd.read_csv(
        MANGO_CSV,
        sep=";",
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
    ).fillna("")
    df_mango.columns = df_mango.columns.str.strip()

    df_zara = pd.read_csv(ZARA_CSV, encoding="utf-8").fillna("")
    df_zalando = pd.read_csv(ZALANDO_CSV, encoding="utf-8").fillna("")

    # --- Mango ---
    mango = pd.DataFrame({
        "product_name": df_mango.get("product_name", ""),
        "description": df_mango.get("description", ""),
        "family_raw": df_mango.get("product_family", ""),
        "image": df_mango.get("image", ""),
        "url": df_mango.get("url", ""),
        "raw_color": df_mango.get("colour", df_mango.get("color", "")),
    })
    mango["source"] = "mango"

    # --- Zara ---
    zara = pd.DataFrame({
        "product_name": df_zara.get("product_name", ""),
        "description": df_zara.get("description", ""),
        "family_raw": df_zara.get("product_family", ""),
        "image": df_zara.get("image", ""),
        "url": df_zara.get("url", ""),
        "raw_color": df_zara.get("colour", df_zara.get("colour_code", df_zara.get("color", ""))),
    })
    zara["source"] = "zara"

    # --- Zalando ---
    # brand_names = df_zalando.get("brand", "").apply(extract_brand_name)
    zalando = pd.DataFrame({
        "product_name": df_zalando.get("product_name", df_zalando.get("name", "")),
        "description": df_zalando.get("description", ""),
        "family_raw": df_zalando.get("name", "").apply(zalando_family_from_name),  # si existe
        "image": df_zalando.get("main_image", ""),
        "url": df_zalando.get("product_url", df_zalando.get("url", "")),
        "raw_color": df_zalando.get("color", df_zalando.get("colors", "")),
        "source": df_zalando.get("brand_name", "")
    })

    df = pd.concat([mango, zalando, zara], ignore_index=True)

    # # Normalizamos strings
    # for col in df.columns:
    #     df[col] = df[col].apply(safe_str)

    # # Fix mojibake
    # df["product_name"] = df["product_name"].apply(fix_mojibake)
    # df["description"] = df["description"].apply(fix_mojibake)
    # df["product_family"] = df["product_family"].apply(fix_mojibake)
    # df["raw_color"] = df["raw_color"].apply(fix_mojibake)

    # # Canonicaliza familia y color (y sobrescribe product_family para que tu ranking lo use)
    
    # df["canonical_family"] = df.apply(
    #     lambda r: canonicalize_family(
    #         r.get("product_name", ""),
    #         r.get("description", ""),
    #         r.get("product_family", "")
    #     ),
    #     axis=1
    # )

    # # Canonicaliza color: devuelve (canonical_color, raw_color_clean)
    # tmp_color = df.apply(
    #     lambda r: canonicalize_color(
    #         r.get("raw_color", ""),
    #         r.get("product_name", ""),
    #         r.get("description", "")
    #     ),
    #     axis=1
    # )
    # df[["canonical_color", "raw_color_clean"]] = pd.DataFrame(tmp_color.tolist(), index=df.index)

    # # No pises product_family original: crea un campo para búsqueda/ranking
    # df["family_for_search"] = df["canonical_family"].where(
    #     df["canonical_family"].astype(str).str.strip() != "",
    #     df["product_family"]
    # )

    # df["color"] = df["canonical_color"].where(
    #     df["canonical_color"].astype(str).str.strip() != "",
    #     df["raw_color_clean"]
    # )
    
    for col in df.columns:
        df[col] = df[col].apply(safe_str)

    # Fix mojibake donde importa
    for col in ["product_name", "description", "family_raw", "raw_color", "source"]:
        if col in df.columns:
            df[col] = df[col].apply(fix_mojibake)

    # Color: canónico si matchea, si no raw limpio
    tmp_color = df.apply(
        lambda r: canonicalize_color(r.get("raw_color", "")),
        axis=1
    )
    df[["canonical_color", "raw_color_clean"]] = pd.DataFrame(tmp_color.tolist(), index=df.index)
    df["color"] = df["canonical_color"].where(
        df["canonical_color"].astype(str).str.strip() != "",
        df["raw_color_clean"]
    )

    # Familia para búsqueda: por ahora, no inventamos canon
    # Usamos la señal más fiable por fuente: family_raw
    df["family_for_search"] = df["family_raw"]

    print("Vacíos en 'color':", (df["color"].astype(str).str.strip() == "").sum())
    print(df[["raw_color", "raw_color_clean", "canonical_color", "color"]].head(10))

    print("Columnas DF final:", df.columns.tolist())
    print(df[["source", "product_name", "url"]].head())
    print("Vacíos en 'product_family':", (df["family_raw"].astype(str).str.strip() == "").sum())

    # Documento de texto para embeddings
    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        name = row.get("product_name", "")
        desc = row.get("description", "")
        # cf = row.get("canonical_family", "")
        # cc = row.get("canonical_color", "")
        # source = row.get("source", "")

        # text = ". ".join([x for x in [name, desc, cf, cc, source] if safe_str(x)]).strip()
        cf = row.get("family_for_search", "")  # usa la mejor familia disponible
        color_hint = row.get("color", "")      # canon o raw limpio
        source = row.get("source", "")

        text = ". ".join([x for x in [name, desc, cf, color_hint, source] if safe_str(x)]).strip()

        if not text:
            continue

        doc_id = str(uuid.uuid4())
        ids.append(doc_id)
        documents.append(text)

        # meta = row.to_dict()
        # meta["id"] = doc_id
        # metadatas.append(meta)
        meta = {k: safe_str(v) for k, v in row.to_dict().items()}
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


if __name__ == "__main__":
    ingest_chroma_zalando_mango_zara()
