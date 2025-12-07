import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from app.api.utils.families_translator import add_canonical_families

EMB_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

MANGO_CSV = "./data/Mango Products Cleaned.csv"
ZARA_CSV = "./data/Zara - Products Cleaned.csv"

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def build_text(row: pd.Series) -> str:
    parts = [
        safe_str(row.get("product_name", "")),
        safe_str(row.get("description", "")),
        safe_str(row.get("product_family", "")),  # le ayuda a agrupar
    ]
    return ". ".join([p for p in parts if p])

def main():
    df_mango = pd.read_csv(MANGO_CSV, encoding="utf-8").fillna("")
    df_zara = pd.read_csv(ZARA_CSV, encoding="utf-8").fillna("")

    df = pd.concat([df_mango, df_zara], ignore_index=True)
    
    df = add_canonical_families(df)

    # Solo filas con product_family no vacío
    df = df[df["canonical_family"].astype(str).str.strip() != ""]

    texts = [build_text(row) for _, row in df.iterrows()]
    labels = df["canonical_family"].astype(str).tolist()

    print(f"Entrenando con {len(texts)} ejemplos y {len(set(labels))} clases.")

    encoder = SentenceTransformer(EMB_MODEL)
    X = encoder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(X, y)

    with open("./data/product_family_clf.pkl", "wb") as f:
        pickle.dump({"clf": clf, "label_encoder": le}, f)

    print("Modelo de product_family guardado en ./data/product_family_clf.pkl")

if __name__ == "__main__":
    main()
