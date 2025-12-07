# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from app.api.utils.product_family_classifier import build_text, EMB_MODEL, pickle

# MODEL_PATH = "./data/product_family_clf.pkl"
# ZALANDO_DATASET = "./data/Zalando products Cleaned.csv"  

# def main():
#     # 1) Cargar dataset nuevo
#     df = pd.read_csv(ZALANDO_DATASET, encoding="utf-8").fillna("")

#     # 2) Construir texto igual que en entrenamiento
#     texts = [build_text(row) for _, row in df.iterrows()]

#     # 3) Cargar encoder y modelo
#     encoder = SentenceTransformer(EMB_MODEL)

#     with open(MODEL_PATH, "rb") as f:
#         saved = pickle.load(f)

#     clf = saved["clf"]
#     le = saved["label_encoder"]

#     # 4) Embeddings
#     X = encoder.encode(
#         texts,
#         batch_size=64,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#     )

#     # 5) Predicciones
#     y_pred = clf.predict(X)
#     y_pred_labels = le.inverse_transform(y_pred)

#     # 6) Añadir columna al mismo dataframe
#     df["canonical_family"] = y_pred_labels

#     # 7) Guardar SOBRE EL MISMO FICHERO
#     df.to_csv(ZALANDO_DATASET, index=False, encoding="utf-8")

#     print(f"Actualizado {ZALANDO_DATASET} con la columna canonical_family")

# if __name__ == "__main__":
#     main()


import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# === Configuración ===
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Fichero de Zalando (ajusta la ruta si hace falta)
ZALANDO_DATASET = "./data/Zalando products Cleaned.csv"

# Tamaño de batch para cada llamada a Gemini
BATCH_SIZE = 100  # puedes subir/bajar según veas

# Categorías canónicas generales
CANONICAL_CATEGORIES = [
    "Camisetas",
    "Camisas y blusas",
    "Jerséis y cárdigans",
    "Sudaderas",
    "Tops y bodies",
    "Pantalones",
    "Vaqueros",
    "Shorts",
    "Vestidos",
    "Monos y petos",
    "Abrigos y chaquetas",
    "Blazers",
    "Ropa interior",
    "Ropa de baño",
    "Calzado",
    "Bolsos",
    "Accesorios",
    "Hogar",
    "Infantil",
]

# Para Zalando, si Gemini devuelve algo raro o de hogar/infantil,
# lo mapeamos a "Accesorios"
ZALANDO_ALLOWED = set(CANONICAL_CATEGORIES)  # las aceptamos todas inicialmente


def build_product_text(row: pd.Series) -> str:
    """
    Construye el texto que verá Gemini.
    Intenta usar product_name, name y description si existen.
    """
    parts = []

    for col in ["product_name", "name", "description"]:
        if col in row:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                parts.append(val)

    text = " - ".join(parts)
    return text.strip()


def build_prompt(batch_rows: list[tuple[int, str]]) -> str:
    """
    batch_rows: lista de tuplas (idx_global, texto_producto)
    """
    cats_str = ", ".join(f'"{c}"' for c in CANONICAL_CATEGORIES)

    lines = []
    for idx, desc in batch_rows:
        # recortamos descripción por si es extremadamente larga
        desc_short = desc[:800]
        lines.append(f"{idx}: {desc_short}")

    desc_block = "\n".join(lines)

    prompt = f"""
        Actúa como un clasificador experto de productos de moda y hogar para un e-commerce.

        Te doy varias DESCRIPCIONES de producto (nombre + detalles) en distintos idiomas.

        Debes asignar a cada producto UNA ÚNICA categoría EXACTA de la siguiente lista:

        [{cats_str}]

        Reglas:
        - Devuelve SOLO un JSON válido (sin texto adicional) con el formato:
        [
        {{"idx": 0, "canonical_family": "..."}},
        {{"idx": 1, "canonical_family": "..."}},
        ...
        ]
        - "idx" es el índice que aparece delante de cada descripción.
        - "canonical_family" debe ser EXACTAMENTE una de las categorías de la lista.
        - Si el tipo de producto no es claramente de ropa, calzado o bolsos,
        o tienes dudas, clasifícalo como "Accesorios".
        - No inventes categorías nuevas.

        Descripciones:

        {desc_block}
        """
    return prompt


def main():
    df = pd.read_csv(ZALANDO_DATASET, encoding="utf-8").fillna("")

    n = len(df)
    print(f"Número de productos Zalando: {n}")

    texts = [build_product_text(row) for _, row in df.iterrows()]

    canonical = [None] * n

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = [(i, texts[i]) for i in range(start, end)]

        prompt = build_prompt(batch)
        print(f"Procesando batch {start}–{end-1}...")

        try:
            resp = model.generate_content(prompt)
        except Exception as e:
            print(f"Error llamando a Gemini en batch {start}–{end}: {e}")
            continue

        text = (resp.text or "").strip()

        # Buscar JSON dentro de la respuesta
        s = text.find("[")
        e = text.rfind("]")
        if s == -1 or e == -1:
            print("No se encontró JSON en la respuesta de Gemini. Respuesta cruda:")
            print(text)
            continue

        try:
            data = json.loads(text[s:e+1])
        except json.JSONDecodeError as e:
            print("Error haciendo json.loads del batch. Respuesta recortada:")
            print(text[:500])
            print("Error:", e)
            continue

        # Rellenar predicciones
        for item in data:
            idx = item.get("idx")
            fam = item.get("canonical_family", "").strip()

            if idx is None:
                continue

            # Normalizamos el valor
            if fam not in ZALANDO_ALLOWED:
                # Si no está en nuestras categorías, o es algo raro,
                # lo mandamos a "Accesorios"
                fam = "Accesorios"

            # Opcional: si quieres forzar que Hogar/Infantil no salgan en Zalando:
            if fam in ("Hogar", "Infantil"):
                fam = "Accesorios"

            canonical[idx] = fam

        # Espera suave entre llamadas
        time.sleep(0.5)

    # Fallback: lo que haya quedado sin asignar -> Accesorios
    for i in range(n):
        if not canonical[i]:
            canonical[i] = "Accesorios"

    df["canonical_family"] = canonical

    # Guardar SOBRE el mismo CSV
    df.to_csv(ZALANDO_DATASET, index=False, encoding="utf-8")
    print(f"Actualizado {ZALANDO_DATASET} con la columna canonical_family")


if __name__ == "__main__":
    main()

