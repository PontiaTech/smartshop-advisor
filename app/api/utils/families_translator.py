# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# print("Modelos disponibles:")
# for m in genai.list_models():
#     supports = ", ".join(m.supported_generation_methods)
#     print(f"- {m.name}  ->  {supports}")


import os
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import json
import unicodedata

load_dotenv()

#genai.configure(api_key=os.environ["GEMINI_API_KEY"])
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY no está definida en el entorno")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

def build_prompt(families: list[str]) -> str:
    # Construimos una lista numerada para que el modelo no se pierda
    lines = []
    for i, fam in enumerate(families):
        lines.append(f"{i}: {fam}")

    families_block = "\n".join(lines)

    return f"""
Actúa como traductor experto en moda y comercio electrónico.

Te proporciono una lista de nombres de familias de producto de ropa y hogar.
Los textos están en varios idiomas (inglés, francés, árabe, polaco, etc.).

Debes devolver EXCLUSIVAMENTE un JSON válido con el siguiente formato:

[
  {{"id": 0, "raw_family": "...", "family_es": "..."}},
  {{"id": 1, "raw_family": "...", "family_es": "..."}},
  ...
]

Reglas:
- "id" debe coincidir exactamente con el número de la lista que te doy.
- "raw_family" debe copiar el texto original tal cual.
- "family_es" debe ser la traducción natural al español (sin explicaciones extra).
- No añadas texto antes ni después del JSON. Solo el JSON.

Lista de familias:

{families_block}
"""

def main():
    # 1) Cargar únicas
    df = pd.read_csv("./data/raw_families.csv")
    df["raw_family"] = df["raw_family"].astype(str).str.strip()
    families = df["raw_family"].tolist()

    # 2) Construir prompt
    prompt = build_prompt(families)

    # 3) Llamar a Gemini UNA sola vez
    response = model.generate_content(prompt)
    text = (response.text or "").strip()

    # 4) Intentar localizar el JSON en la respuesta
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No se encontró JSON en la respuesta:\n{text}")

    json_str = text[start:end+1]
    data = json.loads(json_str)

    # 5) Convertir a DataFrame
    out_rows = []
    for item in data:
        out_rows.append({
            "id": item["id"],
            "raw_family": item["raw_family"],
            "family_es": item["family_es"],
        })

    df_out = pd.DataFrame(out_rows).sort_values("id")
    df_out = df_out.drop(columns=["id"])

    df_out.to_csv("./data/families_translated.csv", index=False, encoding="utf-8")
    print("Guardado families_translated.csv")
    
# Ahora la parte relativa a con las familias que hemos obtenido definir una nueva columna que haga referencia a la clasificación que nosostros deseamos

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


def norm(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    # arreglar mojibake típico tipo "Ã³" -> "ó"
    repl = {
        "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
        "Ã±": "ñ",
    }
    for bad, good in repl.items():
        t = t.replace(bad, good)
    t = unicodedata.normalize("NFKC", t).lower()
    return t

def infer_canonical_family(family_es: str) -> str:
    t = norm(family_es)

    # 1) Infantil
    if any(w in t for w in ["bebé", "bebe", "prenda exterior bebé", "prendas bebé"]):
        if any(w in t for w in ["zapatos", "botines", "zapatillas", "calzado"]):
            return "Infantil"
        if any(w in t for w in ["gorros", "guantes", "calcetines", "accesorios", "juguetes",
                                "chándales bebé", "cortavientos bebé"]):
            return "Infantil"
        if any(w in t for w in ["pantalones", "leggings", "bermudas", "falda", "petos",
                                "peleles", "prendas bebé"]):
            return "Infantil"
        return "Infantil"

    # 2) Textil hogar
    if any(w in t for w in [
        "funda nórdica", "fundas de almohada",
        "sábana", "sabanas",
        "edredón", "edredones",
        "toallas",
        "alfombras de baño",
        "cojines decorativos",
        "mantas",
        "sábanas y fundas",
        "edredones y almohadas",
        "fundas de cojín",
    ]):
        return "Hogar"

    # 3) Decoración hogar
    if any(w in t for w in [
        "cristalería", "vajillas", "menaje", "cubiertos",
        "objetos decoraci", "marcos de fotos", "portavelas",
        "mantelería", "mobiliario", "hogar",
        "accesorios de cocina", "accesorios de decoración",
    ]):
        return "Hogar"

    # 4) Calzado
    if any(w in t for w in ["zapatos", "botines", "botas", "zapatillas", "calzado"]):
        return "Calzado"

    # 5) Bolsos / marroquinería
    if any(w in t for w in ["bolsos", "carteras", "monederos", "marroquiner", "neceseres"]):
        return "Bolsos"

    # 6) Ropa de baño
    if any(w in t for w in ["trajes de baño", "bikinis", "bañadores", "ropa de baño", "prendas de baño"]):
        return "Ropa de baño"

    # 7) Ropa interior / pijamas / homewear
    if any(w in t for w in [
        "ropa interior",
        "sujetadores",
        "calcetines",
        "batas / ropa de casa",
        "albornoces y batas",
        "albornoz / bata",
        "pijamas",
        "pijamas y bodies",
        "batas ",
        "bata ",
    ]):
        return "Ropa interior"

    # 8) Exterior: abrigos, chaquetas
    if any(w in t for w in [
        "abrigos y chaquetas", "chaquetas de abrigo",
        "prendas de abrigo", "sobrecamisas",
        "cazadoras", "anoraks", "cortavientos",
        "prenda exterior",
    ]):
        return "Abrigos y chaquetas"

    if any(w in t for w in ["abrigo", "abrigos", "chaquetas y sobrecamisas", "chaquetas",
                            "chaquetas y camisas oversize", "novedades en ropa de abrigo"]):
        return "Abrigos y chaquetas"

    # 9) Blazers / americanas
    if any(w in t for w in ["blazers", "americanas", "chaquetas y blazers"]):
        return "Blazers"

    # 10) Partes de abajo (adulto)
    if any(w in t for w in ["pantalones y leggings", "leggings y joggers",
                            "leggings y pantalones deportivos", "joggers"]):
        return "Pantalones"
    if "leggings" in t:
        return "Pantalones"
    if "pantalones" in t:
        return "Pantalones"
    if any(w in t for w in ["shorts", "bermudas"]):
        return "Shorts"
    if "vaqueros" in t or "jeans" in t:
        return "Vaqueros"

    # 11) Vestidos / monos / petos
    if "petos y vestidos" in t or t.startswith("petos "):
        return "Monos y petos"
    if "monos" in t:
        return "Monos y petos"
    if "vestidos y monos" in t:
        return "Monos y petos"
    if "vestidos" in t:
        return "Vestidos"

    # 12) Partes de arriba (adulto)
    if "sudadera" in t or "sudaderas" in t or "chándales" in t:
        return "Sudaderas"

    if any(w in t for w in ["jerséi", "jerséys", "jerséis", "jersé",
                            "cárdigan", "cárdigans", "cárdig", "chalecos"]):
        return "Jerséis y cárdigans"

    if any(w in t for w in ["camisetas deportivas", "tops y camisetas"]):
        return "Camisetas"
    if any(w in t for w in ["camisetas y tops", "camisetas y blusas", "blusas y tops"]):
        return "Camisetas"
    if "tops y otras prendas" in t or t == "tops":
        return "Tops y bodies"
    if "tops " in t and "polos" not in t:
        return "Tops y bodies"
    if "camisetas" in t:
        return "Camisetas"

    if any(w in t for w in ["camisas y blusas", "blusas y camisas", "camisas y camisetas"]):
        return "Camisas y blusas"
    if "camisas" in t or "blusas" in t:
        return "Camisas y blusas"

    # Polos -> los tratamos como camisetas
    if "polos" in t:
        return "Camisetas"

    # 13) Accesorios moda (no hogar)
    if "hogar" not in t and any(w in t for w in [
        "accesorios",
        "gafas de sol",
        "bufandas", "pañuelos",
        "gorros", "guantes",
        "joyería", "bisuter",
        "cinturones", "tirantes",
        "gorros y guantes",
    ]):
        return "Accesorios"

    # 14) Ropa deportiva / running
    if any(w in t for w in ["ropa deportiva", "running"]):
        return "Camisetas"

    # 15) Último recurso
    return ""

def add_canonical_families(df_original: pd.DataFrame) -> pd.DataFrame:
    # df_original = pd.read_csv("./data/.csv", encoding="utf-8")
    df_mapping = pd.read_csv("./data/families_translated.csv", encoding="utf-8", sep=None, engine="python")
    
    df_or = df_original.copy()
    df_or.columns = [c.strip() for c in df_or.columns]

    if "product_family" not in df_or.columns:
        raise ValueError(
            f"df_original no tiene columna 'product_family'. "
            f"Columnas encontradas: {list(df_or.columns)}"
        )
    
    df_new = df_or.merge(df_mapping, on="product_family", how="left")
    df_new["canonical_family"] = df_new["family_es"].apply(infer_canonical_family)
    df_new = df_new.drop(columns=["family_es"])
    
    return df_new

if __name__ == "__main__":
    main()
