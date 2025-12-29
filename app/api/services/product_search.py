import os
from typing import List, Dict, Any
import re
import numpy as np
import requests
from chromadb import HttpClient
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from app.api.services.classifier import predict_article_type
from app.database.ingest_to_chroma_robust import fix_mojibake

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products_all")

TEXT_EMB_MODEL = os.getenv(
    "EMB_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")

STOP = {"de","la","el","y","para","con","sin","un","una","unos","unas","que","en","por","del","al"}

TYPE_TO_CANON = {
    "coat": "coat",
    "jacket": "jacket",
    "sweater": "sweater",
    "dress": "dress",
    "pants": "pants",
    "skirt": "skirt",
    "shoes": "shoes",
    "scarf": "scarf",
    "bag": "bag",
    # si tu clasificador usa otras etiquetas, añádelas aquí
}

# Sinónimos/variantes multilenguaje (rápido y práctico)
FAMILY_SYNONYMS = {
    "coat": [
        "abrigo", "coat", "overcoat", "manteau", "mantel", "cappotto",
        "winter coat", "daunenmantel", "mantele", "mantello"
    ],
    "jacket": [
        "chaqueta", "jacket", "jacke", "veste", "giacca",
        "winterjacke", "puffer", "puffer jacket", "anorak", "parka", "blouson"
    ],
    "sweater": [
        "jersey", "sweater", "pullover", "strickpullover", "strick", "knit",
        "cardigan", "jumper"
    ],
    "dress": [
        "vestido", "dress", "robe", "kleid", "abito"
    ],
    "pants": [
        "pantalon", "pantalón", "pants", "trousers", "hose", "jeans",
        "stoffhose", "chino", "leggings"
    ],
    "skirt": [
        "falda", "skirt", "jupe", "rock", "gonna"
    ],
    "shoes": [
        "zapato", "zapatos", "shoes", "schuhe", "chaussures",
        "sneaker", "sneakers", "trainers", "boot", "boots", "stiefel",
        "sandals", "pumps", "loafers", "mules"
    ],
    "scarf": [
        "bufanda", "scarf", "écharpe", "echarpe", "schal"
    ],
    "bag": [
        "bolso", "bag", "handbag", "tasche", "sac", "umhängetasche", "rucksack"
    ],
}


client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(COLLECTION_NAME)

text_encoder = SentenceTransformer(TEXT_EMB_MODEL)
clip_encoder = SentenceTransformer(CLIP_MODEL)


def _norm(s: str) -> str:
    s = fix_mojibake(s or "")
    s = s.lower()
    s = re.sub(r"[^a-zà-ÿ0-9\s/-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _contains_any(text: str, terms: list[str]) -> bool:
    if not text:
        return False
    for t in terms:
        tt = _norm(t)
        if not tt:
            continue
        if re.search(rf"\b{re.escape(tt)}\b", text):
            return True
    return False

def basic_search(query: str, n_results: int = 10) -> List[Dict[str, Any]]:
    """Primero buscamos por texto en la BDD vectorial."""
    query_emb = text_encoder.encode(
        [query],
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
        
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    items: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        
        text_score = 1.0 / (1.0 + dist)
        dist = float(dist)
        
        # items.append(
        #     {
        #         "product_name": meta.get("product_name", ""),
        #         "product_family": meta.get("product_family", ""),
        #         "description": meta.get("description", ""),
        #         "source": meta.get("source", ""),
        #         "url": meta.get("url", ""),
        #         "image": meta.get("image", ""),
        #         "distance": float(dist),
        #         "text_score": text_score,
        #         "raw_document": doc,
        #         "raw_metadata": meta,
        #     }
        # )
        pf = (
            meta.get("family_for_search")
            or meta.get("family_raw")
            or meta.get("product_family", "")
        )

        color = (
            meta.get("color")
            or meta.get("raw_color_clean")
            or meta.get("raw_color", "")
        )

        items.append(
            {
                "product_name": meta.get("product_name", ""),
                "product_family": pf,
                "description": meta.get("description", ""),
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
                "image": meta.get("image", ""),
                "color": color,
                "distance": float(dist),
                "text_score": text_score,
                "raw_document": doc,
                "raw_metadata": meta,
            }
        )
    return items


def load_image_from_url(url: str) -> Image.Image | None:
    """Descarga una imagen desde URL y la convierte a RGB."""
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None
    

def category_boost(predicted_type: str | None, product_family: str | None, product_name: str | None = None, description: str | None = None,) -> float:
    """
    Bonus sencillo según si la familia del producto encaja con el tipo predicho.
    Heurística muy simple: coincidencia por substring case-insensitive.
    """
    if not predicted_type:
        return 0.0

    canon = TYPE_TO_CANON.get(str(predicted_type).lower(), "")
    if not canon:
        return 0.0

    synonyms = FAMILY_SYNONYMS.get(canon, [])
    fam_text = _norm(product_family or "")
    name_text = _norm(product_name or "")
    desc_text = _norm(description or "")
    combined = (name_text + " " + desc_text).strip()

    # 1) Señal fuerte: match en product_family/family_raw
    if _contains_any(fam_text, synonyms):
        return 1.0

    # 2) Señal media: match en nombre+descripción
    if _contains_any(combined, synonyms):
        return 0.5

    return 0.0

def compute_imp_cat_eff(predicted_type: str, candidates: list[dict], imp_cat: float) -> float:
    if imp_cat <= 0 or not predicted_type:
        return 0.0

    top = candidates[:10] if len(candidates) >= 10 else candidates
    if not top:
        return 0.0

    boosts = []
    for c in top:
        boosts.append(
            category_boost(
                predicted_type=predicted_type,
                product_family=c.get("product_family"),
                product_name=c.get("product_name"),
                description=c.get("description"),
            )
        )

    # Consenso: cuántos tienen evidencia (>=0.5)
    strongish = sum(1 for b in boosts if b >= 0.5)
    ratio = strongish / len(boosts)

    return imp_cat if ratio >= 0.30 else 0.0


def extract_keywords(query: str) -> list[str]:
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", (query or "").lower())
    words = [w for w in words if len(w) >= 4 and w not in STOP]
    return words[:6]

def keyword_match_score(query: str, product_text: str) -> float:
    kws = extract_keywords(query)
    if not kws:
        return 1.0
    t = (product_text or "").lower()
    hits = sum(1 for k in kws if k in t)
    return hits / len(kws)
  
    
def complete_search(query: str, n_results: int = 5, candidates_k: int = 20, imp_text: float = 0.45, imp_image: float = 0.5, imp_cat: float = 0.05) -> dict:
    """
        Búsqueda completa:
        1) Clasifica la intención (tipo de producto).
        2) Recupera candidatos por texto desde Chroma.
        3) Reordena usando CLIP sobre imágenes.
        4) Combina texto + imagen + tipo de producto en un score final.
    """
    
    predicted_type = predict_article_type(query)

    candidates = basic_search(query, n_results=candidates_k)
    if not candidates:
        return {"predicted_type": predicted_type, "results": []}

    # Ajusta pesos efectivos de categoría en función de si hay señal real en candidatos
    imp_cat_eff = compute_imp_cat_eff(predicted_type, candidates, imp_cat)

    # Embedding del texto en espacio CLIP
    text_emb = clip_encoder.encode([query], convert_to_numpy=True)[0]

    scored: List[Dict[str, Any]] = []
    clip_available = 0
    
    query_kws = extract_keywords(query)

    for c in candidates:
        img_url = c.get("image")
        text_score = float(c.get("text_score", 0.0))

        # OJO: con la nueva ingesta, basic_search debería rellenar "product_family"
        # desde family_for_search/family_raw (y también "color" si lo añadiste).
        product_family = c.get("product_family")
        cat_boost = category_boost(
            predicted_type=predicted_type,
            product_family=product_family,
            product_name=c.get("product_name"),
            description=c.get("description"),
        )

        # CLIP score (si hay imagen descargable)
        clip_score = -1.0
        if img_url:
            img = load_image_from_url(img_url)
            if img is not None:
                img_emb = clip_encoder.encode(img, convert_to_numpy=True)  # type: ignore[arg-type]
                num = float(np.dot(text_emb, img_emb))
                denom = float(np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
                clip_score = num / denom if denom != 0 else -1.0

        # Normalizamos sim a [0,1]
        clip_sim = (clip_score + 1.0) / 2.0

        if clip_score != -1.0:
            clip_available += 1

        c["clip_score"] = clip_score
        c["clip_sim"] = clip_sim
        c["category_boost"] = cat_boost

        # Keyword match: mezcla name/desc/familia/color/source para capturar señales literales
        kw_text = " ".join([
            c.get("product_name", ""),
            c.get("description", ""),
            c.get("product_family", ""),
            c.get("color", ""),
            c.get("source", ""),
        ])
        kw = float(keyword_match_score(query, kw_text))
        c["kw_score"] = kw

        # Score base
        base = imp_text * text_score + imp_image * clip_sim + imp_cat_eff * cat_boost

        # Mezcla final (keyword aporta un empujón pequeño pero útil)
        c["score"] = 0.85 * base + 0.15 * kw
        
        fam = _norm(c.get("product_family", ""))
        desc = (c.get("description") or "").strip()
        name = (c.get("product_name") or "").strip()

        quality_penalty = 0.0

        # family vacía o muy poco informativa
        if not fam or fam in {"", "n/a", "na", "none"}:
            quality_penalty += 0.05

        # descripción demasiado corta (suele ser mala señal)
        if len(desc) < 25:
            quality_penalty += 0.05

        # descripción casi igual al nombre (ingesta repetida / poco informativa)
        if _norm(desc) and _norm(name) and _norm(desc)[:60] == _norm(name)[:60]:
            quality_penalty += 0.05

        # si keyword match es bajo, penaliza extra (evita "botas" en "vaqueros negros")
        if kw < 0.34:
            quality_penalty += 0.08
        elif kw < 0.50:
            quality_penalty += 0.04

        c["quality_penalty"] = quality_penalty
        c["score"] = max(0.0, float(c["score"]) - quality_penalty)

        scored.append(c)

    # Si NO hay ninguna imagen usable, recalcula score sin el componente de imagen
    if clip_available == 0:
        for c in scored:
            text_score = float(c.get("text_score", 0.0))
            cat_boost = float(c.get("category_boost", 0.0))
            kw = float(c.get("kw_score", 0.0))

            base = imp_text * text_score + imp_cat_eff * cat_boost
            c["clip_score"] = -1.0
            c["clip_sim"] = 0.0
            c["score"] = 0.9 * base + 0.1 * kw

    # Ordenamos por score final descendente
    scored.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    
    base_thr = 0.68

    # si la query trae señal clara, sube el umbral (más precisión)
    if len(query_kws) >= 3:
        base_thr += 0.05

    # si la query es corta tipo "y de deporte?", baja para no quedarte sin nada
    if len(query.split()) <= 3:
        base_thr -= 0.05

    # si hay un ganador claro, el resto suele ser ruido -> sube el listón
    if len(scored) >= 3:
        gap = float(scored[0].get("score", 0.0)) - float(scored[2].get("score", 0.0))
        if gap > 0.12:
            base_thr += 0.04

    # clamp seguro
    base_thr = min(max(base_thr, 0.55), 0.80)

    # Filtro opcional para evitar basura (mantengo tu lógica)
    filtered = [x for x in scored if float(x.get("score", 0.0) or 0.0) >= 0.6]
    top = (filtered[:n_results]) if filtered else (scored[:n_results])

    return {"predicted_type": predicted_type, "results": top}