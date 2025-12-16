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


client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(COLLECTION_NAME)

text_encoder = SentenceTransformer(TEXT_EMB_MODEL)
clip_encoder = SentenceTransformer(CLIP_MODEL)


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
        
        items.append(
            {
                "product_name": meta.get("product_name", ""),
                "product_family": meta.get("product_family", ""),
                "description": meta.get("description", ""),
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
                "image": meta.get("image", ""),
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
    

def category_boost(predicted_type: str | None, product_family: str | None) -> float:
    """
    Bonus sencillo según si la familia del producto encaja con el tipo predicho.
    Heurística muy simple: coincidencia por substring case-insensitive.
    """
    if not predicted_type or not product_family:
        return 0.0

    pt = str(predicted_type).lower()
    pf = str(product_family).lower()

    if pt in pf or pf in pt:
        return 1.0

    return 0.0

def compute_imp_cat_eff(predicted_type: str, candidates: list[dict], imp_cat: float) -> float:
    if imp_cat <= 0:
        return 0.0
    boosts = []
    for c in candidates[:10]:
        boosts.append(category_boost(predicted_type, c.get("product_family")))
    avg = sum(boosts) / max(len(boosts), 1)
    return imp_cat if avg >= 0.3 else 0.0


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
  
    
def complete_search(query: str, n_results: int = 5, candidates_k: int = 20, imp_text: float = 0.4, imp_image: float = 0.5, imp_cat: float = 0.1) -> dict:
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
    
    imp_cat_eff = compute_imp_cat_eff(predicted_type, candidates, imp_cat)

    # Embedding de texto en espacio CLIP
    text_emb = clip_encoder.encode(
        [query],
        convert_to_numpy=True,
    )[0]

    scored: List[Dict[str, Any]] = []
    clip_available = 0

    # for c in candidates:
    #     img_url = c.get("image")
    #     text_score = float(c.get("text_score", 0.0))
        
    #     # vemos a ver si la categoría que se ha predicho encaja con la familia del producto
    #     product_family = c.get("product_family")
    #     cat_boost = category_boost(predicted_type, product_family)

    #     if not img_url:
    #         clip_score = -1.0
    #     else:
    #         img = load_image_from_url(img_url)
    #         if img is None:
    #             clip_score = -1.0
    #         else:
    #             img_emb = clip_encoder.encode(  # type: ignore[arg-type]
    #                 img,
    #                 convert_to_numpy=True,
    #             )
    #             num = float(np.dot(text_emb, img_emb))
    #             denom = float(np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
    #             clip_score = num / denom if denom != 0 else -1.0


    #     # Normalizamos sim a [0,1]
    #     clip_sim = (clip_score + 1.0) / 2.0

    #     c["clip_score"] = clip_score
    #     c["category_boost"] = cat_boost

    #     final_score = imp_text * text_score + imp_image * clip_sim + imp_cat_eff * cat_boost

    #     c["score"] = final_score

    #     scored.append(c)
        
    
    

    # # Ordenamos por score final descendente
    # scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    # top = scored[:n_results]
    # return {
    #     "predicted_type": predicted_type,
    #     "results": top,
    # }
    
    for c in candidates:
        img_url = c.get("image")
        text_score = float(c.get("text_score", 0.0))

        product_family = c.get("product_family")
        cat_boost = category_boost(predicted_type, product_family)

        clip_score = -1.0
        if img_url:
            img = load_image_from_url(img_url)
            if img is not None:
                img_emb = clip_encoder.encode(img, convert_to_numpy=True)  # type: ignore[arg-type]
                num = float(np.dot(text_emb, img_emb))
                denom = float(np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
                clip_score = num / denom if denom != 0 else -1.0

        clip_sim = (clip_score + 1.0) / 2.0
        c["clip_score"] = clip_score
        c["clip_sim"] = clip_sim 
        c["category_boost"] = cat_boost

        if clip_score != -1.0:
            clip_available += 1 

        c["score"] = imp_text * text_score + imp_image * clip_sim + imp_cat_eff * cat_boost
        scored.append(c)

    if clip_available == 0:
        for c in scored:
            text_score = float(c.get("text_score", 0.0))
            cat_boost = float(c.get("category_boost", 0.0))
            c["score"] = imp_text * text_score + imp_cat_eff * cat_boost
            c["clip_score"] = -1.0
            c["clip_sim"] = 0.0

    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    filtered = [x for x in scored if float(x.get("score", 0.0) or 0.0) >= 0.6] # para que saque solo las score mayores a 0.6 yeliminar cosas raras que salen
    top = (filtered[:n_results]) if filtered else (scored[:n_results])

    return {"predicted_type": predicted_type, "results": top}