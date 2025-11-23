import os
from typing import List, Dict, Any

import numpy as np
import requests
from chromadb import HttpClient
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products_all")

TEXT_EMB_MODEL = os.getenv(
    "EMB_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")


_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
_collection = _client.get_collection(COLLECTION_NAME)

_text_encoder = SentenceTransformer(TEXT_EMB_MODEL)
_clip_encoder = SentenceTransformer(CLIP_MODEL)

def basic_search(query: str, n_results: int = 10) -> List[Dict[str, Any]]:
    """Primero buscamos por texto en la BDD vectorial."""
    query_emb = _text_encoder.encode(
        [query],
        convert_to_numpy=True,
    ).tolist()

    results = _collection.query(
        query_embeddings=query_emb,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
        
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    items: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        items.append(
            {
                "product_name": meta.get("product_name", ""),
                "product_family": meta.get("product_family", ""),
                "description": meta.get("description", ""),
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
                "image": meta.get("image", ""),
                "distance": float(dist),
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
    
    
    
def complete_search(
    query: str,
    n_results: int = 5,
    candidates_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    De todos los que han salido del filtro inicial por texto, aplicamos un reranking CLIP para sacar de esos solo los más relevantes.
    """
    candidates = basic_search(query, n_results=candidates_k)
    if not candidates:
        return []

    text_emb = _clip_encoder.encode(
        [query],
        convert_to_numpy=True,
    )[0]

    scored: List[Dict[str, Any]] = []

    for c in candidates:
        img_url = c.get("image") or c.get("url")
        if not img_url:
            c["clip_score"] = -1.0
            scored.append(c)
            continue

        img = load_image_from_url(img_url)
        if img is None:
            c["clip_score"] = -1.0
            scored.append(c)
            continue

        img_emb = _clip_encoder.encode( # type: ignore[arg-type]
            img,                 
            convert_to_numpy=True,
        )

        num = float(np.dot(text_emb, img_emb))
        denom = float(np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
        sim = num / denom if denom != 0 else -1.0

        c["clip_score"] = sim
        scored.append(c)

    scored.sort(key=lambda x: x.get("clip_score", -1.0), reverse=True)
    return scored[:n_results]