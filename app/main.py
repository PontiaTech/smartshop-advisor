from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from PIL import Image
import pickle
import io
import chromadb
import numpy as np
from typing import Optional

# --- Inicializar FastAPI ---
app = FastAPI(title="SmartShop Search API", version="1.0")

# --- Cargar modelo y encoder ---
print("Cargando modelo de clasificación y Sentence-BERT...")
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
with open("classifier_model.pkl", "rb") as f:
    clf = pickle.load(f)

# --- Inicializar cliente Chroma ---
client = chromadb.HttpClient(host="chroma", port=8000)
collection = client.get_collection("products")

# --- Modelos de entrada ---
class SearchInput(BaseModel):
    query: Optional[str] = None
    image_url: Optional[str] = None  # si prefieres subir, lo cambiamos por UploadFile


# --- Funciones auxiliares ---
def predict_article_type(query: str) -> str:
    emb = sbert.encode([query])
    pred = clf.predict(emb)[0]
    return pred

def get_text_embedding(query: str) -> List[float]:
    return sbert.encode([query])[0].tolist()

def get_image_embedding(image_bytes: bytes) -> List[float]:
    # Usar CLIP si quieres algo real. Aquí placeholder:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_emb = np.random.rand(384).tolist()  # placeholder temporal
    return img_emb

def query_chroma(embedding: List[float], article_type: str, n_results: int = 3):
    # Filtramos por label del modelo de clasificación
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where={"articleType": article_type}  # filtra por tipo predicho
    )
    return results


from fastapi import Body, UploadFile, File

# --- ENDPOINTS ---
@app.post("/search")
async def search(
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Endpoint multimodal (texto + imagen)
    """
    if not query and not image:
        return {"error": "Debe proporcionar texto o imagen."}

    # Si hay texto → predecir tipo de artículo
    if query:
        article_type = predict_article_type(query)
        text_emb = get_text_embedding(query)
    else:
        article_type = "Other"
        text_emb = None

    # Si hay imagen → generar embedding
    image_embedding = None
    if image:
        image_bytes = await image.read()
        image_embedding = get_image_embedding(image_bytes)

    # Combinar embeddings
    if text_emb and image_embedding:
        combined_emb = [(t + i)/2 for t,i in zip(text_emb, image_embedding)]
    elif text_emb:
        combined_emb = text_emb
    elif image_embedding:
        combined_emb = image_embedding

    # Consultar Chroma
    results = query_chroma(combined_emb, article_type)

    # Formatear salida
    hits = []
    for doc, meta, dist in zip(results["documents"][0],
                               results["metadatas"][0],
                               results["distances"][0]):
        hits.append({"name": doc, "metadata": meta, "similarity": float(dist)})

    return {"predicted_article_type": article_type, "top_results": hits}