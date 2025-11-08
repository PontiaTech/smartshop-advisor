from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import pickle, chromadb, numpy as np, traceback, spacy, json

app = FastAPI(
    title="🧠 API - SmartShop Advisor",
    version="1.0",
    description="""
    API para realizar búsquedas inteligentes de productos de moda en una BBDD vectorial mediante embeddings y clasificación automática.

    Funcionalidades principales:
    - Genera embeddings de texto con CLIP.
    - Clasifica el tipo de producto.
    - Busca en ChromaDB los productos más similares.
    - Soporta consultas encadenadas usando contexto previo.
    """,
    contact={
        "name": "Equipo SmartShop Advisor",
        "email": "soporte@smartshop.com",
    },
)

# --- Carga de modelos ---
encoder = SentenceTransformer("clip-ViT-B-32")

with open("classifier_model.pkl", "rb") as f:
    clf = pickle.load(f)

# --- Conexión a Chroma ---
client = chromadb.HttpClient(host="chroma", port=8000)
collection = client.get_collection("products")

# --- Variables globales ---
last_article_type = None
last_embeddings = []
last_query = ""

# --- NLP ---
nlp = spacy.load("en_core_web_sm")


# --- Funciones auxiliares ---
def predecir_tipo_articulo(query: str) -> str:
    """Predice el tipo de producto usando el clasificador entrenado."""
    return clf.predict(encoder.encode([query]))[0]


def generar_embedding(q: str) -> list:
    """Convierte el texto en un vector de embedding."""
    emb = encoder.encode([q], convert_to_numpy=True)[0]
    return emb.tolist()


def tiene_sujeto(texto: str) -> bool:
    """Detecta si el texto contiene un sujeto (para saber si es consulta nueva o seguimiento)."""
    doc = nlp(texto)
    return any(token.dep_ in ("nsubj", "nsubj_pass") for token in doc)


# --- Modelos de entrada y salida ---
class PeticionBusqueda(BaseModel):
    query: str = Field(..., description="Texto que describe el producto o necesidad del usuario.")
    history: Optional[List[str]] = Field(default_factory=list, description="Historial opcional de mensajes previos.")


class ResultadoProducto(BaseModel):
    nombre: str = Field(..., description="Nombre o identificador del producto.")
    metadatos: dict = Field(..., description="Metadatos asociados al producto.")
    similitud: float = Field(..., description="Nivel de similitud entre 0 y 1.")
    enlace: str = Field(..., description="Enlace al producto o página relacionada.")


class RespuestaBusqueda(BaseModel):
    tipo_predicho: str = Field(..., description="Tipo de producto detectado por el modelo.")
    resultados: List[ResultadoProducto] = Field(..., description="Productos más similares encontrados.")


# --- Endpoint principal ---
@app.post(
    "/search",
    response_model=RespuestaBusqueda,
    summary="Buscar productos similares",
    description="""
    Endpoint principal de la API de búsqueda.  
    Toma una consulta de texto, predice el tipo de producto y devuelve los resultados más similares
    de la base de datos de Chroma.

    - Si la consulta no tiene sujeto, se interpreta como **seguimiento de la búsqueda anterior**.  
    - Si tiene sujeto, se considera una **nueva búsqueda**.
    """,
)
async def search(
    request: Request,
    query: Optional[str] = Form(None, description="Texto de la consulta (modo FormData o Swagger)."),
    history: Optional[str] = Form("[]", description="Historial previo en formato JSON."),
):
    """
    Compatible con:
    - Swagger / Gradio → usan multipart/form-data.
    - Postman u otros → pueden enviar JSON.
    """
    global last_article_type, last_query, last_embeddings

    try:
        # Intentar leer JSON directamente si viene en el cuerpo
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            query = data.get("query")
            history = data.get("history", [])
        else:
            # Si viene por form-data (Swagger o Gradio)
            history = json.loads(history) if history else []

        if not query:
            raise HTTPException(status_code=400, detail="Falta el parámetro 'query'.")

        # --- Embedding ---
        emb_texto = generar_embedding(query)
        print(f"📦 Longitud del embedding: {len(emb_texto)}")

        if last_embeddings and not tiene_sujeto(query):
            print("No se detectó sujeto → combinando con el embedding anterior.")
            comb_emb = 0.8 * np.array(emb_texto) + 0.2 * np.array(last_embeddings)
        else:
            print("Se detectó sujeto → nueva búsqueda.")
            comb_emb = emb_texto
            last_embeddings = emb_texto

        # --- Clasificación ---
        if last_query and not tiene_sujeto(query):
            consulta_completa = f"{last_query}. {query}"
            articulo = predecir_tipo_articulo(consulta_completa)
            print(f"Consulta combinada con la anterior: {consulta_completa}")
        else:
            articulo = predecir_tipo_articulo(query)
            last_query = query
        print(f"🧠 Tipo predicho: {articulo}")

        # --- Búsqueda en Chroma ---
        filtros = [articulo]
        if last_article_type and last_article_type != articulo:
            filtros.append(last_article_type)
            print("También se usará el tipo anterior:", last_article_type)

        last_article_type = articulo

        r = collection.query(
            query_embeddings=[comb_emb],
            n_results=3,
            where={"articleType": {"$in": filtros}},
        )

        resultados = [
            ResultadoProducto(
                nombre=doc,
                metadatos=meta,
                similitud=float(1 - dist),
                enlace=f"https://{doc.replace(' ', '_')}.com",
            )
            for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            )
        ]

        return RespuestaBusqueda(tipo_predicho=articulo, resultados=resultados)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
