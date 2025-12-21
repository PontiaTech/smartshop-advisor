from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import pickle, chromadb, numpy as np, traceback, spacy, json
from app.api.utils.language_detection import translate_results

import time
import random
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.observability.logging_config import setup_logger
from app.observability.metrics import REQUEST_COUNT, ERROR_COUNT, REQUEST_LATENCY
from fastapi.responses import Response

from app.api.services.product_search import complete_search
from app.api.schemas import CompleteSearchProduct, CompleteSearchRequest, CompleteSearchResponse, ChatRequest, ChatResponse, WebSearchProduct
from app.api.utils.chat_utils import history_to_text, results_to_bullets, web_results_to_bullets
from app.api.services.web_search import web_search_products
from app.api.ai.llms import get_gemini_llm
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from app.api.ai.system_prompts import CHATBOT_SYSTEM_PROMPT, FULLY_DETAILED_CHATBOT_SYSTEM_PROMPT
from chromadb.errors import NotFoundError
import os

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

#Mapear los logs
logger = setup_logger()

# --- Carga de modelos ---
encoder = SentenceTransformer("clip-ViT-B-32")

with open("classifier_model.pkl", "rb") as f:
    clf = pickle.load(f)

# --- Conexión a Chroma ---
# client = chromadb.HttpClient(host="chroma", port=8000)
# try:
#     # collection = client.get_collection("products")
# except Exception as e:
#     logger.warning(f"No se pudo cargar la collection 'products' en Chroma: {e}")
#     collection = None

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "products_all")

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

def get_chroma_collection():
    """
    Devuelve la colección de Chroma. Si no existe, lanza un error claro
    en vez de romper al importar el módulo.
    """
    try:
        return client.get_collection(CHROMA_COLLECTION)
    except NotFoundError:
        # Aquí podrías hacer create_collection si quieres una vacía:
        # return client.create_collection(CHROMA_COLLECTION)
        raise RuntimeError(
            f"La colección '{CHROMA_COLLECTION}' no existe en Chroma. "
            f"Ejecuta primero el proceso de ingesta (ingest-smartshopadvisor)."
        )

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


# --- Endpoint raíz ---
@app.get("/")
async def index(request: Request):
    start_time = time.time()
    status = 200
    try:
        # Simula latencia variable
        time.sleep(random.uniform(0.1, 0.5))
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            "Solicitud recibida en /log",
            extra={"endpoint": "/", "ip": client_ip, "status": status, "method": request.method}
        )
        return {"message": "API para realizar búsquedas inteligentes de productos de moda en una BBDD vectorial mediante embeddings y clasificación automática. ✅"}
    except Exception as e:
        status = 500
        logger.error(
            "Error en endpoint raíz",
            extra={"endpoint": "/", "error": str(e), "status": status, "method": request.method}
        )
        return JSONResponse({"error": "Error en endpoint raíz"}, status_code=status)
    finally:
        record_request("/", status, start_time, method=request.method)


def record_request(endpoint: str, status: int, start_time: float, method: str = "GET"):
    duration = time.time() - start_time
    # Asegurar que los labels sean strings
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

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

        collection = get_chroma_collection()
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

@app.get("/random_event")
async def random_event(request: Request):
    start_time = time.time()
    rnd = random.random()
    status = 200

    if rnd < 0.7:
        msg = "Evento exitoso"
        logger.info(msg, extra={"endpoint": "/random_event", "type": "info", "status": status, "method": request.method})
    elif rnd < 0.9:
        msg = "Evento crítico simulado"
        logger.error(msg, extra={"endpoint": "/random_event", "type": "CriticalError", "status": 500, "method": request.method})
        ERROR_COUNT.labels(endpoint="/random_event", error_type="CriticalError").inc()
        status = 500
    else:
        msg = "Evento de advertencia simulado"
        logger.warning(msg, extra={"endpoint": "/random_event", "type": "warning", "status": status, "method": request.method})

    record_request("/random_event", status, start_time, method=request.method)
    return JSONResponse({"message": msg, "status": status}, status_code=status)

@app.get("/metrics")
async def metrics():
    """Endpoint para Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



# Pipeline multimodal con combinación de similitud semántica, similitud visual y clasificación de intención de usuario.
@app.post("/complete_search", response_model=CompleteSearchResponse,summary="Búsqueda texto + imagen en productos")
async def complete_search_endpoint(body: CompleteSearchRequest):
    try:
        result = complete_search(body.query, n_results=body.top_k)

        predicted_type = result.get("predicted_type")
        raw_results = result.get("results", [])

        api_results = [
            CompleteSearchProduct(
                product_name=r.get("product_name", ""),
                product_family=r.get("product_family"),
                description=r.get("description"),
                source=r.get("source"),
                url=r.get("url"),
                image=r.get("image"),
                score=float(r.get("clip_score", 0.0)),
            )
            for r in raw_results
        ]

        return CompleteSearchResponse(predicted_type=predicted_type, results=api_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
@app.post("/chat", response_model=ChatResponse, summary="Chatbot usando RAG + Gemini")
async def chat_endpoint(body: ChatRequest):
    try:
        # RAG
        rag = complete_search(body.query, n_results=body.top_k)

        predicted_type = rag.get("predicted_type")
        raw_results = rag.get("results", []) or []

        # score final
        results = [
            CompleteSearchProduct(
                product_name=r.get("product_name", ""),
                product_family=r.get("product_family"),
                description=r.get("description"),
                source=r.get("source"),
                url=r.get("url"),
                image=r.get("image"),
                score=float(r.get("score") or r.get("text_score") or 0.0),
                color=r.get("color", None),
            )
            for r in raw_results
        ]
        
        
        target_lang = (body.target_language or "es").strip().lower()
        llm = get_gemini_llm()
        try:
            results = await translate_results(results=results, llm=llm, target_lang=target_lang)
        except Exception as e:
            logger.warning("translate_results failed (ignored): %s", e)
            
        web_results: list[WebSearchProduct] = []
        web_txt = ""

        try:
            web_items = await web_search_products(body.query, k=3, lang=target_lang)
            web_txt = web_results_to_bullets(web_items)

            web_results = [
                WebSearchProduct(
                    title=(it.get("product_name") or it.get("title") or "").strip(),
                    url=it.get("url"),
                    snippet=(it.get("description") or it.get("snippet") or None),
                    source=it.get("source"),
                )
                for it in (web_items or [])
                if it.get("url")
            ]
        except Exception as e:
            logger.warning("web_search_products failed (ignored): %s", e)
            web_results = []
            web_txt = ""

        hist_txt = history_to_text(body.history)
        limit = min(max(body.top_k, 6), 12)
        products_txt = results_to_bullets(results, limit=limit)
        # web_items = await web_search_products(body.query, k=3, lang=target_lang)
        # web_txt = web_results_to_bullets(web_items)


        prompt = ChatPromptTemplate.from_messages([
            ("system", FULLY_DETAILED_CHATBOT_SYSTEM_PROMPT),
            ("human",
             "Idioma de respuesta: {target_language}\n\n"
             "Historial:\n{history}\n\n"
             "Query:\n{query}\n\n"
             "Resultados disponibles (no inventes nada fuera de esto):\n{products}\n\n"
             "Resultados encontrados en internet (si están vacíos, ignora esta sección):\n{web_products}\n\n"
             "Genera la respuesta."
            )
        ])

        
        chain = prompt | llm
        llm_out = await chain.ainvoke({
            "target_language": target_lang,
            "history": hist_txt or "(vacío)",
            "query": body.query,
            "products": products_txt or "(sin resultados)",
            "web_products": web_txt or "(sin resultados web)",
        })

        answer = getattr(llm_out, "content", None) or str(llm_out)

        return ChatResponse(
            answer=answer.strip(),
            predicted_type=predicted_type,
            results=results,
            web_results=web_results or None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))