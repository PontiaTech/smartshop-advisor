from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer
import pickle, chromadb, numpy as np, traceback, spacy, json
from app.api.utils.language_detection import translate_results
from app.api.ai.results_regulator import compact_products_for_judge, pick_by_indices

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
from app.api.ai.llms import get_llm
# from app.api.utils.search_context_improved import build_search_query, sanitize_web_query
from app.api.utils.search_context import build_search_query, sanitize_web_query, sanitize_rag_query
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from app.api.ai.system_prompts import CHATBOT_SYSTEM_PROMPT, FULLY_DETAILED_CHATBOT_SYSTEM_PROMPT
from chromadb.errors import NotFoundError
import os
import re
from datetime import datetime, timezone


def _preview(s: str, n: int = 140) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[:n]

def _history_tail_plain(history, n: int = 6):
    out = []
    for m in (history or [])[-n:]:
        sender = (getattr(m, "sender", "") or "")
        content = getattr(m, "content", None)
        txt = content if isinstance(content, str) else ""
        out.append(f"{sender}:{_preview(txt, 120)}")
    return out

def _append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def looks_like_followup(q: str) -> bool:
        ql = (q or "").strip().lower()
        starters = (
            "y ", "y que", "que sean", "también", "pero", "más", "mejor", "de ", "con ",
            "y si", "y en", "y para", "y del", "y de",
        )
        return (len(ql) < 40 and ql.startswith(starters)) or ql in {"sí", "vale", "ok", "perfecto"}

def _get_msg_role_and_text(msg: Any) -> tuple[str, str]:
        # Acepta dict {"role": "...", "content": "..."} o tuplas (role, content)
        if isinstance(msg, dict):
            role = (msg.get("role") or msg.get("type") or "").strip().lower()
            text = (msg.get("content") or msg.get("text") or msg.get("message") or "").strip()
            return role, text
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            role = str(msg[0] or "").strip().lower()
            text = str(msg[1] or "").strip()
            return role, text
        return "", str(msg or "").strip()

def last_user_utterance(history: Any) -> str:
        if not history:
            return ""
        for m in reversed(history):
            role, text = _get_msg_role_and_text(m)
            if role in {"user", "human"} and text:
                return text
        # si no hay roles, prueba el último texto “usable”
        for m in reversed(history):
            _, text = _get_msg_role_and_text(m)
            if text:
                return text
        return ""

def extract_json_obj(text: str) -> dict:
        """
        Extrae el primer objeto JSON {...} de una respuesta que puede traer texto alrededor.
        """
        s = (text or "").strip()
        if not s:
            raise ValueError("Empty judge output")

        # caso ideal: es JSON puro
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)

        # intenta extraer bloque {...}
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in judge output")
        return json.loads(m.group(0))
    
def history_last_user_query(history) -> str:
    if not history:
        return ""
    for m in reversed(history):
        # dict
        if isinstance(m, dict):
            role = (m.get("sender") or m.get("role") or "").lower()
            content = (m.get("content") or m.get("text") or "").strip()
            if role in {"user", "human"} and content:
                return content
        # tupla/lista
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            role = str(m[0] or "").lower()
            content = str(m[1] or "").strip()
            if role in {"user", "human"} and content:
                return content
        else:
            # si no sabemos, lo ignoramos
            continue
    return ""
    
    

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


# @app.post("/chat", response_model=ChatResponse, summary="Chatbot usando RAG + Ollama")
# async def chat_endpoint(body: ChatRequest):
#     try:
#         logger.info(
#             "Chat request received",
#             extra={
#                 "endpoint": "/chat",
#                 "query": body.query,
#                 "history_len": len(body.history or []),
#                 "method": "POST",
#             },
#         )

#         # Sacamos el contexto del chat
#         search_query, used_ctx = build_search_query(body.query, body.history)

#         logger.info(
#             "Search context resolved",
#             extra={
#                 "endpoint": "/chat",
#                 "original_query": body.query,
#                 "search_query": search_query,
#                 "used_ctx": used_ctx,
#             },
#         )

#         target_lang = (body.target_language or "es").strip().lower()
#         llm = get_llm()

#         # RAG
#         rag = complete_search(search_query, n_results=body.top_k)
#         predicted_type = rag.get("predicted_type")
#         raw_results = rag.get("results", []) or []

#         # --- REGULADOR (LLM) para separar best/similar/ruido ---
#         judge_prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "Eres un validador estricto de relevancia para un buscador de productos.\n"
#                     "Evalúas productos del catálogo interno frente a una consulta.\n"
#                     "Devuelve SOLO JSON válido. No añadas texto fuera del JSON.",
#                 ),
#                 (
#                     "human",
#                     "Consulta del usuario:\n{query}\n\n"
#                     "Productos candidatos (cada línea empieza por su índice i):\n{items}\n\n"
#                     "Clasifica cada producto en UNA categoría:\n"
#                     "- internal_best: encaja claramente con la consulta\n"
#                     "- internal_similar: se parece pero no cumple totalmente\n"
#                     "- discard: no es relevante\n\n"
#                     "Reglas:\n"
#                     "- Usa solo índices que EXISTAN en la lista.\n"
#                     "- Sé conservador: si dudas, usa internal_similar.\n\n"
#                     "Devuelve JSON EXACTO con este formato:\n"
#                     "Devuelve JSON EXACTO:\n"
#                     "{{\n"
#                     '  "internal_best":[1],\n'
#                     '  "internal_similar":[2,3],\n'
#                     '  "discard":[4,5],\n'
#                     '  "note":"1 frase breve"\n'
#                     "}}"
#                 ),
#             ]
#         )

#         judge = {"internal_best": [], "internal_similar": [], "discard": [], "note": ""}

#         try:
#             items_txt = compact_products_for_judge(raw_results, limit=8)
#             judge_out = await (judge_prompt | llm).ainvoke(
#                 {"query": search_query, "items": items_txt}
#             )
#             judge_txt = getattr(judge_out, "content", None) or str(judge_out)
#             judge = json.loads(judge_txt)
#         except Exception as e:
#             logger.warning(
#                 "judge failed (ignored): %s",
#                 e,
#                 extra={"endpoint": "/chat", "error": str(e)},
#             )
#             # fallback: no rompemos el flujo
#             judge = {
#                 "internal_best": [],
#                 "internal_similar": list(range(1, min(len(raw_results), 8) + 1)),
#                 "discard": [],
#                 "note": "Fallback: regulador no disponible.",
#             }

#         raw_best = pick_by_indices(raw_results, judge.get("internal_best", []))
#         raw_similar = pick_by_indices(raw_results, judge.get("internal_similar", []))

#         # si el regulador deja todo vacío y había resultados, al menos usa top3 como similares
#         if not raw_best and not raw_similar and raw_results:
#             raw_similar = raw_results[: min(len(raw_results), 3)]
#         # --- FIN REGULADOR ---

#         # Construimos results (best + similar) para mantener compatibilidad con la UI
#         raw_for_response = raw_best + raw_similar

#         results = [
#             CompleteSearchProduct(
#                 product_name=r.get("product_name", ""),
#                 product_family=r.get("product_family"),
#                 description=r.get("description"),
#                 source=r.get("source"),
#                 url=r.get("url"),
#                 image=r.get("image"),
#                 score=float(r.get("score") or r.get("text_score") or 0.0),
#                 color=r.get("color", None),
#             )
#             for r in raw_for_response
#         ]

#         # Traducción (si aplica)
#         try:
#             results = await translate_results(results=results, llm=llm, target_lang=target_lang)
#             logger.info("translate_results ok", extra={"endpoint": "/chat", "target_lang": target_lang})
#         except Exception as e:
#             logger.warning(
#                 "translate_results failed (ignored): %s",
#                 e,
#                 extra={"endpoint": "/chat", "error": str(e)},
#             )

#         # WEB search
#         web_results: list[WebSearchProduct] = []
#         web_txt = ""
#         try:
#             logger.info("WEB query", extra={"q": search_query})
#             web_items = await web_search_products(search_query, k=3, lang=target_lang)
#             logger.info("web_items received", extra={"count": len(web_items or []), "sample": (web_items[0] if web_items else None),})
#             web_txt = web_results_to_bullets(web_items)
            
#             if (web_items or []) and not (web_txt or "").strip():
#                 blocks = []
#                 for i, it in enumerate(web_items[:3], start=1):
#                     title = (it.get("product_name") or it.get("title") or "No disponible").strip()
#                     source = (it.get("source") or "No disponible").strip()
#                     url = (it.get("url") or "No disponible").strip()
#                     img = (it.get("image") or it.get("thumbnail") or "").strip() or "No disponible"
#                     snippet = (it.get("description") or it.get("snippet") or "").replace("\n", " ").strip() or "No disponible"

#                     blocks.append(
#                         f"[{i}]\n"
#                         f"Nombre: {title}\n"
#                         f"Fuente: {source}\n"
#                         f"Descripción/Motivo: {snippet}\n"
#                         f"Imagen: {img}\n"
#                         f"URL: {url}"
#                     )
#                 web_txt = "\n\n".join(blocks).strip()

#             web_results = [
#                 WebSearchProduct(
#                     title=(it.get("product_name") or it.get("title") or "").strip(),
#                     url=it.get("url"),
#                     snippet=(it.get("description") or it.get("snippet") or None),
#                     source=it.get("source"),
#                     image=it.get("image") or None,
#                 )
#                 for it in (web_items or [])
#                 if it.get("url")
#             ]
            
#             logger.info(
#                 "web_results built",
#                 extra={
#                     "count": len(web_results or []),
#                     "sample": (web_results[0].model_dump() if web_results else None),
#                 },
#             )
#         except Exception as e:
#             logger.warning(
#                 "web_search_products failed (ignored): %s",
#                 e,
#                 extra={"endpoint": "/chat", "error": str(e)},
#             )
#             web_results = []
#             web_txt = ""

#         # Preparamos texto para el prompt final (best / similar separados)
#         hist_txt = history_to_text(body.history)

#         best_count = len(raw_best)
#         results_best = results[:best_count]
#         results_similar = results[best_count:]

#         products_best_txt = (
#             results_to_bullets(results_best, limit=min(len(results_best), 3))
#             if results_best
#             else "(vacío)"
#         )

#         limit = min(max(body.top_k, 6), 12)
#         products_similar_txt = (
#             results_to_bullets(results_similar, limit=limit)
#             if results_similar
#             else "(vacío)"
#         )

#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", FULLY_DETAILED_CHATBOT_SYSTEM_PROMPT),
#                 (
#                     "human",
#                     "Idioma de respuesta: {target_language}\n\n"
#                     "Historial:\n{history}\n\n"
#                     "Query:\n{query}\n\n"
#                     "Catálogo interno (mejor coincidencia):\n{products_best}\n\n"
#                     "Catálogo interno (similares):\n{products_similar}\n\n"
#                     "Resultados encontrados en internet (si están vacíos, ignora esta sección):\n{web_products}\n\n"
#                     "Reglas de salida (importante):\n"
#                     "- Devuelve SOLO la respuesta final para el usuario.\n"
#                     "- NO copies ni pegues literalmente listas, URLs, ni bloques completos de los resultados.\n"
#                     "- Usa esos resultados solo como contexto para escoger 3-5 recomendaciones.\n"
#                     "- Si 'Catálogo interno (mejor coincidencia)' está vacío, dilo explícitamente y apóyate en 'similares' y/o 'web'.\n"
#                     "- Para cada recomendación: nombre del producto + 1 frase corta del porqué encaja.\n\n"
#                     "Genera la respuesta.",
#                 ),
#             ]
#         )

#         chain = prompt | llm
#         llm_out = await chain.ainvoke(
#             {
#                 "target_language": target_lang,
#                 "history": hist_txt or "(vacío)",
#                 "query": body.query,
#                 "search_query": search_query,
#                 "products_best": products_best_txt,
#                 "products_similar": products_similar_txt,
#                 "web_products": web_txt or "(sin resultados web)",
#             }
#         )

#         answer = getattr(llm_out, "content", None) or str(llm_out)

#         logger.info(
#             "Chat response generated",
#             extra={
#                 "endpoint": "/chat",
#                 "used_context": used_ctx,
#                 "results_count": len(results),
#                 "web_results_count": len(web_results),
#             },
#         )

#         return ChatResponse(
#             answer=answer.strip(),
#             predicted_type=predicted_type,
#             results=results,
#             web_results=web_results or None,
#         )

#     except Exception as e:
#         logger.exception("Chat endpoint failed", extra={"endpoint": "/chat", "error": str(e)})
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, summary="Chatbot usando RAG + Ollama")
async def chat_endpoint(body: ChatRequest):
    try:
        logger.info(
            "Chat request received",
            extra={
                "endpoint": "/chat",
                "query": body.query,
                "history_len": len(body.history or []),
                "method": "POST",
            },
        )

        # 1) Contexto base
        search_query, used_ctx = build_search_query(body.query, body.history)
        _append_jsonl(
            "/tmp/chat_requests.jsonl",
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "query": body.query,
                "history_len": len(body.history or []),
                "history_tail": _history_tail_plain(body.history, n=8),
                "search_query": search_query,
                "used_ctx": used_ctx,
            },
        )
        logger.info(
            "CTX_DEBUG user_q=%r | resolved_q=%r | used_ctx=%s | hist_len=%d | hist_tail=%s",
            body.query,
            search_query,
            used_ctx,
            len(body.history or []),
            _history_tail_plain(body.history, n=8),
        )

        # 2) Sanitización por motor
        rag_q = sanitize_rag_query(search_query)
        web_q = sanitize_web_query(search_query)

        logger.info(
            "Search context resolved",
            extra={
                "endpoint": "/chat",
                "original_query": body.query,
                "search_query": search_query,
                "used_ctx": used_ctx,
                "rag_q": rag_q,
                "web_q": web_q,
                "history_len": len(body.history or []),
                "last_user": history_last_user_query(body.history),
            },
        )

        target_lang = (body.target_language or "es").strip().lower()
        llm = get_llm()

        # -------------------------
        # RAG
        # -------------------------
        try:
            rag = complete_search(rag_q, n_results=body.top_k)  # usa rag_q, no search_query
        except Exception as e:
            logger.warning(
                "RAG failed (ignored): %s",
                e,
                extra={"endpoint": "/chat", "error": str(e), "rag_q": rag_q},
            )
            rag = {"predicted_type": None, "results": []}

        predicted_type = rag.get("predicted_type")
        raw_results = rag.get("results", []) or []

        logger.info(
            "RAG retrieved",
            extra={
                "endpoint": "/chat",
                "predicted_type": predicted_type,
                "n_raw_results": len(raw_results),
                "top_scores": [
                    float(r.get("score") or r.get("text_score") or 0.0)
                    for r in (raw_results[:5] or [])
                ],
                "top_names": [r.get("product_name") for r in (raw_results[:5] or [])],
            },
        )

        # -------------------------
        # REGULADOR (LLM)
        # -------------------------
        judge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Eres un validador estricto de relevancia para un buscador de productos.\n"
                    "Evalúas productos del catálogo interno frente a una consulta.\n"
                    "Devuelve SOLO JSON válido. No añadas texto fuera del JSON.",
                ),
                (
                    "human",
                    "Consulta del usuario:\n{query}\n\n"
                    "Productos candidatos (cada línea empieza por su índice i):\n{items}\n\n"
                    "Clasifica cada producto en UNA categoría:\n"
                    "- internal_best: encaja claramente con la consulta\n"
                    "- internal_similar: se parece pero no cumple totalmente\n"
                    "- discard: no es relevante\n\n"
                    "Reglas:\n"
                    "- Usa solo índices que EXISTAN en la lista.\n"
                    "- Sé conservador: si dudas, usa internal_similar.\n\n"
                    "Devuelve JSON EXACTO con este formato:\n"
                    "{{\n"
                    '  "internal_best":[1],\n'
                    '  "internal_similar":[2,3],\n'
                    '  "discard":[4,5],\n'
                    '  "note":"1 frase breve"\n'
                    "}}",
                ),
            ]
        )

        judge = {"internal_best": [], "internal_similar": [], "discard": [], "note": ""}

        try:
            items_txt = compact_products_for_judge(raw_results, limit=8)
            judge_out = await (judge_prompt | llm).ainvoke(
                {"query": search_query, "items": items_txt}
            )
            judge_txt = getattr(judge_out, "content", None) or str(judge_out)
            judge = extract_json_obj(judge_txt)
        except Exception as e:
            logger.warning(
                "judge failed (ignored): %s",
                e,
                extra={"endpoint": "/chat", "error": str(e)},
            )
            judge = {
                "internal_best": [],
                "internal_similar": list(range(1, min(len(raw_results), 8) + 1)),
                "discard": [],
                "note": "Fallback: regulador no disponible.",
            }

        raw_best = pick_by_indices(raw_results, judge.get("internal_best", []))
        raw_similar = pick_by_indices(raw_results, judge.get("internal_similar", []))

        if not raw_best and not raw_similar and raw_results:
            raw_similar = raw_results[: min(len(raw_results), 3)]

        raw_for_response = raw_best + raw_similar

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
            for r in raw_for_response
        ]

        # Traducción (si aplica)
        try:
            results = await translate_results(
                results=results, llm=llm, target_lang=target_lang
            )
            logger.info(
                "translate_results ok",
                extra={"endpoint": "/chat", "target_lang": target_lang},
            )
        except Exception as e:
            logger.warning(
                "translate_results failed (ignored): %s",
                e,
                extra={"endpoint": "/chat", "error": str(e)},
            )

        # -------------------------
        # WEB search
        # -------------------------
        web_results: list[WebSearchProduct] = []
        web_txt = ""
        try:
            logger.info("WEB query", extra={"q_raw": search_query, "q_sanitized": web_q})
            web_items = await web_search_products(web_q, k=3, lang=target_lang)

            logger.info(
                "web_items received",
                extra={
                    "count": len(web_items or []),
                    "sample": (web_items[0] if web_items else None),
                },
            )
            
            logger.info(
                "web_items keys",
                extra={
                    "keys0": list(web_items[0].keys()) if (web_items and isinstance(web_items[0], dict)) else None,
                    "sample0": web_items[0] if web_items else None,
                },
            )

            web_txt = web_results_to_bullets(web_items)
            logger.info("web_txt built", extra={"len": len(web_txt or ""), "preview": (web_txt[:300] if web_txt else "")})

            # Construcción robusta de web_results (no tumbar todo por 1 item)
            web_results = []
            for it in (web_items or []):
                if not (it.get("url") or "").strip():
                    continue
                try:
                    web_results.append(
                        WebSearchProduct(
                            title=(it.get("product_name") or it.get("title") or "").strip(),
                            url=(it.get("url") or "").strip(),
                            snippet=(it.get("description") or it.get("snippet") or None),
                            source=(it.get("source") or None),
                            image=(it.get("image") or it.get("thumbnail") or None),
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "WebSearchProduct validation failed (skipping item)",
                        extra={"error": str(e), "item": it},
                    )

            logger.info(
                "web_results built",
                extra={
                    "count": len(web_results or []),
                    "sample": (web_results[0].model_dump() if web_results else None),
                },
            )

            logger.info(
                "decision_summary",
                extra={
                    "endpoint": "/chat",
                    "used_ctx": used_ctx,
                    "final_search_query": search_query,
                    "rag_raw_results": len(raw_results),
                    "internal_results": len(results),
                    "web_items": len(web_items or []),
                    "web_results": len(web_results or []),
                },
            )

        except Exception as e:
            logger.warning(
                "web_search_products failed (ignored): %s",
                e,
                extra={"endpoint": "/chat", "error": str(e)},
            )
            web_results = []
            web_txt = ""

        # -------------------------
        # Prompt final
        # -------------------------
        hist_txt = history_to_text(body.history)

        best_count = len(raw_best)
        results_best = results[:best_count]
        results_similar = results[best_count:]

        products_best_txt = (
            results_to_bullets(results_best, limit=min(len(results_best), 3))
            if results_best
            else "(vacío)"
        )

        limit = min(max(body.top_k, 6), 12)
        products_similar_txt = (
            results_to_bullets(results_similar, limit=limit)
            if results_similar
            else "(vacío)"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FULLY_DETAILED_CHATBOT_SYSTEM_PROMPT),
                (
                    "human",
                    "Idioma de respuesta: {target_language}\n\n"
                    "Historial:\n{history}\n\n"
                    "Query original del usuario:\n{user_query}\n\n"
                    "Query resuelta (con contexto si aplica):\n{search_query}\n\n"
                    "Catálogo interno (mejor coincidencia):\n{products_best}\n\n"
                    "Catálogo interno (similares):\n{products_similar}\n\n"
                    "Resultados encontrados en internet:\n{web_products}\n\n"
                    "has_web={has_web}\n"
                    "WEB_RESULTS_COUNT={web_results_count}\n\n"
                    "Reglas de salida (muy importante):\n"
                    "- Si has_web=true: debes incluir una sección 'Encontrados en la web' con al menos 1 recomendación basada en web_products.\n"
                    "- Si has_web=true: NO puedes decir que no hay resultados web.\n"
                    "- Si has_web=false: puedes decir que no hay resultados web.\n\n"
                    "Reglas de salida:\n"
                    "- Devuelve SOLO la respuesta final para el usuario.\n"
                    "- NO copies ni pegues literalmente listas, URLs, ni bloques completos de los resultados.\n"
                    "- Usa los resultados (internos y web) solo como contexto para escoger 3-5 recomendaciones.\n"
                    "- Si 'Catálogo interno (mejor coincidencia)' está vacío, dilo explícitamente y apóyate en 'similares' y/o 'web'.\n"
                    "- Para cada recomendación: nombre del producto + 1 frase corta del porqué encaja.\n\n"
                    "Genera la respuesta.",
                ),
            ]
        )


        chain = prompt | llm
        # logger.info(
        #     "final_prompt_inputs",
        #     extra={
        #         "target_language": target_lang,
        #         "query": body.query,
        #         "search_query": search_query,
        #         "internal_best_len": len(products_best_txt or ""),
        #         "internal_similar_len": len(products_similar_txt or ""),
        #         "web_items_count": len(web_items or []),
        #         "web_results_count": len(web_results or []),
        #         "web_txt_len": len(web_txt or ""),
        #         "web_txt_is_empty": not bool((web_txt or "").strip()) or (web_txt or "").startswith("(sin resultados"),
        #         "web_txt_preview": (web_txt or "")[:250],
        #     },
        # )
        logger.info("WEB_TXT_LEN=%s", len(web_txt or ""))
        logger.info("WEB_TXT_PREVIEW=%s", (web_txt or "")[:300])
        llm_out = await chain.ainvoke(
            {
                "target_language": target_lang,
                "history": hist_txt or "(vacío)",
                "user_query": body.query,
                "search_query": search_query,
                "products_best": products_best_txt,
                "products_similar": products_similar_txt,
                "web_products": web_txt or "(sin resultados web)",
                "has_web": bool(web_results),
                "web_results_count": len(web_results or []),
            }
        )

        answer = getattr(llm_out, "content", None) or str(llm_out)

        logger.info(
            "Chat response generated",
            extra={
                "endpoint": "/chat",
                "used_context": used_ctx,
                "results_count": len(results),
                "web_results_count": len(web_results),
                "final_search_query": search_query,
            },
        )

        return ChatResponse(
            answer=answer.strip(),
            predicted_type=predicted_type,
            results=results,
            web_results=web_results or None,
        )

    except Exception as e:
        logger.exception(
            "Chat endpoint failed", extra={"endpoint": "/chat", "error": str(e)}
        )
        raise HTTPException(status_code=500, detail=str(e))


