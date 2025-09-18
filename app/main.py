from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

app = FastAPI()

class SearchTextInput(BaseModel):
    query: str

class SearchMultimodalInput(BaseModel):
    query: str
    image_url: str  # O puedes usar UploadFile si quieres subir la imagen


# --- PLACEHOLDERS PARA FUNCIONES ---
def generate_text_embedding(query: str):
    #Llamar a OpenAI o HuggingFace para generar embedding
    return [0.1, 0.2, 0.3]  # Placeholder

def generate_image_embedding(image_bytes: bytes):
    #Usar CLIP o modelo similar para generar embedding
    return [0.5, 0.6, 0.7]  # Placeholder

def query_vector_database(embedding: List[float]):
    #Llamar a Pinecone, Weaviate, FAISS...
    # Devolver lista de productos con score
    return [
        {"product_id": 1, "name": "Producto de ejemplo", "score": 0.92},
        {"product_id": 2, "name": "Producto de ejemplo 2", "score": 0.85}
    ]

def rank_results(results):
    #Ordenar resultados según score u otras reglas de negocio
    return sorted(results, key=lambda x: x["score"], reverse=True)

def refine_query_with_llm(query: str):
    #Llamar a GPT o similar para reformular query si es necesario
    return query  # De momento, devolvemos igual


# --- ENDPOINTS ---
@app.post("/search/text")
def search_by_text(input: SearchTextInput):
    refined_query = refine_query_with_llm(input.query)
    embedding = generate_text_embedding(refined_query)
    raw_results = query_vector_database(embedding)
    ranked_results = rank_results(raw_results)

    return {
        "message": f"Resultados para: '{refined_query}'",
        "results": ranked_results
    }

@app.post("/search/image")
async def search_by_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    embedding = generate_image_embedding(image_bytes)
    raw_results = query_vector_database(embedding)
    ranked_results = rank_results(raw_results)

    return {
        "message": f"Resultados para la imagen '{image.filename}'",
        "results": ranked_results
    }

@app.post("/search/multimodal")
async def search_multimodal(input: SearchMultimodalInput):
    # Paso 1: Embedding de texto
    refined_query = refine_query_with_llm(input.query)
    text_embedding = generate_text_embedding(refined_query)

    # Paso 2: Embedding de imagen (placeholder: suponemos que ya la tienes en bytes)
    image_embedding = generate_image_embedding(b"fake_image")

    # Paso 3: Combinar embeddings (promedio simple)
    multimodal_embedding = [
        (t + i) / 2 for t, i in zip(text_embedding, image_embedding)
    ]

    raw_results = query_vector_database(multimodal_embedding)
    ranked_results = rank_results(raw_results)

    return {
        "message": f"Resultados para búsqueda multimodal: '{refined_query}'",
        "results": ranked_results
    }
