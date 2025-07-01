from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(title="SmartShop Advisor")

# Definir estructura de datos

class SearchTextInput(BaseModel):
    query: str #El input del usuario eg."botas de piel blancas"

class SearchMultimodalInput(BaseModel):
    query: str

# Endpoints para la entrada de información
## Text - Image - Multimodal

@app.post("/search/text")
def search_by_text(input: SearchTextInput):
    return {
        "message": f"Buscando productos relacionados con tu búsqueda: '{input.query}'",
        "results": [] #tendremos que pasar la lista de productos resultado
    }

@app.post("/search/image")
def search_by_image(image: UploadFile = File(...)):
    return {
        "message": "Buscando productos relacionados con la imagen:",
        "filename": image.filename,
        "results": [] #tendremos que pasar la lista de productos resultado
    }

@app.post("/search/multimodal")
def search_multimodal(input: SearchMultimodalInput):
    return {
        "message": f"Buscando productos relacionados con tu búsqueda:: '{input.query}'",
        "results": [] #tendremos que pasar la lista de productos resultado
    }

