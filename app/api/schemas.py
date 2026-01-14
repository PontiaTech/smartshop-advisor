from pydantic import BaseModel, Field, field_validator, model_validator, EmailStr, HttpUrl
from typing import Optional, List 

# class User(BaseModel):
    
#     name: str = Field(..., min_length=3, max_length=20, description="User name")
#     surname: str = Field(..., min_length=10, max_length=40, description="User surname")
#     hashed_password: str = Field(..., min_length=3, max_length=20, description="User password hashed before being saved")
#     contact_mail: EmailStr
#     age: Optional[int] = Field(None, ge=16, description="Age (optional) to register, in the EU the minmum is 16 years old")
    
#     @field_validator("name")
#     def nonspace_name(cls, value):
#         value = value.strip()
#         if " " in value:
#             raise ValueError("Names can't contain spaces!")
#         elif not any(vowel in value.lower() for vowel in ["a", "e", "i", "o", "u"]):
#             raise ValueError("Names must contain vowels!")
#         else:
#             return value
    
    
# class Product(BaseModel):
    
#     name: str = Field()
#     description: str = Field(..., min_length=10, max_length=300, description="Description of the product")
#     price: float = Field(..., ge=0.1, description="Product's price")
#     brand: str = Field(..., min_length=3, description="Product's brand")
#     colour: str = Field(..., min_length=3, description="Product's colour")
#     size: str = Field(..., min_length=1, description="Product's size")

   
# class UserPreferences(BaseModel):
    
#     foot_size: Optional[str] = Field(..., min_length=1, description="User's foot size (optional)")
#     shirt_size: Optional[str] = Field(..., min_length=1, description="User's shirt size (optional)")
#     pants_size: Optional[str] = Field(..., min_length=1, description="User's footsize (optional)")
#     preferred_colours: Optional[List[str]] = None
#     preferred_brands: Optional[List[str]] = None
    

# class UserHistory(BaseModel):
    
#     query: Optional[str] = Field(None, max_length=500)
#     image_url: Optional[str] = None
    

# A paula le funciona
# class ResultadoProducto(BaseModel):
#     nombre: str = Field(..., description="Nombre o identificador del producto.")
#     metadatos: dict = Field(..., description="Metadatos asociados al producto.")
#     similitud: float = Field(..., description="Nivel de similitud entre 0 y 1.")
#     enlace: str = Field(..., description="Enlace al producto o página relacionada.")


# class RespuestaBusqueda(BaseModel):
#     tipo_predicho: str = Field(..., description="Tipo de producto detectado por el modelo.")
#     resultados: List[ResultadoProducto] = Field(..., description="Productos más similares encontrados.")
    


 
# Clases necesarias para la funcionalidad completa del chat, el resto no las usamos mas y podrian eliminarse
class CompleteSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class CompleteSearchProduct(BaseModel):
    product_name: str
    product_family: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    image: Optional[str] = None
    score: float
    color: Optional[str] = None


class CompleteSearchResponse(BaseModel):
    results: List[CompleteSearchProduct]
    predicted_type: Optional[str] = None
    

class ChatMessage(BaseModel):
    
    sender: str = Field(..., description="It indicates who is talking")
    content: Optional[str] = Field(None, description="Text message content")
    image_url: Optional[HttpUrl] = Field(None, description="URL of the image sent")
    

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    history: Optional[List[ChatMessage]] = None  # usa tu ChatMessage existente
    target_language: str = "es"
    
class WebSearchProduct(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    source: Optional[str] = None
    image: Optional[str] = None
    

class ChatResponse(BaseModel):
    answer: str
    predicted_type: Optional[str] = None
    results: List[CompleteSearchProduct]
    web_results: Optional[List[WebSearchProduct]] = None
    
    
