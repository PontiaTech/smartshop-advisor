import requests
import json

# --- Configuración ---
FASTAPI_URL = "http://localhost:8000/search"  # Ajusta si tu puerto es diferente

# --- Queries de prueba ---
queries = [
    "red running shoes",
    "blue jeans",
    "silver watch",
    "black tshirt",
]

# --- Función para testear una query ---
def test_query(query_text):
    print(f"\n🚀 Enviando query: '{query_text}'")

    payload = {"query": query_text}

    try:
        response = requests.post(
            FASTAPI_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al conectar con FastAPI: {e}")
        return

    print(f"📦 Estado HTTP: {response.status_code}")

    if response.status_code != 200:
        print("❌ Respuesta inesperada:")
        print(response.text)
        return

    # Intentar decodificar JSON
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print("❌ No se pudo decodificar JSON:", e)
        print("Respuesta cruda:", response.text)
        return

    # Mostrar información paso a paso
    predicted_type = data.get("predicted_article_type")
    top_results = data.get("top_results", [])

    print(f"🧠 Tipo predicho: {predicted_type}")
    print(f"📊 Número de resultados devueltos: {len(top_results)}")

    for i, hit in enumerate(top_results, start=1):
        name = hit.get("name")
        meta = hit.get("metadata")
        sim = hit.get("similarity")
        print(f"{i}. {name} | metadata: {meta} | similarity: {sim:.4f}")

# --- Ejecutar pruebas ---
if __name__ == "__main__":
    for q in queries:
        test_query(q)
