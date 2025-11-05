from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from sentence_transformers import SentenceTransformer
from PIL import Image
import pickle, io, chromadb, numpy as np, json
import traceback

app = FastAPI(title="SmartShop Search API", version="2.0")

encoder = SentenceTransformer("clip-ViT-B-32")

print("🚀 Booting API...")

try:
    encoder_dim = encoder.get_sentence_embedding_dimension()
    print("✅ Encoder loaded — dim:", encoder_dim)
except Exception as e:
    print("❌ Encoder error at startup:", e)

with open("classifier_model.pkl", "rb") as f:
    clf = pickle.load(f)

client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("products")


def predict_article_type(query):
    return clf.predict(encoder.encode([query]))[0]

def embed_text(q):
    emb = encoder.encode([q], convert_to_numpy=True)[0]
    print(f"📝 Text embedding shape: {emb.shape}") 
    return emb.tolist()

async def embed_image(file: UploadFile):
    try:
        # Leer bytes y convertir a imagen PIL
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        emb = encoder.encode(img, convert_to_numpy=True)
        print(f"🖼️ Image embedding shape: {emb.shape}")
        return emb.tolist()
    except Exception:
        raise HTTPException(400, "Imagen inválida")
 
    

# --- Endpoint solo texto ---
@app.post("/search")
async def search(
    query: str = Form(...),  # obligatorio ahora
    history: str = Form("[]"),
    body: dict = Body(None)
):
    # ✅ Fallback para JSON si llega body
    if body:
        query = body.get("query", query)
        history = body.get("history", history)

    print("\n===========================")
    print("📩 Incoming request:")
    print("query:", query)
    print("history:", history)

    # ✅ Parse history
    try:
        conv = json.loads(history) if history else []
    except:
        print("⚠️ Error parseando historial. Lo reseteo.")
        conv = []

    if not query:
        raise HTTPException(status_code=400, detail="Envía texto")

    try:
        conv.append({"role": "user", "content": query})

        # ✅ Embeddings solo texto
        text_emb = embed_text(query)

        print(f"📦 Text embedding length before query: {len(text_emb)}")

        # --- Predecir artículo ---
        article = predict_article_type(query)
        print(f"🧠 Predicted label: {article}")

        # --- Query en Chroma ---
        r = collection.query(
            query_embeddings=[text_emb],
            n_results=3,
            where={"articleType": article}
        )

        results = []
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            results.append({
                "name": doc,
                "metadata": meta,
                "similarity": float(1 - dist)
            })

        print("✅ Response READY")
        print("article:", article)
        print("1st result:", results[0] if results else "No results")

        return {
            "predicted_article_type": article,
            "top_results": results
        }

    except Exception as e:
        print("❌ Exception in /search:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))