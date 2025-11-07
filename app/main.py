from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from sentence_transformers import SentenceTransformer
from PIL import Image
import pickle, io, chromadb, numpy as np, json, traceback
import spacy

app = FastAPI(title="SmartShop Search API", version="2.1")

encoder = SentenceTransformer("clip-ViT-B-32")

with open("classifier_model.pkl", "rb") as f:
    clf = pickle.load(f)

client = chromadb.HttpClient(host="chroma", port=8000)
collection = client.get_collection("products")

# --- Memoria ligera en servidor ---
last_article_type = None
last_embeddings = []  # lista de np.array
last_query= ""


def predict_article_type(query):
    return clf.predict(encoder.encode([query]))[0]


def embed_text(q):
    emb = encoder.encode([q], convert_to_numpy=True)[0]
    return emb.tolist()


# Detectar si la frase tiene sujeto para ver si es follow up o no
nlp = spacy.load("en_core_web_sm")

def has_subject(text: str) -> bool:
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ("nsubj", "nsubj_pass"):
            return True
    return False

@app.post("/search")
async def search(
    query: str = Form(...),
    history: str = Form("[]"),
    body: dict = Body(None)
):
    global last_article_type, last_query, last_embeddings

    if body:
        query = body.get("query", query)

    if not query:
        raise HTTPException(400, "Envía texto")

    try:
        # --- Generar embedding ---
        text_emb = embed_text(query)
        print(f"📦 Text embedding length before query: {len(text_emb)}")
        if last_embeddings and not has_subject(query):
            print("No subject so merging")
            comb_emb = 0.8 * np.array(text_emb) + 0.2 * np.array(last_embeddings)
        else:
            print("We found subject")
            comb_emb = text_emb
            last_embeddings = text_emb
        # --- Clasificar tipo de artículo ---
        if last_query and not has_subject(query):
            full_query = f"{last_query}. {query}"
            article = predict_article_type(full_query)
            print(f"We merged with the previous query  so we have {full_query}")
        else:
            article = predict_article_type(query)
            last_query = query
        print(f"🧠 Predicted label: {article}")


        filter_articles = [article]
        if last_article_type and last_article_type != article:
            filter_articles.append(last_article_type)
            print("Using also last article", last_article_type)

        last_article_type = article

        r = collection.query(
            query_embeddings=[comb_emb],
            n_results=3,
            where={"articleType": {"$in": filter_articles}}
        )

        results = [
            {
                "name": doc,
                "metadata": meta,
                "similarity": float(1 - dist),
                "link": f"https://{doc.replace(' ', '_')}.com"
            }
            for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]

        return {
            "predicted_article_type": article,
            "top_results": results
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
