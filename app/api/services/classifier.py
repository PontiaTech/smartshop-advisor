from sentence_transformers import SentenceTransformer
import pickle


#usamos el encoder con porque es con el que se ha entrenado el clasificador y si no me da cosas raras
TEXT_CLASSIFIER_MODEL = "clip-ViT-B-32" 
encoder = SentenceTransformer(TEXT_CLASSIFIER_MODEL)

with open("classifier_model.pkl", "rb") as classifier_file:
    clf = pickle.load(classifier_file)


def predict_article_type(query: str) -> str:
    emb = encoder.encode([query])
    return clf.predict(emb)[0]