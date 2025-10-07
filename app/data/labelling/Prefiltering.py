import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

# --- 1. Cargar dataset ---
csv_path = "/Users/paulacamprecios/.cache/kagglehub/datasets/bhavikjikadara/e-commerce-products-images/versions/2/styles.csv"
df = pd.read_csv(csv_path, usecols=["productDisplayName", "articleType"]).dropna()

X = df["productDisplayName"].astype(str)
y = df["articleType"].astype(str)

# --- 2. Filtrar clases poco frecuentes ---
min_count = 5
counts = y.value_counts()
y = y.apply(lambda x: x if counts[x] >= min_count else "Other")

# --- 3. Dividir en train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Generar embeddings ---
sbert = SentenceTransformer("all-MiniLM-L6-v2")
X_train_emb = sbert.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = sbert.encode(X_test.tolist(), show_progress_bar=True)

# --- 5. Entrenar clasificador ---
clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
clf.fit(X_train_emb, y_train)

# --- 6. Evaluar rendimiento ---
y_pred = clf.predict(X_test_emb)
print(classification_report(y_test, y_pred))

# --- 7. Guardar modelo y encoder ---
with open("classifier_model.pkl", "wb") as f:
    pickle.dump(clf, f)

sbert.save("sbert_model")

print("✅ Modelo entrenado y guardado como 'classifier_model.pkl' y 'sbert_model/'")
