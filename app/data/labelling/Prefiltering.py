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

# --- 3. Dividir Train/Test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Cargar modelo multimodal MULTILINGÜE ---
model = SentenceTransformer("clip-ViT-B-32")

# --- 5. Generar embeddings ---
X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True, convert_to_numpy=True)
X_test_emb  = model.encode(X_test.tolist(),  show_progress_bar=True, convert_to_numpy=True)

# --- 6. Entrenar clasificador ---
clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
clf.fit(X_train_emb, y_train)

# --- 7. Evaluar ---
y_pred = clf.predict(X_test_emb)
print(classification_report(y_test, y_pred))

# --- 8. Guardar modelo y encoder ---
with open("classifier_model.pkl", "wb") as f:
    pickle.dump(clf, f)

model.save("multilingual_clip_encoder")

print("✅ Clasificador guardado como 'classifier_model.pkl'")
print("✅ Encoder guardado en 'multilingual_clip_encoder/'")
