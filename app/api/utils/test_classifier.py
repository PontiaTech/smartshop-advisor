from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from app.api.utils.families_translator import add_canonical_families
from app.api.utils.product_family_classifier import build_text, EMB_MODEL, pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


MANGO_CSV = "./data/Mango Products Cleaned.csv"
ZARA_CSV = "./data/Zara - Products Cleaned.csv"

def main():
    df_mango = pd.read_csv(MANGO_CSV, encoding="utf-8").fillna("")
    df_zara = pd.read_csv(ZARA_CSV, encoding="utf-8").fillna("")

    df = pd.concat([df_mango, df_zara], ignore_index=True)
    df = add_canonical_families(df)

    # Solo filas con canonical_family no vacío
    df = df[df["canonical_family"].astype(str).str.strip() != ""]

    texts = [build_text(row) for _, row in df.iterrows()]
    labels = df["canonical_family"].astype(str).tolist()

    print(f"Total ejemplos: {len(texts)}; clases: {len(set(labels))}")

    # --- train / test split ---
    X_train_texts, X_test_texts, y_train_labels, y_test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,   # mantiene proporción de clases
        random_state=42,
    )

    encoder = SentenceTransformer(EMB_MODEL)
    X_train = encoder.encode(X_train_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    X_test = encoder.encode(X_test_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_labels)
    y_test = le.transform(y_test_labels)

    # clf = LogisticRegression(max_iter=200, n_jobs=-1)
    # clf.fit(X_train, y_train)
    
    # clf = LinearSVC(C=1.0, class_weight="balanced")
    # clf.fit(X_train, y_train)
    
    param_grid = {
    "C": [0.1, 0.3, 1.0, 3.0, 10.0]
    }

    base_clf = LinearSVC(class_weight="balanced")

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print("Buscando mejor valor de C para LinearSVC...")
    grid.fit(X_train, y_train)

    print(f"Mejor C encontrado: {grid.best_params_}")

    clf = grid.best_estimator_
    
    # num_classes = len(le.classes_)
    # clf = LGBMClassifier(
    #     objective="multiclass",
    #     num_class=num_classes,
    #     class_weight="balanced",
    #     n_estimators=500,s
    #     learning_rate=0.05,
    #     max_depth=-1,
    #     n_jobs=-1,
    # )

    # clf.fit(X_train, y_train)
    
    # --- evaluación ---
    y_pred = clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)

    print("\n=== Classification report (por clase canónica) ===")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    # Si quieres ver la matriz de confusión:
    # import numpy as np
    # import itertools
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
    # print(cm)

    # --- guardado del modelo entrenado completo ---
    with open("./data/product_family_clf.pkl", "wb") as f:
        pickle.dump({"clf": clf, "label_encoder": le}, f)

    print("Modelo de product_family guardado en ./data/product_family_clf.pkl")

if __name__ == "__main__":
    main()