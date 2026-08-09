"""
Trains a lightweight logistic regression classifier that predicts
whether an (actual_output, expected_output) pair is a "good" match,
using engineered features plus a semantic similarity signal from the
existing sentence-transformers model.

Run manually: python -m app.ml.train_classifier
Produces: app/ml/classifier.joblib
"""
import csv

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def build_features(rows: list[dict], model: SentenceTransformer) -> tuple[np.ndarray, np.ndarray]:
    actual_texts = [r["actual_output"] for r in rows]
    expected_texts = [r["expected_output"] for r in rows]

    actual_emb = model.encode(actual_texts, convert_to_tensor=True)
    expected_emb = model.encode(expected_texts, convert_to_tensor=True)

    features = []
    for i, row in enumerate(rows):
        semantic_sim = util.cos_sim(actual_emb[i], expected_emb[i]).item()
        word_overlap = float(row["word_overlap"])
        length_ratio = float(row["length_ratio"])
        features.append([semantic_sim, word_overlap, length_ratio])

    X = np.array(features)
    y = np.array([int(row["label"]) for row in rows])
    return X, y


def train(data_path: str = "training_data.csv", output_path: str = "app/ml/classifier.joblib"):
    with open(data_path) as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows")
    print("Encoding with sentence-transformers (this takes a moment)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X, y = build_features(rows, model)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["bad", "good"]))
    print("Feature coefficients [semantic_sim, word_overlap, length_ratio]:", clf.coef_[0])

    joblib.dump(clf, output_path)
    print(f"\nSaved trained model -> {output_path}")


if __name__ == "__main__":
    train()