"""
train_model.py

Trains a classifier on the hand-landmark data collected by
data_collection.py, and saves the trained model + label encoder
to the models/ directory.

USAGE:
    python src/train_model.py

Requires data/landmarks.csv to already contain samples for at least
2 different labels (run data_collection.py first).
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"No training data found at {CSV_PATH}. "
            "Run data_collection.py for each sign first."
        )

    df = pd.read_csv(CSV_PATH)
    if df["label"].nunique() < 2:
        raise ValueError(
            "Need at least 2 different signs collected before training. "
            f"Currently have: {df['label'].unique().tolist()}"
        )

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Classes: {list(encoder.classes_)}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Model saved to {os.path.abspath(MODEL_PATH)}")
    print(f"Label encoder saved to {os.path.abspath(ENCODER_PATH)}")


if __name__ == "__main__":
    main()
