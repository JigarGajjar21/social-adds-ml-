"""
ml/train.py
-----------
Training pipeline for the Social Network Ads classifier.
Saves a versioned model artifact to ml/artifacts/model.pkl
"""

import os
import sys
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "social_ads.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "model.pkl")

MODEL_VERSION = "2.0.0"


def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Null values:\n%s", df.isnull().sum())
    return df


def preprocess(df: pd.DataFrame):
    """Encode and split data. Returns X_train, X_test, y_train, y_test, scaler."""
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

    logger.info("Gender encoding: %s",
                dict(zip(le.classes_, le.transform(le.classes_))))

    X = df[["Gender", "Age", "EstimatedSalary"]]
    y = df["Purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train(X_train, y_train) -> RandomForestClassifier:
    logger.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    """Evaluate model and return test accuracy."""
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)

    logger.info("Test Accuracy : %.4f", acc)
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    # 5-fold cross-validation on full dataset
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    cv = cross_val_score(model, X_full, y_full, cv=5, scoring="accuracy")
    logger.info("CV Accuracy  : %.4f (+/- %.4f)", cv.mean(), cv.std())

    return round(float(acc), 4)


def save_artifact(model, scaler, accuracy: float, output_path: str):
    """Save model + scaler + metadata as a versioned artifact."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    artifact = {
        "model":      model,
        "scaler":     scaler,
        "version":    MODEL_VERSION,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy":   accuracy,
        "features":   ["Gender", "Age", "EstimatedSalary"],
    }

    joblib.dump(artifact, output_path)
    logger.info("Artifact saved to %s (version %s)", output_path, MODEL_VERSION)


def main():
    df                                    = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, sc = preprocess(df)
    model                                 = train(X_train, y_train)
    accuracy                              = evaluate(model, X_train, X_test, y_train, y_test)
    save_artifact(model, sc, accuracy, OUTPUT_PATH)
    logger.info("Done. Model version %s | Accuracy: %.4f", MODEL_VERSION, accuracy)


if __name__ == "__main__":
    main()
