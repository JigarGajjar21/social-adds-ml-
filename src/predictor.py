import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_model  = None
_scaler = None
_meta   = None


def load_model(model_path: str):
    """
    Load model artifacts from disk.
    Called once at app startup.
    """
    global _model, _scaler, _meta

    try:
        artifact = joblib.load(model_path)
        _model   = artifact["model"]
        _scaler  = artifact["scaler"]
        _meta    = {
            "version":    artifact.get("version", "unknown"),
            "trained_at": artifact.get("trained_at", "unknown"),
            "accuracy":   artifact.get("accuracy", "unknown"),
            "features":   artifact.get("features", []),
        }
        logger.info(
            "Model loaded | version=%s | accuracy=%s | trained_at=%s",
            _meta["version"], _meta["accuracy"], _meta["trained_at"]
        )
    except FileNotFoundError:
        logger.error("model.pkl not found at path: %s", model_path)
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Run ml/train.py first to train and save the model."
        )
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise


def predict(gender: int, age: int, salary: int) -> dict:
    """
    Run a single prediction.

    Returns:
        {
            "purchased": 1 or 0,
            "label": "Will Purchase" or "Will Not Purchase",
            "confidence": 0.87,
            "probabilities": {"will_purchase": 0.87, "will_not_purchase": 0.13}
        }
    """
    if _model is None or _scaler is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    features = pd.DataFrame(
        [[gender, age, salary]],
        columns=["Gender", "Age", "EstimatedSalary"]
    )
    scaled   = _scaler.transform(features)

    prediction   = int(_model.predict(scaled)[0])
    probabilities = _model.predict_proba(scaled)[0]  # [prob_class_0, prob_class_1]

    confidence = float(round(probabilities[prediction], 4))

    logger.info(
        "Prediction | gender=%d age=%d salary=%d | result=%d | confidence=%.2f",
        gender, age, salary, prediction, confidence
    )

    return {
        "purchased": prediction,
        "label":     "Will Purchase" if prediction == 1 else "Will Not Purchase",
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%",
        "probabilities": {
            "will_purchase":     float(round(probabilities[1], 4)),
            "will_not_purchase": float(round(probabilities[0], 4)),
        },
    }


def get_model_info() -> dict:
    """Return metadata about the currently loaded model."""
    if _meta is None:
        return {"status": "not loaded"}
    return {"status": "loaded", **_meta}
