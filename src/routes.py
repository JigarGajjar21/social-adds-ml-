import logging
from flask import Blueprint, render_template, request, jsonify
from src import limiter
from src.predictor import predict, get_model_info
from src.validators import validate_prediction_input

logger    = logging.getLogger(__name__)
main_bp   = Blueprint("main", __name__)


# ─────────────────────────────────────────────
# HTML Routes
# ─────────────────────────────────────────────

@main_bp.route("/")
def home():
    return render_template("index.html")


@main_bp.route("/predict", methods=["POST"])
@limiter.limit("30 per minute")
def predict_form():
    """Handle HTML form submission."""
    v = validate_prediction_input(
        request.form.get("gender"),
        request.form.get("age"),
        request.form.get("salary"),
    )

    if not v.valid:
        logger.warning("Form validation failed: %s", v.error)
        return render_template("index.html", error_text=v.error)

    try:
        result = predict(v.gender, v.age, v.salary)
        return render_template("index.html", result=result)
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        return render_template("index.html", error_text="Prediction failed. Please try again.")


# ─────────────────────────────────────────────
# JSON REST API Routes
# ─────────────────────────────────────────────

@main_bp.route("/api/predict", methods=["POST"])
@limiter.limit("60 per minute")
def api_predict():
    """
    JSON prediction endpoint.

    Request body:
        {"gender": 1, "age": 30, "salary": 50000}

    Response:
        {
            "success": true,
            "purchased": 1,
            "label": "Will Purchase",
            "confidence": 0.87,
            "confidence_pct": "87.0%",
            "probabilities": {
                "will_purchase": 0.87,
                "will_not_purchase": 0.13
            }
        }
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"success": False, "error": "Invalid or empty JSON body"}), 400

    v = validate_prediction_input(
        body.get("gender"),
        body.get("age"),
        body.get("salary"),
    )

    if not v.valid:
        logger.warning("API validation failed: %s", v.error)
        return jsonify({"success": False, "error": v.error}), 422

    try:
        result = predict(v.gender, v.age, v.salary)
        return jsonify({"success": True, **result}), 200
    except Exception as e:
        logger.error("API prediction error: %s", str(e))
        return jsonify({"success": False, "error": "Prediction failed. Please try again."}), 500


@main_bp.route("/api/batch-predict", methods=["POST"])
@limiter.limit("10 per minute")
def api_batch_predict():
    """
    Batch JSON prediction endpoint.

    Request body:
        [
            {"gender": 1, "age": 30, "salary": 50000},
            {"gender": 0, "age": 45, "salary": 80000}
        ]

    Response:
        {
            "success": true,
            "count": 2,
            "results": [ {...}, {...} ]
        }
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    body = request.get_json(silent=True)
    if not isinstance(body, list):
        return jsonify({"success": False, "error": "Request body must be a JSON array"}), 400

    if len(body) > 100:
        return jsonify({"success": False, "error": "Batch size limit is 100 records"}), 400

    results = []
    for i, item in enumerate(body):
        v = validate_prediction_input(
            item.get("gender"),
            item.get("age"),
            item.get("salary"),
        )
        if not v.valid:
            results.append({"index": i, "success": False, "error": v.error})
            continue
        try:
            result = predict(v.gender, v.age, v.salary)
            results.append({"index": i, "success": True, **result})
        except Exception as e:
            logger.error("Batch prediction error at index %d: %s", i, str(e))
            results.append({"index": i, "success": False, "error": "Prediction failed"})

    return jsonify({"success": True, "count": len(results), "results": results}), 200


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@main_bp.route("/health")
def health():
    """
    Health check endpoint for load balancers and monitoring tools.
    Returns 200 if the app and model are ready.
    """
    info = get_model_info()
    status = "ok" if info.get("status") == "loaded" else "degraded"
    http_code = 200 if status == "ok" else 503

    return jsonify({
        "status":      status,
        "model":       info,
    }), http_code
