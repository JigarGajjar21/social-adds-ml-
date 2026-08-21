# Social Network Ads Prediction

A production-ready Machine Learning web application that predicts whether a
social media user will purchase a product — built with Flask, scikit-learn,
and a full REST API.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Tests](https://img.shields.io/badge/Tests-27%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Live Demo

> Clone the repo, run two commands, open your browser.

```bash
pip install -r requirements.txt
python run.py
```
Visit: **http://localhost:5000**

---

## What This Project Does

Given a user's **Gender**, **Age**, and **Estimated Salary**, the model predicts:
- Will they purchase the advertised product?
- How confident is the prediction? (e.g. 87.0%)

---

## Screenshots

### Prediction UI
The web interface takes user inputs and shows a color-coded result with a
confidence score and progress bar.

- Green = Will Purchase
- Red = Will Not Purchase
- Confidence bar shows model certainty

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Random Forest (scikit-learn) |
| Web Framework | Flask 3.1 |
| Model Serialization | joblib |
| Input Validation | Custom validator with dataclasses |
| Rate Limiting | Flask-Limiter |
| Production Server | Waitress (Windows) / Gunicorn (Linux) |
| Testing | pytest + pytest-flask (27 tests) |
| Config Management | python-dotenv, environment-based configs |
| Containerization | Docker |

---

## Project Structure

```
social_network_ads_project/
├── app/
│   ├── __init__.py       # Flask app factory + structured logging
│   ├── routes.py         # HTML routes + REST API endpoints
│   ├── predictor.py      # ML prediction logic (decoupled from web)
│   └── validators.py     # Centralized input validation
├── ml/
│   ├── train.py          # Full training pipeline with cross-validation
│   └── artifacts/
│       └── model.pkl     # Trained model artifact (versioned)
├── tests/
│   ├── test_validators.py   # 12 unit tests
│   └── test_routes.py       # 15 integration tests
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── config.py             # Dev / Testing / Production configs
├── run.py                # Application entry point
├── requirements.txt      # Pinned dependencies
├── Dockerfile
└── .env.example          # Environment variable template
```

---

## API Reference

### Single Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "gender": 1,
  "age": 30,
  "salary": 50000
}
```

Response:
```json
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
```

### Batch Prediction (up to 100 records)
```http
POST /api/batch-predict
Content-Type: application/json

[
  {"gender": 1, "age": 30, "salary": 50000},
  {"gender": 0, "age": 45, "salary": 80000}
]
```

### Health Check
```http
GET /health

{
  "status": "ok",
  "model": {
    "version": "2.0.0",
    "accuracy": 0.9,
    "trained_at": "2026-08-21"
  }
}
```

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 90.0% |
| CV Mean (5-fold) | 89.75% |
| Precision (class 1) | 0.84 |
| Recall (class 1) | 0.90 |
| F1-Score | 0.87 |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 27 tests pass:
- 12 unit tests (input validation edge cases)
- 15 integration tests (all routes, API, health check)

---

## Docker

```bash
docker build -t social-ads-predictor .
docker run -p 8000:8000 social-ads-predictor
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | `development` or `production` |
| `SECRET_KEY` | dev value | Change in production |
| `MODEL_PATH` | `ml/artifacts/model.pkl` | Path to model artifact |
| `PORT` | `5000` | Port to listen on |

---

## Key Engineering Decisions

- **App factory pattern** — makes the app testable and configurable per environment
- **ML decoupled from web** — `predictor.py` has zero Flask imports; can be used standalone
- **Versioned model artifacts** — every saved model includes version, accuracy, and timestamp
- **joblib over pickle** — faster serialization, safer for sklearn objects
- **Structured logging** — timestamped logs with module context, not `print()`
- **Rate limiting** — prevents API abuse (30/min HTML, 60/min API)
- **Confidence scores** — `predict_proba()` exposes model certainty, not just yes/no

---

## Author

Built as a demonstration of production ML engineering practices.

Feel free to open an issue or reach out with questions.
