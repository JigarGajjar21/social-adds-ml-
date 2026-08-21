"""
run.py
------
Application entry point.

Development:
    python run.py

Production (Windows):
    waitress-serve --host=0.0.0.0 --port=8000 run:app

Production (Linux):
    gunicorn run:app --bind 0.0.0.0:8000 --workers 4
"""

import os
from app import create_app
from app.predictor import load_model
from config import get_config

cfg = get_config()
app = create_app(cfg)

# Load ML model at startup
with app.app_context():
    load_model(app.config["MODEL_PATH"])

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = app.config.get("DEBUG", False)
    app.run(host="0.0.0.0", port=port, debug=debug)
