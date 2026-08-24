"""
run.py
------
Development entry point.

Development : python run.py
Production  : gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2
"""
import os
from src import create_app
from src.predictor import load_model
from config import get_config

cfg = get_config()
app = create_app(cfg)

with app.app_context():
    load_model(app.config["MODEL_PATH"])

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = app.config.get("DEBUG", False)
    app.run(host="0.0.0.0", port=port, debug=debug)
