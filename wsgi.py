"""
wsgi.py
-------
WSGI entry point for gunicorn.
Gunicorn command: gunicorn wsgi:app
"""
import os
from app import create_app
from app.predictor import load_model
from config import get_config

cfg = get_config()
app = create_app(cfg)

with app.app_context():
    load_model(app.config["MODEL_PATH"])
