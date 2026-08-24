"""
wsgi.py
-------
WSGI entry point for gunicorn.
Usage: gunicorn wsgi:app
"""
import os
from src import create_app
from src.predictor import load_model
from config import get_config

cfg = get_config()
app = create_app(cfg)

with app.app_context():
    load_model(app.config["MODEL_PATH"])
