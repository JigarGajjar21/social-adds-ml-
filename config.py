import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

class Config:
    """Base configuration shared by all environments."""
    SECRET_KEY        = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
    MODEL_PATH        = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "artifacts", "model.pkl"))
    RATELIMIT_DEFAULT = "100 per hour"
    RATELIMIT_STORAGE_URI = "memory://"

class DevelopmentConfig(Config):
    DEBUG   = True
    TESTING = False

class TestingConfig(Config):
    DEBUG   = False
    TESTING = True
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "artifacts", "model.pkl")

class ProductionConfig(Config):
    DEBUG   = False
    TESTING = False

config_map = {
    "development": DevelopmentConfig,
    "testing":     TestingConfig,
    "production":  ProductionConfig,
}

def get_config():
    env = os.environ.get("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)
