import logging
import sys
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import get_config

# Global limiter instance (attached to app in create_app)
limiter = Limiter(key_func=get_remote_address)


def create_app(config_class=None):
    """
    Flask application factory.
    Creates and configures the app instance.
    """
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # Load config
    cfg = config_class or get_config()
    app.config.from_object(cfg)

    # Setup logging
    _configure_logging(app)

    # Init extensions
    limiter.init_app(app)

    # Register blueprints
    from src.routes import main_bp
    app.register_blueprint(main_bp)

    app.logger.info("App started in %s mode", cfg.__name__)
    return app


def _configure_logging(app):
    """Set up structured logging to stdout."""
    log_level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Remove default Flask handlers, add ours
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(log_level)
