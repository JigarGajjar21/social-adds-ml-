# ── Build Stage ───────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Train model if artifact is missing
RUN python ml/train.py

# ── Runtime ───────────────────────────────────────────────────
ENV FLASK_ENV=production
EXPOSE 8000

# Use waitress (production WSGI server)
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=8000", "run:app"]
