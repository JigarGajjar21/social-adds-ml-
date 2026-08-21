"""Integration tests for Flask routes."""
import pytest
from app import create_app
from app.predictor import load_model
from config import TestingConfig


@pytest.fixture(scope="module")
def client():
    app = create_app(TestingConfig)
    with app.app_context():
        load_model(app.config["MODEL_PATH"])
    with app.test_client() as c:
        yield c


class TestHomePage:

    def test_home_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_home_contains_form(self, client):
        r = client.get("/")
        assert b"predictForm" in r.data


class TestFormPredict:

    def test_valid_form_predict(self, client):
        r = client.post("/predict", data={"gender": "1", "age": "30", "salary": "50000"})
        assert r.status_code == 200
        # Should show confidence text
        assert b"Confidence" in r.data

    def test_invalid_age_shows_error(self, client):
        r = client.post("/predict", data={"gender": "1", "age": "0", "salary": "50000"})
        assert r.status_code == 200
        assert b"Age" in r.data

    def test_negative_salary_shows_error(self, client):
        r = client.post("/predict", data={"gender": "1", "age": "30", "salary": "-100"})
        assert r.status_code == 200
        assert b"Salary" in r.data


class TestAPIPredict:

    def test_api_valid_request(self, client):
        r = client.post(
            "/api/predict",
            json={"gender": 1, "age": 30, "salary": 50000},
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert "purchased" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_api_invalid_age(self, client):
        r = client.post(
            "/api/predict",
            json={"gender": 1, "age": 0, "salary": 50000},
        )
        assert r.status_code == 422
        data = r.get_json()
        assert data["success"] is False
        assert "Age" in data["error"]

    def test_api_missing_field(self, client):
        r = client.post("/api/predict", json={"gender": 1})
        assert r.status_code == 422
        data = r.get_json()
        assert data["success"] is False

    def test_api_wrong_content_type(self, client):
        r = client.post("/api/predict", data={"gender": "1", "age": "30", "salary": "50000"})
        assert r.status_code == 400

    def test_api_prediction_values(self, client):
        r = client.post(
            "/api/predict",
            json={"gender": 1, "age": 30, "salary": 50000},
        )
        data = r.get_json()
        assert data["purchased"] in (0, 1)
        assert 0.0 <= data["confidence"] <= 1.0


class TestBatchPredict:

    def test_batch_valid(self, client):
        payload = [
            {"gender": 1, "age": 30, "salary": 50000},
            {"gender": 0, "age": 45, "salary": 80000},
        ]
        r = client.post("/api/batch-predict", json=payload)
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert data["count"] == 2

    def test_batch_not_array(self, client):
        r = client.post("/api/batch-predict", json={"gender": 1})
        assert r.status_code == 400

    def test_batch_limit(self, client):
        payload = [{"gender": 1, "age": 25, "salary": 40000}] * 101
        r = client.post("/api/batch-predict", json=payload)
        assert r.status_code == 400


class TestHealth:

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_health_returns_model_version(self, client):
        r = client.get("/health")
        data = r.get_json()
        assert data["model"]["version"] is not None
