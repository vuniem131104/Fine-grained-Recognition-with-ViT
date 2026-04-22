import io
import base64
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import httpx
from api import app, postprocess_predictions, classes, insert_prediction


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_env(monkeypatch):
    """Set fake environment variables so the app doesn't crash on startup."""
    monkeypatch.setenv("KSERVE_MODEL_URL", "http://fake-kserve")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("POSTGRES_USER", "user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pass")
    monkeypatch.setenv("POSTGRES_PORT", "5432")

@pytest.fixture
def client(mock_env):
    """
    Creates a TestClient for FastAPI.
    We MUST patch psycopg2 and httpx BEFORE entering the TestClient context
    because they are initialized during the app's `lifespan` startup.
    """
    with patch("api.psycopg2.connect") as mock_pg_connect, \
         patch("api.httpx.AsyncClient") as mock_httpx_client:
        
        mock_client_instance = AsyncMock()
        mock_httpx_client.return_value = mock_client_instance
        
        mock_conn = MagicMock()
        mock_pg_connect.return_value = mock_conn

        with TestClient(app) as test_client:
            test_client.mock_http = mock_client_instance
            test_client.mock_db = mock_conn
            yield test_client

@pytest.fixture
def dummy_image_bytes():
    """Generates a valid 10x10 JPEG image in memory."""
    file_obj = io.BytesIO()
    image = Image.new("RGB", (10, 10), color="red")
    image.save(file_obj, format="JPEG")
    file_obj.seek(0)
    return file_obj.read()

def test_postprocess_predictions_success():
    idx_2_class = {idx: cls_name for idx, cls_name in enumerate(classes)}
    
    logits = np.zeros(len(classes))
    logits[0] = 10.0  
    logits[1] = 5.0  
    logits[2] = 2.0
    logits[3] = 1.0

    result = postprocess_predictions(logits, idx_2_class)
    
    assert result["predicted_class"] == classes[0]
    assert result["probability"] > 0.9
    assert len(result["top_3_alternatives"]) == 3
    assert result["top_3_alternatives"][0]["class"] == classes[1]

def test_postprocess_predictions_error():
    """Test that bad input raises an exception (helps cover the except block)."""
    with pytest.raises(Exception):
        postprocess_predictions(None, {})

def test_insert_prediction():
    mock_conn = MagicMock()
    insert_prediction("base64string", 0.99, "Bird", [{"class": "Other", "probability": 0.01}], mock_conn)
    
    mock_conn.cursor().execute.assert_called_once()
    mock_conn.commit.assert_called_once()


def test_health_check_success(client):
    """Test /health when KServe is ready."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    client.mock_http.get.return_value = mock_response

    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_health_check_kserve_down(client):
    """Test /health when KServe returns an error."""
    client.mock_http.get.side_effect = httpx.HTTPError("KServe unreachable")

    response = client.get("/health")
    
    assert response.status_code == 503
    assert response.json() == {"status": "error"}

def test_predict_success(client, dummy_image_bytes):
    """Test a successful full flow of the /predict endpoint."""
    
    dummy_logits = np.zeros(len(classes)).tolist()
    dummy_logits[0] = 5.0
    
    mock_response = MagicMock()
    mock_response.json.return_value = {"outputs": [{"data": dummy_logits}]}
    client.mock_http.post.return_value = mock_response

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == classes[0]
    
    client.mock_http.post.assert_called_once()
    
    client.mock_db.cursor().execute.assert_called_once()

def test_predict_invalid_image(client):
    """Test uploading a text file instead of an image."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"this is not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]

def test_predict_kserve_timeout(client, dummy_image_bytes):
    """Test behavior when KServe takes too long to respond."""
    client.mock_http.post.side_effect = httpx.TimeoutException("Timeout")

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 504
    assert "timeout" in response.json()["detail"].lower()

def test_predict_kserve_http_error(client, dummy_image_bytes):
    """Test behavior when KServe returns a 500/400 error."""
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Model not found"
    
    client.mock_http.post.side_effect = httpx.HTTPStatusError(
        "Error", request=mock_request, response=mock_response
    )

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 502
    assert "Model not found" in response.json()["detail"]