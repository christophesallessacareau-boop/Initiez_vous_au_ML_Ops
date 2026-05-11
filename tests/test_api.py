
"""
Tests de l'API FastAPI — Home Credit Default Risk
Lancer avec : pytest tests/test_api.py -v --tb=short
"""
 
import pytest
from unittest.mock import MagicMock, patch
import httpx
from fastapi.testclient import TestClient
from api import app, app_state, API_KEY_VALUE
 
# Patch du chargement MLflow AVANT l'import de api.py pour éviter de faire du vrai MLflow pendant les tests
with patch("mlflow.set_tracking_uri"), \
     patch("mlflow.pyfunc.load_model") as mock_load, \
     patch("mlflow.models.get_model_info") as mock_info:
 
    # Simuler un modèle qui retourne proba 0.3
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.3]
    mock_load.return_value = mock_model
 
    # Simuler la signature
    mock_sig = MagicMock()
    mock_sig.inputs.to_dict.return_value = [
        {"name": "AMT_CREDIT", "type": "double"},
        {"name": "AMT_INCOME_TOTAL", "type": "double"},
        {"name": "DAYS_BIRTH", "type": "long"},
        {"name": "DAYS_EMPLOYED", "type": "long"},
    ]
    mock_info_obj = MagicMock()
    mock_info_obj.signature = mock_sig
    mock_info.return_value = mock_info_obj 
     
    # Injecter le mock dans l'état global
    app_state["model"] = mock_model
    app_state["feature_names"] = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED"]
    app_state["feature_schema"] = mock_sig.inputs.to_dict()
 
 
HEADERS = {"X-API-Key": API_KEY_VALUE}
 
VALID_PAYLOAD = {
    "AMT_CREDIT": 500000.0,
            "AMT_INCOME_TOTAL": 150000.0,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -3000,
            "EXT_SOURCE_1": 0.52,
            "EXT_SOURCE_2": 0.73,
            "EXT_SOURCE_3": 0.61,
            "AMT_ANNUITY": 25000.0,
            "AMT_GOODS_PRICE": 450000.0,
            "DAYS_ID_PUBLISH": -2000,
            "DAYS_LAST_PHONE_CHANGE": -1000,
            "CODE_GENDER_M": 1,
            "NAME_EDUCATION_TYPE_Higher_education": 0,
}
 
 
@pytest.fixture
def client():
    return TestClient(app)
 
 
# --- /health ---
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
 
 
# --- /predict sans clé API ---
def test_predict_missing_api_key(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 403
 
 
# --- /predict clé invalide ---
def test_predict_wrong_api_key(client):
    r = client.post("/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "mauvaise-cle"})
    assert r.status_code == 403
 
 
# --- /predict données valides ---
def test_predict_valid(client):
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "default_probability" in body
    assert "risk_level" in body
    assert body["risk_level"] in ("HIGH", "LOW")
    assert 0.0 <= body["default_probability"] <= 1.0
 
 
# --- /predict DAYS_BIRTH positif (validation Pydantic) ---
def test_predict_invalid_days_birth(client):
    bad = VALID_PAYLOAD.copy()
    bad["DAYS_BIRTH"] = 100  # doit être négatif
    r = client.post("/predict", json=bad, headers=HEADERS)
    assert r.status_code == 422
 
 
# --- /predict AMT_CREDIT négatif ---
def test_predict_negative_credit(client):
    bad = VALID_PAYLOAD.copy()
    bad["AMT_CREDIT"] = -1000
    r = client.post("/predict", json=bad, headers=HEADERS)
    assert r.status_code == 422
 
 
# --- /model-info ---
def test_model_info(client):
    r = client.get("/model-info", headers=HEADERS)
    assert r.status_code == 200
    assert "features" in r.json()