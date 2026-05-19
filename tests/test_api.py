"""
Tests de l'API FastAPI — Home Credit Default Risk
Lancer avec : pytest tests/test_api.py -v --tb=short
"""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# ── Tous les patchs AVANT l'import de api.py ──────────────────────────────────
# patch("api.log_request", new_callable=AsyncMock), empêche une vraie insertion PostgreSQL avec l'engine factice
# log_request est défini dans api.py et appelé depuis api.py
with patch.dict(os.environ, {
    "DATABASE_URL": "postgresql+asyncpg://fake:fake@localhost:5432/fakedb",
    "API_KEY":      "test-api-key",
    "HF_TOKEN":     "fake-hf-token",
}), \
patch("api.create_async_engine")  as mock_engine_factory, \
patch("api.init_db",              new_callable=AsyncMock), \
patch("api.log_request",          new_callable=AsyncMock), \
patch("api.hf_hub_download")      as mock_hf_download, \
patch("api.cloudpickle.load")     as mock_pickle_load:

    # ── Engine PostgreSQL factice ──────────────────────────────────────────
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    mock_engine_factory.return_value = mock_engine

    # ── Modèle factice ────────────────────────────────────────────────────
    # predict_proba est appelé en priorité dans /predict
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.7, 0.3]]  # proba défaut = 0.3
    mock_pickle_load.return_value = mock_model

    # hf_hub_download appelé 2 fois : model.pkl puis threshold.txt
    mock_hf_download.side_effect = ["/tmp/fake_model.pkl", "/tmp/fake_threshold.txt"]

    from api import app, app_state, API_KEY_VALUE

# ── Injecter l'état global (lifespan désactivé) ───────────────────────────────
app_state["model"]     = mock_model
app_state["threshold"] = 0.5
app_state["engine"]    = mock_engine


# ── Constantes ────────────────────────────────────────────────────────────────
HEADERS = {"X-API-Key": API_KEY_VALUE}

VALID_PAYLOAD = {
    "AMT_CREDIT":                           500000.0,
    "AMT_INCOME_TOTAL":                     150000.0,
    "DAYS_BIRTH":                           -12000,
    "DAYS_EMPLOYED":                        -3000,
    "EXT_SOURCE_1":                         0.52,
    "EXT_SOURCE_2":                         0.73,
    "EXT_SOURCE_3":                         0.61,
    "AMT_ANNUITY":                          25000.0,
    "AMT_GOODS_PRICE":                      450000.0,
    "DAYS_ID_PUBLISH":                      -2000,
    "DAYS_LAST_PHONE_CHANGE":               -1000,
    "CODE_GENDER_M":                        1,
    "NAME_EDUCATION_TYPE_Higher_education":  0,
}


# ── Fixture ───────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    # Compatible toutes versions de Starlette :
    # on remplace le lifespan de l'app par un no-op
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app.router.lifespan_context = noop_lifespan # générateur async vide
    # FastAPI n'exécute ni la connexion DB ni le téléchargement HF au démarrage du TestClient 
    # app_state reste celui qu'on a injecté manuellement au niveau module
    return TestClient(app)


# ── Tests ──────────────────────────────────────────────────────────────────────
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_missing_api_key(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 403


def test_predict_wrong_api_key(client):
    r = client.post("/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "mauvaise-cle"})
    assert r.status_code == 403


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "default_probability" in body
    assert "risk_level"          in body
    assert "threshold_used"      in body          # champ présent dans PredictionResponse
    assert body["risk_level"] in ("HIGH", "LOW")
    assert 0.0 <= body["default_probability"] <= 1.0


def test_predict_invalid_days_birth(client):
    bad = {**VALID_PAYLOAD, "DAYS_BIRTH": 100}
    r = client.post("/predict", json=bad, headers=HEADERS)
    assert r.status_code == 422


def test_predict_negative_credit(client):
    bad = {**VALID_PAYLOAD, "AMT_CREDIT": -1000}
    r = client.post("/predict", json=bad, headers=HEADERS)
    assert r.status_code == 422


def test_model_info(client):
    r = client.get("/model-info", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "features"   in body
    assert "threshold"  in body
    assert "model_repo" in body