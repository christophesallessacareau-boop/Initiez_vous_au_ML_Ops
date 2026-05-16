"""
Home Credit Default Risk - API de scoring

Chargement du modèle depuis Hugging Face Hub au démarrage (lifespan).
Sécurisation par API Key, validation Pydantic, gestion d'erreurs.
Logging en base SQLite (fichier api_logs.db)
Enregistrement : timestamp, inputs, output, temps d'exécution, statut HTTP
Endpoint /monitoring/stats pour un aperçu rapide
Lancement : uvicorn api:app --reload
Documentation Swagger automatique : http://127.0.0.1:8000/docs
"""

from dotenv import load_dotenv

import os
import time
import logging
import sqlite3
import json
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

import cloudpickle
from huggingface_hub import hf_hub_download

import pandas as pd
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

# ── Chargement du .env ────────────────────────────────────────────────────────
load_dotenv()


# ── Logging ───────────────────────────────────────────────────────────────────
# LOGGING: INFO pour les messages Info, error, critical
# logger__name__ pour identifier la source des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN")
HF_MODEL_REPO: str = "ChristopheSalles31/credit-scoring-model"

# Chemin du fichier SQLite (modifiable via .env)
DB_PATH: str = os.getenv("DB_PATH", "api_logs.db")

# Clé API pour sécuriser les endpoints (à stocker en variable d'env / secret HF)
API_KEY_VALUE: str = os.getenv("API_KEY")
API_KEY_NAME: str = "X-API-Key"
if not API_KEY_VALUE:
    raise RuntimeError("Variable d'environnement API_KEY manquante. ")

# ── modèle chargé une seule fois ───────────────────────────
# et métadonnées chargées une seule fois au démarrage, le tout stocké en RAM
# recommandé par FastAPI: toutes les requêtes l’utilisent sans recharger
# le modèle
app_state: dict = {}


# ── Base de données SQLite ────────────────────────────────────────────────────


def init_db() -> None:
    """
    Crée la table de logs si elle n'existe pas encore.

    Colonnes :
        id               clé primaire auto-incrémentée
        timestamp        date/heure UTC de l'appel (format ISO 8601)
        endpoint         route appelée (ex: /predict)
        http_status      code retourné (200, 422, 500…)
        execution_ms     temps de traitement en millisecondes
        inputs           features du client sérialisées en JSON
        default_proba    probabilité de défaut retournée (NULL si erreur)
        risk_level       HIGH ou LOW (NULL si erreur)
        error_message    message d'erreur si applicable
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                endpoint       TEXT    NOT NULL,
                http_status    INTEGER NOT NULL,
                execution_ms   REAL    NOT NULL,
                inputs         TEXT,
                default_proba  REAL,
                risk_level     TEXT,
                error_message  TEXT
            )
        """)
        conn.commit()
    logger.info(f"Base SQLite initialisée : {DB_PATH}")


def log_request(
    endpoint: str,
    http_status: int,
    execution_ms: float,
    inputs: dict | None = None,
    default_proba: float | None = None,
    risk_level: str | None = None,
    error_message: str | None = None,
) -> None:
    """
    Insère une ligne dans api_logs.
    Les inputs sont sérialisés en JSON.
    Les erreurs d'écriture sont loggées mais n'interrompent pas l'API.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO api_logs
                    (timestamp, endpoint, http_status, execution_ms,
                     inputs, default_proba, risk_level, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    endpoint,
                    http_status,
                    round(execution_ms, 2),
                    json.dumps(inputs) if inputs else None,
                    default_proba,
                    risk_level,
                    error_message,
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.error(f"Impossible d'écrire dans SQLite : {exc}")


# ── Lifespan : chargement du modèle au démarrage ─────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Téléchargement du modèle depuis HF Hub...")

    # Initialisation de la base de données au démarrage
    init_db()

    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="model.pkl",
            repo_type="model",
            token=HF_TOKEN,
        )
        threshold_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="threshold.txt",
            repo_type="model",
            token=HF_TOKEN,
        )
        with open(model_path, "rb") as f:
            app_state["model"] = cloudpickle.load(f)
        with open(threshold_path, "r") as f:
            app_state["threshold"] = float(f.read().strip())

        logger.info(f"Modèle chargé. Seuil : {app_state['threshold']:.4f}")

    except Exception as exc:
        logger.error(f"Impossible de charger le modèle : {exc}")
        raise RuntimeError(f"Échec du chargement : {exc}") from exc

    yield

    app_state.clear()
    logger.info("API arrêtée")


# ── Application FastAPI ───────────────────────────────────────────────────────

app = FastAPI(
    title="Home Credit Default Risk — API de scoring",
    description=(
        "Prédit la probabilité de défaut de paiement d'un client "
        "à partir de ses caractéristiques financières.\n\n"
        "**Authentification** : fournir l'en-tête `X-API-Key`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Sécurité — API Key ────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Vérifie la clé API fournie dans l'en-tête X-API-Key."""
    if api_key != API_KEY_VALUE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé API invalide ou manquante.",
        )
    return api_key


# ── Schémas Pydantic ──────────────────────────────────────────────────────────


class CreditFeatures(BaseModel):
    """
    Caractéristiques financières du client.
    Toutes les valeurs numériques sont validées avant la prédiction.
    Les erreurs 422 sont automatiquement documentées dans Swagger.
    """

    AMT_CREDIT: float = Field(..., description="Montant du crédit demandé", gt=0)
    AMT_INCOME_TOTAL: float = Field(..., description="Revenu annuel total", gt=0)
    DAYS_BIRTH: int = Field(..., description="Âge en jours (négatif)", lt=0)
    DAYS_EMPLOYED: int = Field(..., description="Ancienneté emploi en jours")
    EXT_SOURCE_1: Optional[float] = Field(
        None, description="Score externe 1", ge=0, le=1
    )
    EXT_SOURCE_2: Optional[float] = Field(
        None, description="Score externe 2", ge=0, le=1
    )
    EXT_SOURCE_3: Optional[float] = Field(
        None, description="Score externe 3", ge=0, le=1
    )
    AMT_ANNUITY: Optional[float] = Field(None, description="Annuité du crédit", gt=0)
    AMT_GOODS_PRICE: Optional[float] = Field(
        None, description="Prix du bien financé", gt=0
    )
    DAYS_ID_PUBLISH: Optional[int] = Field(
        None, description="Jours depuis renouvellement pièce identité"
    )
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(
        None, description="Jours depuis dernier changement de téléphone"
    )
    CODE_GENDER_M: Optional[int] = Field(
        None, description="Genre masculin (1=M)", ge=0, le=1
    )
    NAME_EDUCATION_TYPE_Higher_education: Optional[int] = Field(None, ge=0, le=1)

    @field_validator("DAYS_BIRTH")
    @classmethod
    def birth_must_be_negative(cls, v: int) -> int:
        if v >= 0:
            raise ValueError(
                "DAYS_BIRTH doit être négatif (exprimé en jours depuis la naissance)."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Probabilité de défaut [0-1]")
    risk_level: str = Field(..., description="HIGH si probabilité > seuil, LOW sinon")
    threshold_used: float = Field(..., description="Seuil de décision appliqué")


# ── Endpoints ─────────────────────────────────────────────────────────────────


# Redirection racine vers /docs pour plus de rapidité
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Monitoring"])
def health():
    """Vérifie que l'API et le modèle sont opérationnels. Pas d'authentification requise."""
    model_loaded = "model" in app_state and app_state["model"] is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_repo": HF_MODEL_REPO,
        "threshold": app_state.get("threshold"),
    }


# endpoint protégé par X-API-Key dans l'en-tête HTTP
@app.get("/model-info", tags=["Monitoring"], summary="Métadonnées du modèle chargé")
def model_info(api_key: str = Security(verify_api_key)):
    """Retourne les informations du modèle actif (requiert X-API-Key)."""
    features = list(CreditFeatures.model_config["json_schema_extra"]["example"].keys())
    return {
        "model_repo": HF_MODEL_REPO,
        "threshold": app_state.get("threshold"),
        "nb_features": len(features),
        "features": features,
    }


@app.get(
    "/monitoring/stats",
    tags=["Monitoring"],
    summary="Statistiques rapides des logs (sans authentification)",
)
def monitoring_stats():
    """
    Aperçu rapide en production : volume, latence moyenne, taux d'erreur.
    Pas d'authentification requise.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                                         AS total_calls,
                    ROUND(AVG(execution_ms), 1)                                      AS avg_latency_ms,
                    ROUND(MAX(execution_ms), 1)                                      AS max_latency_ms,
                    ROUND(AVG(CASE WHEN http_status != 200 THEN 1.0 ELSE 0.0 END) * 100, 2)
                                                                                     AS error_rate_pct,
                    ROUND(AVG(default_proba), 4)                                     AS avg_default_proba
                FROM api_logs
                WHERE endpoint = '/predict'
            """).fetchone()

        return {
            "total_predict_calls": row[0],
            "avg_latency_ms": row[1],
            "max_latency_ms": row[2],
            "error_rate_pct": row[3],
            "avg_default_proba": row[4],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prédiction"],
    summary="Prédit le risque de défaut de paiement",
    responses={
        200: {"description": "Prédiction réussie"},
        403: {"description": "Clé API invalide"},
        422: {"description": "Données d'entrée invalides (validation Pydantic)"},
        503: {"description": "Modèle non disponible"},
        500: {"description": "Erreur interne lors de la prédiction"},
    },
)
def predict(
    features: CreditFeatures,
    api_key: str = Security(verify_api_key),
):
    """
    Reçoit les caractéristiques financières d'un client et retourne :
    - **default_probability** : probabilité de défaut entre 0 et 1
    - **risk_level** : HIGH si probabilité > seuil optimal, LOW sinon
    - **threshold_used** : seuil de décision (issu de TunedThresholdClassifierCV)

    Chaque appel est enregistré dans SQLite (inputs, output, latence, statut).
    """
    start = time.perf_counter()
    inputs_dict = features.model_dump()

    if "model" not in app_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible. Vérifiez les logs de démarrage.",
        )
    try:
        df = pd.DataFrame([inputs_dict])
        logger.info(f"Prédiction pour : {df.to_dict(orient='records')[0]}")

        model = app_state["model"]
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(df)[0][1])
        else:
            proba = float(model.predict(df)[0])

        threshold = app_state.get("threshold", 0.5)
        risk = "HIGH" if proba > threshold else "LOW"
        execution_ms = (time.perf_counter() - start) * 1000

        log_request(
            endpoint="/predict",
            http_status=200,
            execution_ms=execution_ms,
            inputs=inputs_dict,
            default_proba=round(proba, 4),
            risk_level=risk,
        )

        return PredictionResponse(
            default_probability=round(proba, 4),
            risk_level=risk,
            threshold_used=round(threshold, 4),
        )

    except HTTPException:
        raise
    except Exception as exc:
        execution_ms = (time.perf_counter() - start) * 1000
        log_request(
            endpoint="/predict",
            http_status=500,
            execution_ms=execution_ms,
            inputs=inputs_dict,
            error_message=str(exc),
        )
        logger.exception(f"Erreur lors de la prédiction : {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne lors de la prédiction : {str(exc)}",
        )
