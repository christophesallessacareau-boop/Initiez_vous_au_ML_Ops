"""
Home Credit Default Risk - API de scoring:

Chargement du modèle MLflow une seule fois au démarrage (lifespan).
Sécurisation par API Key, validation Pydantic, gestion d'erreurs.
Lancement : uvicorn api:app --reload
Documentation Swagger automatique : http://127.0.0.1:8000/docs
"""

from dotenv import load_dotenv

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc
import mlflow.models
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

# CHARGEMENT DU .env
load_dotenv()


# LOGGING: INFO pour les messages Info, error, critical
# logger__name__ pour identifier la source des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CONFIGURATION  (valeurs par défaut et variables d'environnement)
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    r"file:///C:/Users/chris/Initiez_vous_au_ML_Ops/mlruns",
)
MODEL_PATH: str = os.getenv(
    "MODEL_PATH",
    r"C:/Users/chris/Initiez_vous_au_ML_Ops/mlruns/1/models/m-75f49439c2764e6e91a4371402c42cdc/artifacts",
)


# Clé API pour sécuriser les endpoints (à stocker en secret / variable d'env)
API_KEY_VALUE: str = os.getenv("API_KEY")
API_KEY_NAME: str = "X-API-Key"
if not API_KEY_VALUE:
    raise RuntimeError(
        "Variable d'environnement API_KEY manquante. "
        "Définissez-la dans votre .env ou vos secrets CI/CD."
    )


# modèle + métadonnées chargés une seule fois au démarrage, stockés en RAM
# recommandée par FastAPI: toutes les requêtes l’utilisent sans recharger
# le modèle
app_state: dict = {}


# LIFESPAN : chargement du modèle au démarrage,
# modèle stocké dans app_state. Toutes les requêtes le réutilisent sans
# rechargement

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage de l'API — chargement du modèle...")
    logger.info(f"MODEL_PATH : {MODEL_PATH}")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Chargement direct par chemin fichier (sans run_id)
        app_state["model"] = mlflow.pyfunc.load_model(MODEL_PATH)
        logger.info(" Modèle chargé avec succès.")
        # Récupération du seuil optimal depuis le wrapper sklearn
        inner = app_state["model"]._model_impl
        app_state["threshold"] = float(getattr(inner, "best_threshold_", 0.5))
        logger.info(f"Seuil de décision : {app_state['threshold']:.4f}")
    except Exception as exc:
        logger.error(f" Impossible de charger le modèle : {exc}")
        raise RuntimeError(f"Échec du chargement du modèle : {exc}") from exc
    yield

    app_state.clear()  # nettoyage à l'arrêt
    logger.info("API arrêtée — ressources libérées.")


# APPLICATION API

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


# SÉCURITÉ  — API Key
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Vérifie la clé API fournie dans l'en-tête X-API-Key."""
    if api_key != API_KEY_VALUE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé API invalide ou manquante.",
        )
    return api_key


# ---------------------------------------------------------------------------
# SCHÉMA PYDANTIC  — validation des entrées
# ---------------------------------------------------------------------------
class CreditFeatures(BaseModel):
    """
    Caractéristiques financières du client.
    Toutes les valeurs numériques sont validées avant la prédiction.
    Les erreurs 422 sont automatiquement documentées dans Swagger
    """

    # --- Features principales ---
    AMT_CREDIT: float = Field(...,
                              description="Montant du crédit demandé",
                              gt=0)
    AMT_INCOME_TOTAL: float = Field(...,
                                    description="Revenu annuel total",
                                    gt=0)
    DAYS_BIRTH: int = Field(..., description="Âge en jours (négatif)", lt=0)
    DAYS_EMPLOYED: int = Field(..., description="Ancienneté emploi en jours")
    EXT_SOURCE_1: Optional[float] = Field(
        None, description="Score externe 1", ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(
        None, description="Score externe 2", ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(
        None, description="Score externe 3", ge=0, le=1)
    AMT_ANNUITY: Optional[float] = Field(
        None, description="Annuité du crédit", gt=0)
    AMT_GOODS_PRICE: Optional[float] = Field(
        None, description="Prix du bien financé", gt=0)
    DAYS_ID_PUBLISH: Optional[int] = Field(
        None, description="Jours depuis renouvellement pièce identité")
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(
        None, description="Jours depuis dernier changement de téléphone")
    CODE_GENDER_M: Optional[int] = Field(
        None, description="Genre masculin (1=M)", ge=0, le=1)
    NAME_EDUCATION_TYPE_Higher_education: Optional[int] = Field(
        None, ge=0, le=1)

    @field_validator("DAYS_BIRTH")
    @classmethod
    def birth_must_be_negative(cls, v: int) -> int:
        if v >= 0:
            raise ValueError(
                "DAYS_BIRTH doit être négatif (exprimé en jours depuis la naissance).")
        return v

    model_config = {"json_schema_extra": {
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
            "NAME_EDUCATION_TYPE_Higher_education": 0
        }
    }}


class PredictionResponse(BaseModel):
    default_probability: float = Field(...,
                                       description="Probabilité de défaut [0-1]")
    risk_level: str = Field(...,
                            description="HIGH si probabilité > seuil, LOW sinon")
    threshold_used: float = Field(...,
                                  description="Seuil de décision appliqué")
    run_id: str = Field(...,
                        description="Run MLflow utilisé pour la prédiction")


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

# Redirection racine vers /docs
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
        "model_id": "m-736d091d85ae4f39b54fedb7a32230d7",
        "threshold": app_state.get("threshold", None),
    }


# endpoint protégé par X-API-Key dans l'en-tête HTTP
@app.get("/model-info", tags=["Monitoring"],
         summary="Métadonnées du modèle chargé")
def model_info(api_key: str = Security(verify_api_key)):
    """Retourne les informations du modèle actif (requiert X-API-Key)."""
    features = list(CreditFeatures.model_config["json_schema_extra"]["example"].keys())
    return {
        "model_id": "m-75f49439c2764e6e91a4371402c42cdc",
        "run_id": "47910a657cb04891bd1411f4c486d4e5",
        "model_path": MODEL_PATH,
        "threshold": app_state.get("threshold"),
        "nb_features": len(features),
        "features": features,
    }


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
# endpoint protégé par X-API-Key dans l'en-tête HTTP
def predict(
    features: CreditFeatures,
    api_key: str = Security(verify_api_key),
):
    """
    Reçoit les caractéristiques financières d'un client et retourne :
    - **default_probability** : probabilité de défaut entre 0 et 1
    - **risk_level** : HIGH si probabilité > seuil optimal, LOW sinon
    - **threshold_used** : seuil de décision (issu de TunedThresholdClassifierCV)
    - **run_id** : identifiant du run MLflow utilisé
    """
    if "model" not in app_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible. Vérifiez les logs de démarrage.",
        )
    try:
        df = pd.DataFrame([features.model_dump()])
        logger.info(f"Prédiction pour : {df.to_dict(orient='records')[0]}")

        # predict() sur un pyfunc retourne les probabilités si c'est un
        # classifier
        raw = app_state["model"].predict(df)

        # Selon la forme du retour (array 1D ou 2D)
        if hasattr(raw, "ndim") and raw.ndim == 2:
            proba = float(raw[0][1])   # colonne classe positive
        else:
            proba = float(raw[0])

        # Seuil : récupéré depuis le modèle si disponible, sinon 0.5
        inner = app_state["model"]._model_impl
        threshold = float(getattr(inner, "best_threshold_", 0.5))

        return PredictionResponse(
            default_probability=round(proba, 4),
            risk_level="HIGH" if proba > threshold else "LOW",
            threshold_used=round(threshold, 4),
            run_id="47910a657cb04891bd1411f4c486d4e5",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Erreur lors de la prédiction : {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne lors de la prédiction : {str(exc)}",
        )
