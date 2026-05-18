---
title: ML Ops Credit Scoring
emoji: 🐠
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
short_description: prédire la solvabilité d'un client
---

![CI](https://github.com/christophesallessacareau-boop/Initiez_vous_au_ML_Ops/actions/workflows/ci_cd.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Docker](https://img.shields.io/badge/docker-ready-blue)  
  
  
Home Credit default risk:
Prédire la probabilité de faillite d'un client.

6 notebooks sont réalisés correspondant chacun à l'analyse exploratoire d'un fichier data :
1_bureau_bureau_balance.ipynb
2_credit_card_balance.ipynb
3_installments_payments.ipynb
4_previous_application.ipynb
5_pos_cash_balance.ipynb
6_application_train.ipynb

Déroulement:
1_Bureau_bureau_balance.ipynb est crée à partir de 2 sources (Bureau et bureau_balance, fusionnés via la clé SK_ID_BUREAU-Identifiant d’un crédit externe).
1_Bureau_bureau_balance.ipynb sera fusionné avec 6_application_train via la clé principale SD_ID_CURR (identifiant unique de chaque demande de prêt)

2_credit_card_balance.ipynb sera fusionné avec 6_application_train.ipynb via la clé SD_ID_CURR

3_installments_payment.ipynb a été fusionné avec 6_application_train.ipynb suite à la clé commune SK_ID_PREV (Identifiant d’un ancien crédit) avec 4_previous_application.ipynb

4_previous_application.ipynb sera fusionné avec 6_application_train.ipynb via la clé SD_ID_CURR

5_pos_cash_balance.ipynb sera fusionné avec 6_application_train.ipynb via la clé SD_ID_CURR

6_application_train.ipynb est le principal fichier:
-détient la clé SD_ID_CURR permettant de fusionner les autres fichiers avec ce fichier
-comprend la variable cible que l'on étudie (client solvable / client en faillite)

Une analyse exploratoire et pré-processing sont réalisés sur chacun de ces 6 fichiers.
Leur fusion est effective dans le notebook main.ipynb

main.ipynb:
-fusions des 6 fichiers entre eux
-entrainement de différents modèles
-recherche du meilleur score (le meilleur AUC et le moindre coût) pour l'estimation de la variable cible.
-option: données SHAP (visualisation des critères expliquant la variable cible: globalement ou par individu)

# Partie 2 du projet (API + Docker + Data Drift):Confirmez vos compétences en ML  
  
Utilisation d'une API (FastAPI) pour prédire la solvabilité d'un client au remboursement d'un crédit;  
main_light.upynb reprend le modele choisi (CatBoost) pour ce projet et sauvegardé dans MLflow UI;  
API intégrée dans un conteneur Docker pour la reproductibilité du projet  
  
## Prérequis  
Python 3.12.8  
Docker et Docker Compose v2  
Le répertoire `mlruns/` local avec le modèle entraîné  
  
Cloner le Repo:  
git clone https://github.com/christophesallessacareau-boop/Initiez_vous_au_ML_Ops  
cd Initiez_vous_au_ML_Ops  

Remarque: pour simuler un environnement propre:  
  
Désactiver et supprimer le venv  
deactivate  
Remove-Item -Recurse -Force .venv  

Recréer un venv vierge  
py -3.12 -m venv .venv  

activer l'environnement (sous SE Windows)  
.venv\Scripts\Activate.ps1  

Réinstaller depuis le requirements.txt  
pip install -r requirements.txt  
  
Vider le cache pip  
pip cache purge  
  
## API FastAPI:  
  
### créer une clé sécurisée pour l'API FastAPI:  
Clé API FastAPI (aléatoire de 256 bits) obtenue dans Power Shell par la commande:  
python -c "import secrets; print(secrets.token_hex(32))"  

### utiliser une clé avec API FastAPI:  
pip install python-dotenv  
dans le fichier .env (.env A RENSEIGNER dans .gitignore):  
API_KEY=votre_clé_admin  
  
Remarque: Adapter le chemin du modèle dans .env et dans le code de l'API :  
MLFLOW_TRACKING_URI: "file:///models/mlruns"  
MODEL_PATH="/models/mlruns/1/models/<votre_model_id>/artifacts"  
  
### lancer API:  
uvicorn api:app --reload --port 8000  

### docs Swagger (auto-générés):  
http://127.0.0.1:8000/docs  
  

## Déploiement local (développement)  
### Build avec docker compose:  
docker compose up --build -d  
  
### ou buid avec docker seul:  
docker build -t scoring-model .  
  
Lancer le conteneur en injectant les clés depuis .env:  
docker run -p 8000:8000 --env-file .env scoring-model  
  
## Lancer les tests unitaires:  
pystest tests/   