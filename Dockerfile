# Image de base 
# Python 3.12 sur une base légère (slim = sans outils inutiles)
FROM python:3.12-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Créer le groupe et l'utilisateur pour exécuter l'application de manière sécurisée
# créer le dossier et changer le propriétaire pour éviter les problèmes de permissions lors du montage du volume pour les modèles MLflow
# Création d'un utilisateur non-root pour des raisons de sécurité
RUN groupadd -r appgroup \
    && useradd -r -g appgroup appuser \
    && mkdir -p /models \
    && chown appuser:appgroup /models

# Installation des dépendances Python 
# On copie requirements.txt EN PREMIER (optimisation cache Docker)
# Si le code change mais pas requirements.txt → cette étape n'est pas rejouée
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Switcher vers l'utilisateur non-root
# Important pour la sécurité : éviter d'exécuter l'application en tant que root
USER appuser

# Port exposé par l'API FastAPI (8000 pour l'API, 7860 pour l'interface HF Spaces)
EXPOSE 7860

# Commande de démarrage
# --host 0.0.0.0 : accessible depuis l'extérieur du conteneur
# --port 7860    : port exposé, N.B: 8000 pour l'API, 7860 pour l'interface HF Spaces
# --reload       : à retirer pour plus de rapidité
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]