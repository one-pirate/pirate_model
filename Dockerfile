# Utiliser une image Python officielle
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de MLflow (8080)
EXPOSE 8080

# Définir la variable d'environnement pour MLflow
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:8080

# Lancer ton script principal
CMD ["python", "pirate_model.py"]
