# Plant Disease Detection

## Structure du repo
- `data/` : données (raw / processed)
- `src/` : code source (preprocessing, training, utils, evaluation)
- `models/` : modèles sauvegardés
- `api/` : FastAPI pour inférence
- `docker/`, `Dockerfile` : conteneurs
- `mlruns/` : sorties MLflow

## Mise en place (Windows)
Créer venv & installation:
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt