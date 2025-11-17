# Reproducibilidad del Modelo

Este documento resume cómo validar que el modelo exportado produce los mismos resultados en un entorno distinto al de desarrollo.

## 1. Dependencias y semillas

- Todas las dependencias están fijadas en `requirements.txt` (FastAPI, scikit-learn, pandas, etc.).
- Las semillas (`RANDOM_STATE = 42`) están definidas y reutilizadas en `src/main.py`, `handlers/model_trainer.py` y las pruebas para garantizar splits consistentes.

## 2. Artefactos versionados

- Dataset base (`data/interim/energy_efficiency_modified.csv`) y su versión cleansed (`data/interim/cleansed/energy_efficiency_modified.csv`) están trackeados con DVC (`*.dvc` en los mismos directorios).
- El modelo serializado y metadatos se ubican en `models/energy_efficiency_rf.joblib` y `models/energy_efficiency_rf.json`. Ese artefacto se referencia como `models:/energy_efficiency_rf/latest`.
- Las métricas de referencia viven en `notebooks/results_metrics.csv` y `.html`.

## 3. Pasos para reproducir en un entorno limpio

1. Clona el repositorio y posiciona la rama correspondiente.
2. Crea un entorno virtual y instala las dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Recupera los datasets versionados:
   ```bash
   dvc pull data/interim/energy_efficiency_modified.csv.dvc
   dvc pull data/interim/cleansed/energy_efficiency_modified.csv.dvc
   ```
4. Ejecuta el pipeline:
   ```bash
   python -m src.main --dataset data/energy_efficiency_final.csv \
                      --output notebooks \
                      --figures-dir reports/figures
   ```
5. Compara `notebooks/results_metrics.csv` con la tabla de referencia (valores esperados):

| Model            | Target        | R2      | RMSE    | MAE    |
|------------------|---------------|---------|---------|--------|
| LinearRegression | heating_load  | 0.90827 | 0.02901 | 0.02070 |
| LinearRegression | cooling_load  | 0.88808 | 0.03119 | 0.02316 |
| RandomForest     | heating_load  | 0.99625 | 0.00587 | 0.00378 |
| RandomForest     | cooling_load  | 0.96009 | 0.01862 | 0.01169 |
| GradientBoosting | heating_load  | 0.99869 | 0.00347 | 0.00255 |
| GradientBoosting | cooling_load  | 0.98804 | 0.01020 | 0.00657 |

Los números deben coincidir (± tolerancia flotante) si las semillas y datos son los mismos.

## 4. Servir y validar inferencias

1. Tras correr el pipeline, arranca el servicio:
   ```bash
   uvicorn src.api.app:app --reload
   ```
2. Envía una petición POST a `/predict` en el nuevo entorno con los mismos samples usados de referencia. Ejemplo:
   ```json
   {
     "samples": [
       {
         "relative_compactness": 0.82,
         "surface_area": 650,
         "wall_area": 270,
         "roof_area": 220,
         "overall_height": 3.5,
         "orientation": 4,
         "glazing_area": 0.25,
         "glazing_area_distribution": 2
       }
     ]
   }
   ```
3. La respuesta debe concordar con la obtenida en el entorno original (mismas predicciones para heating/cooling load).

## 5. Evidencia

- Conserva los archivos generados (`notebooks/results_metrics.*`, `reports/figures/**`, `models/*.joblib/json`) y súbelos (o su hash) como evidencia.
- Los experimentos previos están registrados en `mlruns/`; puedes reabrirlos con `mlflow ui --backend-store-uri file:mlruns`.

Con este flujo se demuestra que el modelo entrenado es portátil y reproducible entre entornos.
