# Carpeta `src/`

La raíz del repositorio sigue la convención de Cookiecutter Data Science (`data/`, `notebooks/`, `models/`, `reports/`, etc.).  
Esta carpeta se enfoca únicamente en el código Python del proyecto.

## Estructura General

- `handlers/`  
  Módulos reutilizables para carga de datos, preprocesamiento, entrenamiento y evaluación.

- `energy_efficiency/`  
  Contiene el runner principal (`runner.py`) y el submódulo `modeling/` del template de la materia.

- `config.py`  
  Parámetros globales compartidos entre pruebas y pipelines.

- `main.py`  
  Punto de entrada para ejecutar el pipeline orquestado con los handlers.

- `tests/`  
  Se reubicaron en `tests/unit/` dentro de la raíz para alinear el layout con Cookiecutter; se conservan aquí únicamente las dependencias de código.
## Relación con el pipeline principal (`src/main.py`)

El pipeline principal del proyecto vive en:

`src/main.py`

Este script ejecuta:

1. **Carga del dataset final** desde:  
   `data/energy_efficiency_final.csv`

2. **Preprocesamiento automático** con Pipelines:
   - `SimpleImputer`
   - `StandardScaler`
   - ColumnTransformer (solo numéricas)

3. **Entrenamiento de tres modelos**:
   - Linear Regression  
   - Random Forest  
   - Gradient Boosting  

4. **Evaluación del modelo** con:
   - Holdout (20% de test)
   - Validación cruzada (K-Fold = 5)

5. **Registro en MLflow**:
   - métricas,
   - parámetros,
   - artefactos,
   - modelo serializado.

6. **Exportación de resultados** a: `notebooks/results_metrics.csv`
`notebooks/results_metrics.html`

## Propósito de la carpeta `src/`

Esta carpeta actúa como puente entre:
- el código de modelado (`src/main.py`, `handlers/`)
- y los resultados/evidencias requeridas por la Fase 2 del proyecto.

Su organización permite:
- reproducibilidad,
- trazabilidad,
- y soporte claro para la futura Fase 3. 

## Contribución – Modeling & Engineering Lead (Sebastián)

- Implementación del pipeline de entrenamiento en `handlers/model_trainer.py`, incluyendo:
  - Preprocesamiento automático (conversión a numérico, imputación por mediana y estandarización).
  - Entrenamiento de tres modelos: LinearRegression, RandomForest y GradientBoosting como problema de regresión multi-salida (heating_load y cooling_load).
  - Validación cruzada (5-fold CV) con métricas R², RMSE y MAE.

- Integración del pipeline completo en `src/main.py`, conectando:
  - Carga de datos originales.
  - Limpieza y tratamiento de outliers.
  - Versión de dataset limpio con DVC en `data/interim/cleansed/energy_efficiency_modified.csv`.
  - Entrenamiento y evaluación de modelos.

- Pruebas automatizadas con `pytest` para asegurar la estabilidad del sistema:
  - Tests unitarios de `DataLoader`, `DataPreprocessor` y `ModelTrainer`.
  - Test de integración del pipeline end-to-end (`test_pipeline_integration.py`).
  - Tests de reproducibilidad del modelo (`test_reproducibility.py`) controlando semillas aleatorias.

- Versionado:
  - Datos versionados con DVC y remoto configurado en S3 (`s3://itesm-mna/202502-equipo8/dvc`).
  - Evidencia de versionado de modelos con MLflow en `notebooks/mlflow_evidence`.
