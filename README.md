# TC5044 – Energy Efficiency (Cookiecutter Layout)

<p align="left">
  <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" alt="Cookiecutter Data Science">
  </a>
  <a target="_blank" href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python">
  </a>
  <a target="_blank" href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-async-green?logo=fastapi" alt="FastAPI">
  </a>
  <a target="_blank" href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white" alt="Docker">
  </a>
</p>

Este repositorio sigue la convención de [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/):

```
├── data
│   ├── raw/          # Datos originales
│   ├── external/     # Fuentes externas (placeholder)
│   ├── interim/      # Datos intermedios + resultados DVC
│   └── processed/    # Datasets listos para modelar
├── docs/             # Documentación funcional/técnica
├── models/           # Artefactos serializados (placeholder)
├── notebooks/        # Resultados y evidencia de MLflow
├── references/       # Material de apoyo (placeholder)
├── reports/figures   # Gráficos y reportes (placeholder)
├── src/              # Código fuente (handlers, runner, main, etc.)
└── tests/            # `unit/` y pruebas de integración
```

## Cómo ejecutar el pipeline

```bash
python -m src.main --dataset data/interim/energy_efficiency_modified.csv
python -m src.main --dataset data/energy_efficiency_final.csv --output notebooks --figures-dir reports/figures
```

## Pruebas

```bash
pytest tests/unit                 # Pruebas unitarias
pytest tests/test_pipeline_integration.py
```

## Datos versionados

- Los datasets limpios viven en `data/interim/cleansed/` y se siguen versionando con DVC.
- El dataset final para experimentos está en `data/energy_efficiency_final.csv`.

## Visualizaciones EDA

El pipeline exporta imágenes antes y después de la limpieza a `reports/figures/{before,after}_cleansing/`:

- `histograms.png`
- `boxplots.png`
- `correlation_heatmap.png`

Usa `--figures-dir` o `--skip-figures` en `python -m src.main` para personalizar esta salida.

Consulta `docs/REPRODUCIBILITY.md` para el checklist completo de ejecución en un entorno limpio, comparación de métricas y verificación del servicio FastAPI.

## Modelo serializado y servicio FastAPI

- Artefacto por defecto: `models/energy_efficiency_rf.joblib`
- Metadatos: `models/energy_efficiency_rf.json`
- Referencia (MLflow-like): `models:/energy_efficiency_rf/latest`

### Servir el modelo

1. Exporta/actualiza el modelo:
   ```bash
   python -m src.main --dataset data/energy_efficiency_final.csv
   ```
2. Inicia el servicio:
   ```bash
   uvicorn src.api.app:app --reload
   ```
3. Documentación automática (OpenAPI/Swagger): `http://localhost:8000/docs`

### Endpoint principal

- **POST `/predict`**
  - Entrada: `{"samples": [{relative_compactness, surface_area, ..., glazing_area_distribution}, ...]}`
  - Respuesta: `{"predictions": [{"heating_load": float, "cooling_load": float}, ...]}`

Revisa `src/api/app.py` para más detalles del esquema y validaciones.

### Contenedor Docker

Creamos una imagen ligera que empaqueta FastAPI, dependencias y el modelo exportado.

1. Construcción:
   ```bash
   docker build -t ml-service:latest .
   ```
2. Ejecución local:
   ```bash
   docker run -p 8000:8000 ml-service:latest
   ```
3. Publicación en Docker Hub (ejemplo con cuenta `team8`):
   ```bash
   docker tag ml-service:latest team8/ml-service:1.0.0
   docker push team8/ml-service:1.0.0
   docker tag ml-service:latest team8/ml-service:latest
   docker push team8/ml-service:latest
   ```
   Versiones sugeridas:
   - `team8/ml-service:1.0.1` → imagen asociada al experimento actual (modelo `models:/energy_efficiency_rf/latest`).
   - `team8/ml-service:latest` → alias apuntando a la versión estable más reciente.

Una vez corriendo, los endpoints (`/docs`, `/predict`, `/healthz`) funcionan igual que en local.

### Monitoreo y Data Drift

- Ejecuta `python -m src.monitoring.drift_detection` para crear un dataset de monitoreo alterado, evaluar el modelo exportado y comparar contra la línea base.
- Artefactos:
  - Reporte JSON con métricas, deltas y umbrales: `reports/drift/drift_report.json`.
  - Muestra de datos con drift para análisis manual: `reports/drift/drift_sample.csv`.
- Último resultado:
  - `baseline_r2 = 0.8220`, `drift_r2 = 0.7939` (Δ = -0.0281) → por debajo del umbral (0.05), **sin alerta**.
  - `baseline_rmse = 0.0498`, `drift_rmse = 0.0531` (Δ = +0.0033) → incremento controlado (< 0.01 requerido).
  - Acción recomendada por el script: “Continuar monitoreando”. Si se superan los umbrales establecidos (`ΔR² >= 0.05` o `ΔRMSE >= 0.01`), el reporte devolverá `alert: true` y sugerirá revisar el feature pipeline / reentrenar.

Consulta `src/README.md` y `data/README.md` para más detalles de cada módulo.
