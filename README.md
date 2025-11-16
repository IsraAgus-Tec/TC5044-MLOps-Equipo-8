# TC5044 – Energy Efficiency (Cookiecutter Layout)

Este repositorio ahora sigue la convención de [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/):

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

Consulta `src/README.md` y `data/README.md` para más detalles de cada módulo.
