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
python -m src.main --dataset data/energy_efficiency_final.csv --output notebooks
```

## Pruebas

```bash
pytest tests/unit                 # Pruebas unitarias
pytest tests/test_pipeline_integration.py
```

## Datos versionados

- Los datasets limpios viven en `data/interim/cleansed/` y se siguen versionando con DVC.
- El dataset final para experimentos está en `data/energy_efficiency_final.csv`.

Consulta `src/README.md` y `data/README.md` para más detalles de cada módulo.
