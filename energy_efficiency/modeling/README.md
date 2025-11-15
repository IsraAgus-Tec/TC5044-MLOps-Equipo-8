# Módulo `energy_efficiency.modeling`

Esta carpeta contiene el código base del template de modelado provisto para el proyecto de MLOps.

Aunque el pipeline principal del equipo se implementa en `energy_efficiency/runner.py`,  
estos módulos se mantienen porque:

- documentan la estructura original sugerida por el curso,
- permiten ejecutar un flujo alternativo de experimentación,
- pueden reutilizarse en la Fase 3 para modularizar el entrenamiento.

## Archivos principales

- `config.py`  
  Manejo de configuración (rutas, parámetros, etc.).

- `dataset.py`  
  Lógica de carga de datos y dataset wrappers.

- `features.py`  
  Construcción de features y transformaciones.

- `train.py`  
  Flujo de entrenamiento basado en el template.

- `predict.py`  
  Flujo de predicción basado en modelos entrenados con el template.

- `run_experiments.py`  
  Script para orquestar experimentos con los componentes anteriores.

## Relación con `runner.py`

El archivo `energy_efficiency/runner.py` implementa un pipeline más integrado y moderno:

- Pipelines de `scikit-learn` (preprocesamiento + modelo),
- evaluación con holdout + validación cruzada,
- logging automático en MLflow,
- exportación de métricas a `src/notebooks/results_metrics.*`.

Ambos enfoques conviven en el repositorio:

- `modeling/` → referencia al template de la materia.  
- `runner.py` → solución final propuesta por el equipo para el proyecto.
