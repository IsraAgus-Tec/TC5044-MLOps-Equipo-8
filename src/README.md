# Carpeta `src/`

La carpeta `src/` contiene el código fuente y los resultados estáticos generados por el pipeline del proyecto de eficiencia energética.

## Estructura General

- `src/data/`  
  Reservado para scripts de manejo de datos en Fase 3 (validaciones, utilidades y extracción automatizada).

- `src/handlers/`  
  Espacio para módulos auxiliares como manejo de rutas, configuración o utilidades.  
  Se puede aprovechar en Fase 3 según necesidades de orquestación.

- `src/notebooks/`  
  Carpeta principal de resultados:
  - `results_metrics.csv`
  - `results_metrics.html`
  - `mlflow_evidence/` → Evidencia exportada de MLflow (datasets, artefactos del modelo, estimator, metadata y README específico).
## Relación con el pipeline principal (`runner.py`)

El pipeline principal del proyecto vive en:

energy_efficiency/runner.py

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

6. **Exportación de resultados** a: src/notebooks/results_metrics.csv
src/notebooks/results_metrics.html ## Propósito de la carpeta `src/`

Esta carpeta actúa como puente entre:
- el código de modelado (`energy_efficiency/runner.py` y `modeling/`)
- y los resultados/evidencias requeridas por la Fase 2 del proyecto.

Su organización permite:
- reproducibilidad,
- trazabilidad,
- y soporte claro para la futura Fase 3. 

