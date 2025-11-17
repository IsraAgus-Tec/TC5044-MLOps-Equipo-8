# Evidencia de MLflow

Esta carpeta contiene evidencia exportada manualmente de los experimentos realizados con MLflow durante el entrenamiento de los modelos.

MLflow se ejecuta localmente vía `runner.py` y registra:
- métricas
- artefactos del modelo
- parámetros del pipeline
- datasets utilizados
- estructura del modelo
- metadata de ejecución

La carpeta completa `mlruns/` **no debe subirse al repositorio**, porque contiene información pesada dependiente del entorno y no es apta para control de versiones.  
Por ello, esta carpeta (`mlflow_evidence/`) contiene únicamente la **evidencia mínima necesaria** para revisión académica.

---

## 📂 Contenido típico

### `datasets/`
Copias de los datasets utilizados en un run.

### `model_artifact/`
Artefactos exportados del modelo entrenado (serializaciones, pipelines, etc.).

### `estimator.html`
Reporte visual del estimador entrenado (útil para modelos tipo árbol).

### `run_meta.yaml`
Metadatos del experimento:
- parámetros
- métricas
- timestamps
- id del run
- tags del experimento

### `results_metrics.csv` y `results_metrics.html`
Métricas globales exportadas después de entrenar todos los modelos.

---

## 📝 Notas finales

- Esta carpeta **sí** debe subirse al repositorio.
- Sirve como evidencia de ejecución de la Fase 2.
- Será utilizada como insumo para la Fase 3 del proyecto.
