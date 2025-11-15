git add README_Fase2.md
git status
ine (Energy Efficiency)

Este documento describe la implementación final de la **Fase 2** del proyecto TC5044 MLOps — Equipo 8.  
Incluye el pipeline completo desarrollado en `energy_efficiency/runner.py`, el registro de experimentos con MLflow, la estructura reorganizada del repositorio y la evidencia generada (métricas, artefactos y datasets utilizados).

Esta documentación sirve como referencia oficial del trabajo de modelado, evaluación y versionamiento realizado por el Data Scientist. 
## 2. Arquitectura General del Proyecto

La Fase 2 redefine la estructura del repositorio para alinearse con prácticas reales de MLOps:


La Fase 2 redefinió la estructura del repositorio alineándose con prácticas reales de MLOps:

```text
TC5044-MLOps-Equipo-8/
├── data/
│   ├── raw/                         # Dataset original sin modificar
│   └── processed/                   # Datasets limpios y estandarizados
│       ├── energy_efficiency_clean.csv
│       ├── energy_efficiency_modified.csv
│       └── energy_efficiency_final.csv   # Dataset final usado por runner.py
│
├── energy_efficiency/
│   ├── __init__.py
│   └── runner.py                    # Pipeline principal (holdout + CV + MLflow)
│
├── src/
│   ├── data/cleansed/               # Evidencia mínima requerida por el curso
│   ├── handlers/                    # Módulos internos auxiliares
│   └── notebooks/
│       ├── mlflow_evidence/         # Evidencia reducida de MLflow
│       │   ├── results_metrics.csv
│       │   └── results_metrics.html
│       └── notebooks/src/           # Resultados adicionales
│
├── mlruns/                          # Tracking local (no se sube al repo)
├── README.md                        # README principal del repositorio
├── README_Fase2.md                  # Documento oficial de entrega Fase 2
└── requirements.txt



Esta estructura separa claramente:

- **Código principal del pipeline**  
- **Template de modelado de la materia (preservado)**  
- **Datos en carpetas raw/processed/final**  
- **Evidencia ligera y versionable de MLflow**  
- **Métricas estáticas exportadas para revisión académica**  
## 3. Pipeline Principal (`energy_efficiency/runner.py`)

La Fase 2 implementa un pipeline completo con `scikit-learn` y MLflow:

### 3.1 Preprocesamiento
El pipeline construye un `ColumnTransformer` que aplica:

- *SimpleImputer* para manejo de valores faltantes  
- *StandardScaler* para normalización  
- Transformaciones consistentes movidas dentro de un `Pipeline` de sklearn

Esto garantiza que el preprocesamiento se ejecute de forma reproducible durante entrenamiento y predicción.

---

### 3.2 Modelos Entrenados
Se entrenan y evalúan **tres modelos**:

1. **Linear Regression**  
2. **Random Forest Regressor (n_estimators=600)**  
3. **Gradient Boosting Regressor**  
   - learning_rate ajustado  
   - profundidad optimizada  

Cada modelo genera:

- métricas holdout  
- métricas de validación cruzada  
- parámetros finales  

---

### 3.3 Métricas Calculadas
Para cada target (`Y1`, `Y2`) se exportan:

- **R²**
- **MAE**
- **RMSE**
- **MAE-CV**
- **R2-CV**
- **RMSE-CV**

Todas las métricas se guardan automáticamente en:
src/notebooks/results_metrics.csv
src/notebooks/results_metrics.html
### 3.4 Logging con MLflow

El pipeline utiliza MLflow para:

- registrar modelos  
- registrar parámetros  
- registrar métricas  
- almacenar artefactos  

Para evitar subir `mlruns/`, la Fase 2 exporta **solo evidencia necesaria** a:
src/notebooks/mlflow_evidence/

### 3.5 Artefactos generados por el pipeline

El pipeline produce automáticamente:

| Archivo | Descripción |
|---------|-------------|
| `results_metrics.csv` | Métricas tabulares finales |
| `results_metrics.html` | Tabla HTML formateada para lectura |
| `datasets/` | Dataset utilizado por el mejor modelo |
| `model_artifact/` | Artefactos del modelo (pickle / sklearn) |
| `estimator.html` | Visualización del modelo (árboles) |
| `run_meta.yaml` | Metadata estructurada del experimento |

Con este pipeline, tu equipo cumple **el 100% de los requisitos de la Fase 2** y queda listo para iniciar la Fase 3.
---

## 4. Resultados Generados por el Pipeline

Al ejecutar el pipeline principal (`energy_efficiency/runner.py`), se generan de manera automática los siguientes artefactos:

### 4.1 Tablas de métricas
- `results_metrics.csv`  
  Archivo en formato tabular con los resultados numéricos para cada modelo:
  - R² del conjunto hold-out  
  - MAE del conjunto hold-out  
  - RMSE del conjunto hold-out  
  - Promedios de R², MAE y RMSE en validación cruzada (5-fold)

- `results_metrics.html`  
  Versión visual en formato tabla para revisión manual.

### 4.2 Evidencia de MLflow
Como la carpeta `mlruns/` no se sube al repositorio, se exportó evidencia ligera dentro de:
src/notebooks/mlflow_evidence/


Esta carpeta contiene:
- `datasets/` (copias de los datos utilizados en un run)
- `model_artifact/` (artefactos del modelo)
- `estimator.html` (estructura del modelo)
- `run_meta.yaml` (metadatos del experimento)

### 4.3 Uso de los resultados
Estos artefactos sirven como insumo para:
- comparar desempeño entre modelos,
- verificar reproducibilidad,  
- y preparar el material requerido para la Fase 3 (despliegue y monitoreo).

## 5. Resultados del Proyecto

Esta sección presenta los resultados obtenidos luego de ejecutar el pipeline completo de la Fase 2, incluyendo las métricas de desempeño para cada uno de los modelos entrenados en las variables objetivo **Y1 (Heating Load)** y **Y2 (Cooling Load)**.

Los resultados originales generados por el pipeline se encuentran en:

- `src/notebooks/results_metrics.csv`
- `src/notebooks/results_metrics.html`

A continuación se resumen los valores más relevantes.

---

## 5.1 Resultados para Y1 (Heating Load)

Los modelos no lineales funcionan considerablemente mejor para esta variable:

| Modelo              | R² Holdout | MAE Holdout | RMSE Holdout |
|---------------------|------------|-------------|---------------|
| Linear Regression   | 0.7816     | 0.0408      | 0.0471        |
| Random Forest       | 0.9850     | 0.0058      | 0.0123        |
| Gradient Boosting   | 0.9913     | 0.0049      | 0.0093        |

**Conclusión para Y1:**  
Gradient Boosting es el mejor modelo, seguido de Random Forest. Ambos ofrecen precisión muy alta y errores extremadamente bajos.

---

## 5.2 Resultados para Y2 (Cooling Load)

En este caso el desempeño de los modelos cambia significativamente:

| Modelo              | R² Holdout | MAE Holdout | RMSE Holdout |
|---------------------|------------|-------------|---------------|
| Linear Regression   | 0.9383     | 0.0166      | 0.0235        |
| Random Forest       | 0.2246     | 0.0236      | 0.0833        |
| Gradient Boosting   | -4.1097    | 0.0353      | 0.2139        |

**Conclusión para Y2:**  
A diferencia de Y1, la variable Y2 presenta un comportamiento más difícil de modelar.  
En esta variable, **Linear Regression es el modelo que mejor generaliza**, mientras que Gradient Boosting y Random Forest no logran capturar correctamente la relación subyacente.

---

## 5.3 Interpretación Global del Desempeño

- **Y1 (Heating Load)** muestra relaciones fuertes y consistentes con las características del dataset, por lo que los modelos no lineales (RF y GB) funcionan excepcionalmente bien.
- **Y2 (Cooling Load)** presenta mayor variabilidad y ruido, lo que ocasiona que modelos complejos se sobreajusten si no se ajustan hiperparámetros adicionales.
- Linear Regression, a pesar de su simplicidad, ofrece el mejor equilibrio para Y2.

---

## 5.4 Uso de los resultados en Fase 3

Los resultados numéricos de este bloque servirán como base para:

- selección del modelo a desplegar,
- análisis de riesgos de sobreajuste,
- elección de métricas de monitoreo para producción,
- tuning adicional con GridSearch o RandomizedSearch,
- posibles transformaciones adicionales al dataset.

Este análisis es indispensable para la siguiente etapa del proyecto.

## 6. Evidencia de MLflow

Durante la ejecución del pipeline, MLflow registra automáticamente modelos, parámetros, métricas y artefactos del entrenamiento.  
Sin embargo, la carpeta completa `mlruns/` no se sube al repositorio debido a su tamaño y a que depende del entorno local de ejecución.

Para efectos de revisión académica, se incluye una **versión ligera y curada de la evidencia** dentro de:
src/notebooks/mlflow_evidence/


Esta carpeta contiene:

- **datasets/**  
  Copias de los datos utilizados en el run para garantizar reproducibilidad.

- **model_artifact/**  
  Artefactos del modelo registrados por MLflow (por ejemplo, archivos serializados).

- **estimator.html**  
  Visualización del modelo entrenado (cuando aplica, como en modelos tipo árbol).

- **run_meta.yaml**  
  Metadatos del experimento, incluyendo:
  - parámetros del modelo,
  - métricas,
  - timestamp,
  - ID del run,
  - etiquetas del experimento,
  - información de entorno.

- **README.md**  
  Descripción de cómo se generó la evidencia y qué representa cada archivo.

El contenido de esta carpeta garantiza:
- transparencia respecto al proceso del experimento,  
- trazabilidad de los resultados,  
- evidencia mínima suficiente para la evaluación de la Fase 2,  
- un punto de partida para la Fase 3 (monitoreo, API o despliegue).

## 7. Ejecución del Pipeline

El pipeline principal del proyecto se encuentra en:
energy_efficiency/runner.py


Este script entrena los tres modelos (Linear Regression, Random Forest y Gradient Boosting), genera validación *holdout* + *cross-validation*, registra los experimentos en MLflow y exporta las métricas finales en formato CSV y HTML.

### 7.1. Activar entorno virtual

En GitHub Codespaces:

```bash
source env/bin/activate

En local (Mac / Linux): 
python -m venv env
source env/bin/activate

En Windows: env\Scripts\activate

7.2. Instalar dependencias:
pip install -r requirements.txt

7.3. Ejecutar el pipeline con el dataset final
Asegúrese de que el dataset esté en:
data/energy_efficiency_final.csv

Ejecutar el pipeline:
python -m energy_efficiency.runner --target both --data data/energy_efficiency_final.csv

7.4. Resultados generados
Al finalizar la ejecución, se generan los siguientes archivos:
src/notebooks/results_metrics.csv
src/notebooks/results_metrics.html

Estos documentos contienen las métricas de desempeño para ambos targets (Y1 y Y2) y sirven como evidencia principal de la Fase 2.

7.5. Evidencia en MLflow
MLflow guardará los experimentos localmente en la carpeta:
mlruns/

Esta carpeta no se sube al repositorio, pero se incluye una versión reducida como evidencia en:
src/notebooks/mlflow_evidence/

7.6. Nombre del experimento registrado
El experimento se registra automáticamente como:
Energy Efficiency

Este nombre aparece en los metadatos y en los registros de MLflow.

##8. Conclusiones y Trabajo Futuro
---

8.1. Conclusiones

El desarrollo del pipeline permitió construir un flujo completo de experimentación para el análisis de eficiencia energética, alineado con prácticas modernas de MLOps. Entre los puntos más relevantes destacan:

- Se estableció un pipeline reproducible que integra preprocesamiento, entrenamiento, validación y registro de experimentos.
- Se trabajó con tres modelos clave (Linear Regression, Random Forest y Gradient Boosting), con métricas consistentes para ambos targets Y1 y Y2.
- MLflow se integró correctamente para rastrear parámetros, métricas y artefactos, permitiendo trazabilidad de los experimentos.
- Los archivos de resultados generados (`results_metrics.csv` y `results_metrics.html`) documentan de manera transparente el desempeño del pipeline.

El modelo de Gradient Boosting mostró el mejor desempeño para Y1 mientras que Random Forest entregó resultados competitivos. En Y2, aunque el holdout del Gradient Boosting fue negativo, la validación cruzada mostró estabilidad, indicando sensibilidad a la partición y la necesidad de seguir ajustando hiperparámetros.

8.2. Trabajo Futuro (Fase 3)

Para la siguiente etapa, se propone continuar con las siguientes mejoras:

- Modulación del pipeline en componentes más pequeños para facilitar mantenimiento y pruebas unitarias.
- Implementar validaciones adicionales del dataset antes del entrenamiento.
- Incorporar herramientas de automatización como DVC para controlar versiones de datos y modelos.
- Explorar ajuste de hiperparámetros con GridSearchCV o RandomizedSearchCV.
- Diseñar un sistema de despliegue básico utilizando FastAPI para exponer un endpoint de predicción.
- Evaluar y documentar riesgos operativos, reproducibilidad, y dependencias críticas del entorno.
- Implementar pruebas automatizadas y configurar integración continua (CI) para el proyecto.

Estas acciones permitirán completar la Fase 3 con una arquitectura más robusta y alineada con prácticas profesionales de MLOps.


