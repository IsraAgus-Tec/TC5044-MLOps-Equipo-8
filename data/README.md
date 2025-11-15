# Carpeta data

Esta carpeta contiene toda la información relacionada con las fuentes de datos del proyecto.

## Estructura

### `raw/`
Contiene el dataset original descargado antes de cualquier limpieza o transformación.

### `processed/`
Contiene datasets transformados durante el preprocesamiento:
- `energy_efficiency_clean.csv`: Dataset limpio y listo para experimentación.
- `energy_efficiency_modified.csv`: Transformaciones adicionales utilizadas en pruebas y ajustes.

### `energy_efficiency_final.csv`
Dataset final utilizado por el pipeline de entrenamiento (`energy_efficiency/runner.py`).  
Corresponde a la versión ya limpia y estandarizada lista para modelado.

