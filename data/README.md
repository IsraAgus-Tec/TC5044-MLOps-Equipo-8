# Carpeta data

Esta carpeta contiene toda la información relacionada con las fuentes de datos del proyecto.

## Estructura

### `raw/`
Contiene el dataset original descargado antes de cualquier limpieza o transformación.

### `external/`
Espacio reservado para fuentes externas (APIs, terceros).  
Se inicializa con un `.gitkeep` para mantener la estructura de Cookiecutter.

### `interim/`
Almacena datos intermedios generados por el pipeline:
- `energy_efficiency_modified.csv` y su versión DVC (`.dvc`).
- `cleansed/` → dataset limpio exportado por `src/main.py` y sincronizado con DVC.

### `processed/`
Contiene datasets transformados finales:
- `energy_efficiency_clean.csv`: Dataset limpio y listo para experimentación.
- `energy_efficiency_modified.csv`: Transformaciones adicionales utilizadas en pruebas y ajustes previos.

### `energy_efficiency_final.csv`
Dataset final utilizado por el pipeline de entrenamiento (`src/main.py`).  
Corresponde a la versión ya limpia y estandarizada lista para modelado.
