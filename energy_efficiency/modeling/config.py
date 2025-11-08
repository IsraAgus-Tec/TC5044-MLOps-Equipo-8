"""
Equipo 8 – Configuración del proyecto Energy Efficiency.
Rutas, constantes y parámetros de referencia para entrenamiento y evaluación.
"""

from pathlib import Path

# --- Rutas base del proyecto ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Archivos de datos ---
DATAFILE_NAME = "energy_efficiency_modified.csv"  # mismo nombre en raw/ y processed/

# --- Parámetros generales de modelado ---
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
TARGET_COLS = ["heating_load", "cooling_load"]

# --- Mapeo de nombres (cuando el CSV trae X1..Y2) ---
RENAME_MAP = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}

# --- Columnas numéricas esperadas (post-rename) ---
NUMERIC_COLS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
    "heating_load",
    "cooling_load",
]

# --- MLflow ---
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "Equipo8-EnergyEfficiency"
