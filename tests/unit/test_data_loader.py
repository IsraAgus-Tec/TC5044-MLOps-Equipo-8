# src/tests/test_data_loader.py
import os
import sys
from pathlib import Path

# Asegurar que src/ esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from handlers.data_loader import DataLoader


def test_get_dataframe_from_file_loads_csv():
    loader = DataLoader()
    df = loader.getDataFrameFromFile("data/interim/energy_efficiency_modified.csv")

    # Debe regresar un DataFrame no vacío
    assert df is not None
    assert not df.empty

    # Debe tener al menos 2 columnas
    assert df.shape[1] >= 2

    # El archivo debe existir en data/interim
    csv_path = PROJECT_ROOT / "data" / "interim" / "energy_efficiency_modified.csv"
    assert csv_path.exists()
