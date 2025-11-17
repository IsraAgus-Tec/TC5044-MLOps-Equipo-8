# src/tests/test_pipeline_integration.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import main  # usa el mismo main.py que ejecutas a mano


def test_full_pipeline_creates_cleansed_dataset():
    # Ejecutamos el pipeline completo sin Visual EDA
    main(showVisualEDA=False)

    # Verificamos que el archivo “cleansed” exista en data/interim/cleansed
    cleansed_path = (
        PROJECT_ROOT / "data" / "interim" / "cleansed" / "energy_efficiency_modified.csv"
    )
    assert cleansed_path.exists()
    assert cleansed_path.is_file()
