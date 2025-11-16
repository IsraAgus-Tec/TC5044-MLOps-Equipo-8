# src/tests/test_data_preprocessor.py
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from handlers.data_loader import DataLoader
from handlers.data_preprocessor import DataPreprocessor


def _load_raw_df():
    loader = DataLoader()
    return loader.getDataFrameFromFile("data/energy_efficiency_modified.csv")


def test_preprocessor_convert_numeric_and_impute():
    df = _load_raw_df()
    preprocessor = DataPreprocessor(df)

    # Convierte a numérico y rellena nulos
    preprocessor.convert_numeric()
    preprocessor.impute_missing()

    # No debe haber valores nulos después de la imputación
    assert preprocessor.df.isnull().sum().sum() == 0


def test_preprocessor_detect_outliers_does_not_empty_dataset():
    df = _load_raw_df()
    preprocessor = DataPreprocessor(df)

    preprocessor.convert_numeric()
    preprocessor.impute_missing()

    rows_before = len(preprocessor.df)
    preprocessor.detect_outliers()
    rows_after = len(preprocessor.df)

    # Debe eliminar algunos outliers, pero no vaciar el dataset
    assert rows_after > 0
    assert rows_after <= rows_before