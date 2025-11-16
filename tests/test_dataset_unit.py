# tests/test_dataset_unit.py
from pathlib import Path
import pandas as pd

# Ruta al dataset final usado por tu pipeline
DATA_PATH = Path("data/energy_efficiency_final.csv")


def test_dataset_file_exists():
    """El archivo de datos debe existir en la ruta esperada."""
    assert DATA_PATH.exists(), f"No se encontró el archivo: {DATA_PATH}"


def test_dataset_not_empty():
    """El dataset debe tener al menos una fila y una columna."""
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] > 0, "El dataset no tiene filas"
    assert df.shape[1] > 0, "El dataset no tiene columnas"


def test_dataset_expected_columns():
    """
    El dataset debe contener al menos las columnas objetivo Y1 y Y2.
    (Los nombres de las features pueden variar según el preprocesamiento.)
    """
    df = pd.read_csv(DATA_PATH)

    expected_targets = {"Y1", "Y2"}

    missing_targets = expected_targets - set(df.columns)
    assert not missing_targets, f"Faltan columnas objetivo en el dataset: {missing_targets}"


def test_no_missing_values_in_targets():
    """Las variables objetivo Y1 y Y2 no deben tener valores nulos."""
    df = pd.read_csv(DATA_PATH)
    assert df["Y1"].notna().all(), "Hay valores nulos en Y1"
    assert df["Y2"].notna().all(), "Hay valores nulos en Y2"
