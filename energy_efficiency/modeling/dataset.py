"""
Carga y normalización del dataset para el proyecto Energy Efficiency – Equipo 8.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from .config import (
    PROCESSED_DIR, RAW_DIR, DATAFILE_NAME, RENAME_MAP, NUMERIC_COLS
)


def _candidate_paths() -> list[Path]:
    return [
        PROCESSED_DIR / DATAFILE_NAME,
        RAW_DIR / DATAFILE_NAME,
    ]


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Carga el CSV desde processed/ o raw/ (en ese orden). Aplica renombrado si procede,
    fuerza tipos numéricos y elimina duplicados.
    """
    csv_path = path
    if csv_path is None:
        for p in _candidate_paths():
            if p.exists():
                csv_path = p
                break
    if csv_path is None:
        raise FileNotFoundError("No se encontró el archivo de datos en raw/ ni processed/.")

    df = pd.read_csv(csv_path)

    # Renombrado si las columnas base existen.
    needs_rename = any(col in df.columns for col in RENAME_MAP.keys())
    if needs_rename:
        df = df.rename(columns=RENAME_MAP)

    # Forzar numéricos en las columnas declaradas (si existen)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Limpieza básica
    df = df.drop_duplicates().reset_index(drop=True)
    return df
