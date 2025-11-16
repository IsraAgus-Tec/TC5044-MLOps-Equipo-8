"""
predict.py

Descripción:
Script de INFERENCIA para la Fase 3.

- Carga el dataset final de eficiencia energética.
- Separa variables de entrada (X) y la variable objetivo Y1.
- Construye un pipeline de preprocesamiento (Imputer + StandardScaler).
- Entrena un modelo GradientBoostingRegressor (el mejor para Y1 según resultados_metrics.csv).
- Realiza una predicción de ejemplo sobre el primer registro del dataset.
- Imprime en consola:
    - El registro de entrada usado.
    - El valor predicho de Y1.

Nota:
Para simplificar la demostración de inferencia, este script:
- Vuelve a entrenar el modelo localmente usando TODO el dataset.
- En un escenario productivo, en lugar de reentrenar, se cargaría el modelo
  ya registrado en MLflow o almacenado como artefacto en producción.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def load_data(data_path: Path) -> pd.DataFrame:
    """Carga el dataset de eficiencia energética desde un CSV."""
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    df = pd.read_csv(data_path)
    # Validamos que existan las columnas objetivo esperadas
    if "Y1" not in df.columns or "Y2" not in df.columns:
        raise ValueError(
            "El dataset no contiene las columnas objetivo esperadas 'Y1' y 'Y2'. "
            f"Columnas encontradas: {list(df.columns)}"
        )
    return df


def build_inference_pipeline(feature_names: list[str]) -> Pipeline:
    """
    Construye el pipeline de preprocesamiento + modelo para inferencia.

    - Imputer (mediana) + StandardScaler para todas las variables numéricas.
    - GradientBoostingRegressor como modelo final (mejor para Y1).
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_names),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def run_inference(data_path: Path) -> None:
    """
    Ejecuta el flujo de inferencia:

    1. Carga el dataset.
    2. Separa X (features) e Y1 (target).
    3. Ajusta el pipeline sobre todo el dataset.
    4. Toma un ejemplo (primer registro) y obtiene la predicción.
    5. Imprime resultados en consola.
    """
    print("[BOOT] Script de inferencia iniciado")
    print(f"[INFO] Leyendo datos desde: {data_path}")

    df = load_data(data_path)

    # Usamos todas las columnas excepto Y1 y Y2 como features
    feature_cols = [c for c in df.columns if c not in ("Y1", "Y2")]
    X = df[feature_cols].copy()
    y = df["Y1"].copy()

    print(f"[INFO] Número de observaciones: {len(df)}")
    print(f"[INFO] Features usadas para el modelo: {feature_cols}")

    # Construimos el pipeline y lo entrenamos
    pipeline = build_inference_pipeline(feature_cols)
    pipeline.fit(X, y)

    # Tomamos un ejemplo para inferencia (primer registro)
    example = X.iloc[[0]]  # DataFrame con una sola fila
    pred = pipeline.predict(example)[0]

    # Mostramos el ejemplo de entrada y la predicción
    print("\n[INPUT] Ejemplo de registro usado para la predicción:")
    print(example.to_dict(orient="records")[0])

    print(f"\n[PRED] Predicción de Y1 (Heating Load) para ese registro: {pred:.4f}")
    print("\n[OK] Inferencia completada correctamente.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script de inferencia para el modelo de eficiencia energética (Y1)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/energy_efficiency_final.csv",
        help="Ruta al archivo CSV con el dataset final.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data)
    run_inference(data_path)
