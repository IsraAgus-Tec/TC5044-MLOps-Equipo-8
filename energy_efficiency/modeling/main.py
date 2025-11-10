"""
Equipo 8 – Runner principal.
Ejemplo:
    python -m energy_efficiency.modeling.main --target both
"""

from pathlib import Path
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

from .dataset import load_dataset
from .train import train_models
from .predict import evaluate_models
from .config import RANDOM_STATE, TEST_SIZE, TARGET_COLS


def run() -> None:
    df = load_dataset()

    y = df[TARGET_COLS].copy()
    X = df.drop(columns=TARGET_COLS).copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Entrenar
    result = train_models(pd.concat([X_tr, y_tr], axis=1))

    # Evaluar holdout
    eval_df = evaluate_models(result.models, X_te, y_te).sort_values("R2", ascending=False)

    # Exportar resultados rápidos
    out_csv = Path(__file__).resolve().parent / "results_metrics.csv"
    eval_df.to_csv(out_csv, index=False)
    print("\nResultados (holdout):")
    print(eval_df.round(4))
    print(f"\nCSV: {out_csv.resolve()}")
    print("MLflow UI: mlflow ui --port 5000   (http://127.0.0.1:5000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["both", "heating", "cooling"], default="both")  # reservado
    _ = parser.parse_args()
    run()
