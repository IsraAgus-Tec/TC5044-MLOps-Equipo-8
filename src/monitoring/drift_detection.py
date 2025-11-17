from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from handlers.data_loader import DataLoader
except ModuleNotFoundError:  # pragma: no cover - executed via python -m
    from ..handlers.data_loader import DataLoader  # type: ignore

DEFAULT_DATASET = "data/energy_efficiency_final.csv"
DEFAULT_MODEL = Path("models/energy_efficiency_rf.joblib")
DEFAULT_REPORT_DIR = Path("reports/drift")
RANDOM_STATE = 42
THRESHOLD_DROP_R2 = 0.05
THRESHOLD_INCREASE_RMSE = 0.01

FEATURE_COLUMNS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]
TARGET_COLUMNS = ["heating_load", "cooling_load"]
DATA_LOADER = DataLoader()


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = DATA_LOADER.getDataFrameFromFile(str(path))
    missing = [col for col in FEATURE_COLUMNS + TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"El dataset no contiene las columnas requeridas: {missing}")
    return df


def induce_drift(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift_df = df.copy()

    # Shift overall height (X5) mean upwards by 20%
    drift_df["overall_height"] = drift_df["overall_height"] * 1.2

    # Orientation (X6) seasonal shift - rotate +/- 2 positions
    drift_df["orientation"] = (
        (drift_df["orientation"] + rng.integers(-2, 3, size=len(df))) % 8
    ) + 1

    # Introduce glazing area anomalies on 40% of the rows
    anomalous_idx = drift_df.sample(frac=0.4, random_state=seed).index
    scales = rng.uniform(0.3, 1.8, size=len(anomalous_idx))
    drift_df.loc[anomalous_idx, "glazing_area"] = (
        drift_df.loc[anomalous_idx, "glazing_area"].values * scales
    )

    # Missingness on X2 (surface area) for 15% of samples
    missing_idx = drift_df.sample(frac=0.15, random_state=seed + 1).index
    drift_df.loc[missing_idx, "surface_area"] = drift_df["surface_area"].median()

    return drift_df


def evaluate(model, df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]
    preds = model.predict(X)
    mse = mean_squared_error(y, preds, multioutput="uniform_average")
    metrics = {
        "r2": float(r2_score(y, preds, multioutput="uniform_average")),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, preds, multioutput="uniform_average")),
    }
    return metrics


def compare_metrics(baseline: dict, drift: dict) -> dict:
    delta = {k: drift[k] - baseline[k] for k in baseline.keys()}
    alert = (
        (baseline["r2"] - drift["r2"] >= THRESHOLD_DROP_R2)
        or (drift["rmse"] - baseline["rmse"] >= THRESHOLD_INCREASE_RMSE)
    )
    action = "Re-entrenar y revisar pipeline de features" if alert else "Continuar monitoreando"
    return {"delta": delta, "alert": alert, "recommended_action": action}


def persist_report(report_dir: Path, payload: dict, drift_df: pd.DataFrame):
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "drift_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sample_path = report_dir / "drift_sample.csv"
    drift_df.head(50).to_csv(sample_path, index=False)
    print(f"[drift] Reporte guardado en {report_path}")
    print(f"[drift] Ejemplo de dataset con drift guardado en {sample_path}")


def main(dataset: Path, model_path: Path, report_dir: Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo serializado en {model_path}")

    model = load(model_path)
    baseline_df = load_dataset(dataset)
    baseline_metrics = evaluate(model, baseline_df)
    drift_df = induce_drift(baseline_df)
    drift_metrics = evaluate(model, drift_df)
    comparison = compare_metrics(baseline_metrics, drift_metrics)

    payload = {
        "dataset": dataset.as_posix(),
        "model_path": model_path.as_posix(),
        "baseline_metrics": baseline_metrics,
        "drift_metrics": drift_metrics,
        **comparison,
        "thresholds": {
            "drop_r2": THRESHOLD_DROP_R2,
            "increase_rmse": THRESHOLD_INCREASE_RMSE,
        },
    }
    persist_report(report_dir, payload, drift_df)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación y monitoreo de data drift")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    main(args.dataset, args.model, args.report_dir)
