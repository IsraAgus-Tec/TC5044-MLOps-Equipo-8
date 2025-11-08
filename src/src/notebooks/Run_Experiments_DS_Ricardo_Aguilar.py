"""
Run_Experiments_DS_Ricardo_Aguilar.py
Autor: Ricardo Aguilar (Data Scientist)

- Carga src/data/energy_efficiency_modified.csv
- Entrena 3 modelos (Linear, Random Forest, Gradient Boosting) con Pipelines
- Holdout + 5-fold CV (R2, MAE, RMSE)
- Genera results_metrics.csv e index.html (dashboard)
- Registra runs en MLflow (./mlruns)

Probado con: Python 3.9.6, numpy 1.26.4, scipy 1.10.1,
scikit-learn 1.3.2, pandas 2.2.2, mlflow 2.14.1
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, make_scorer
)
# === Rutas base del proyecto ===
from pathlib import Path

# Intentamos primero ruta relativa (desde la raíz del repo)
DATA_PATH = Path("data/processed/energy_efficiency_modified.csv")

# Si no existe, tratamos de resolverlo desde donde está este archivo
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "energy_efficiency_modified.csv"

# Si aún así no existe, lanzamos error con la ruta absoluta
if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset en: {DATA_PATH.resolve()}")

# ------------------------------------------------------------------#
# Config
# ------------------------------------------------------------------#
SEED = 42
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Energy Efficiency – Ricardo Aguilar"
BRANCH_TAG = "feature/datascientist-ricardoaguilar"

# Guardar todos los resultados en el mismo directorio que este script
BASE_DIR = Path(__file__).resolve().parent

OUT_CSV = BASE_DIR / "results_metrics.csv"
OUT_HTML = BASE_DIR / "index.html"
P_LINEAR = BASE_DIR / "params_linear.json"
P_RF = BASE_DIR / "params_random_forest.json"
P_GB = BASE_DIR / "params_gradient_boosting.json"


# ------------------------------------------------------------------#
# Utils
# ------------------------------------------------------------------#
def detect_targets(df: pd.DataFrame):
    """Detecta nombres reales de los targets en el CSV."""
    candidates = [
        ("heating_load", "cooling_load"),
        ("Heating Load", "Cooling Load"),
        ("Y1", "Y2"),
    ]
    cols = set(map(str, df.columns))
    for a, b in candidates:
        if a in cols and b in cols:
            return [a, b]
    raise ValueError(
        "No se encontraron columnas de target (heating/cooling). "
        f"Cols vistas: {list(df.columns)[:15]} ..."
    )


def metric_bundle(y_true, y_pred):
    """R2, MAE y RMSE promedio multi-salida."""
    return {
        "R2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "MAE": float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average")),
        "RMSE": float(mean_squared_error(y_true, y_pred, multioutput="uniform_average", squared=False)),
    }


def cv_mean(estimator, X, y, cv, scorer):
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scorer, error_score="raise")
    return float(np.mean(scores))


def build_preprocessor(feature_cols):
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def models_zoo():
    lin = LinearRegression()
    rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=600, random_state=SEED, n_jobs=-1
    ))
    gb = MultiOutputRegressor(GradientBoostingRegressor(
        learning_rate=0.1, max_depth=3, n_estimators=100, random_state=SEED
    ))
    return {
        "Linear Regression": (lin, {"fit_intercept": True, "normalize": False}),
        "Random Forest": (rf, {"n_estimators": 600, "max_depth": None, "random_state": SEED}),
        "Gradient Boosting": (gb, {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100, "random_state": SEED}),
    }


def dump_param_jsons():
    with open(P_LINEAR, "w") as f:
        json.dump({"model": "LinearRegression",
                   "params": {"fit_intercept": True, "normalize": False},
                   "data": {"cv_folds": 5}}, f, indent=2)
    with open(P_RF, "w") as f:
        json.dump({"model": "RandomForestRegressor",
                   "params": {"n_estimators": 600, "max_depth": None, "random_state": SEED},
                   "data": {"cv_folds": 5}}, f, indent=2)
    with open(P_GB, "w") as f:
        json.dump({"model": "GradientBoostingRegressor",
                   "params": {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100, "random_state": SEED},
                   "data": {"cv_folds": 5}}, f, indent=2)


def write_dashboard(results: dict):
    df = pd.DataFrame(
        [{"Model": k, **v} for k, v in results.items()]
    )[["Model", "R2", "MAE", "RMSE"]]

    # redondeo para tabla
    df["R2"] = df["R2"].round(3)
    df["MAE"] = df["MAE"].round(3)
    df["RMSE"] = df["RMSE"].round(3)
    df.to_csv(OUT_CSV, index=False)

    rows = "\n".join(
        f"<tr><td>{r['Model']}</td><td>{r['R2']:.3f}</td><td>{r['MAE']:.3f}</td><td>{r['RMSE']:.3f}</td></tr>"
        for _, r in df.iterrows()
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Resultados de Experimentos - Ricardo Aguilar</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px}}
h1{{font-size:40px;margin:0 0 8px}}
table{{width:100%;border-collapse:collapse;margin-top:16px}}
th{{background:#2e4156;color:#fff;padding:14px;text-align:center}}
td{{padding:18px 12px;text-align:center;border-top:1px solid #e9eef3;font-size:18px}}
tr:nth-child(even) td{{background:#f6f9fc}}
.caption{{margin-top:16px;color:#6b7280;font-size:14px}}
</style>
<h1>Resultados de Experimentos - Ricardo Aguilar</h1>
<table>
  <thead><tr><th>Modelo</th><th>R²</th><th>MAE</th><th>RMSE</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p class="caption">Generado automáticamente a partir de artefactos MLflow y CSVs (Ricardo Aguilar).</p>
"""
    OUT_HTML.write_text(html, encoding="utf-8")


def log_mlflow(model_name: str, params_path: Path, metrics: dict, pipeline: Pipeline):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({
            "author": "Ricardo Aguilar",
            "role": "Data Scientist",
            "branch": BRANCH_TAG,
            "dataset": "Energy Efficiency",
            "script": "Run_Experiments_DS_Ricardo_Aguilar.py",
        })
        if params_path.exists():
            with open(params_path) as f:
                pj = json.load(f)
            mlflow.log_params({f"param__{k}": v for k, v in pj.get("params", {}).items()})
            mlflow.log_param("cv_folds", pj.get("data", {}).get("cv_folds", 5))
        mlflow.log_metrics(metrics)
        if OUT_CSV.exists(): mlflow.log_artifact(str(OUT_CSV))
        if OUT_HTML.exists(): mlflow.log_artifact(str(OUT_HTML))
        if params_path.exists(): mlflow.log_artifact(str(params_path))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")


# ------------------------------------------------------------------#
# Main
# ------------------------------------------------------------------#
def main(target_choice: str = "both", test_size: float = 0.2):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    detected_targets = detect_targets(df)

    if target_choice == "heating":
        target_cols = [detected_targets[0]]
    elif target_choice == "cooling":
        target_cols = [detected_targets[1]]
    else:
        target_cols = detected_targets

    y = df[target_cols].copy()
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].copy()

    pre = build_preprocessor(feature_cols)
    zoo = models_zoo()
    dump_param_jsons()

    # scorers para CV
    scorer_r2 = "r2"
    scorer_mae = make_scorer(mean_absolute_error, greater_is_better=False, multioutput="uniform_average")
    scorer_rmse = make_scorer(mean_squared_error, greater_is_better=False, multioutput="uniform_average", squared=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}

    for name, (estimator, _) in zoo.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", estimator)])

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=SEED)
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        hold = metric_bundle(yte, yhat)

        cv_r2 = cv_mean(pipe, X, y, kf, scorer_r2)
        cv_mae = -cv_mean(pipe, X, y, kf, scorer_mae)
        cv_rmse = -cv_mean(pipe, X, y, kf, scorer_rmse)

        results[name] = {k: round(v, 6) for k, v in hold.items()}

        params_path = P_LINEAR if "Linear" in name else (P_RF if "Random" in name else P_GB)
        log_mlflow(
            name,
            params_path,
            {"R2": hold["R2"], "MAE": hold["MAE"], "RMSE": hold["RMSE"],
             "cv_R2_mean": cv_r2, "cv_MAE_mean": cv_mae, "cv_RMSE_mean": cv_rmse},
            pipe,
        )

    write_dashboard(results)

    print("\n=== Resultados holdout (promedio multi-output) ===")
    for m, vals in results.items():
        print(f"{m:>20s} | R2={vals['R2']:.3f}  MAE={vals['MAE']:.3f}  RMSE={vals['RMSE']:.3f}")
    print(f"\nCSV:   {OUT_CSV.resolve()}\nHTML:  {OUT_HTML.resolve()}")
    print("MLflow UI: mlflow ui --port 5000  -> http://127.0.0.1:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["heating", "cooling", "both"], default="both")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(target_choice=args.target, test_size=args.test_size)
