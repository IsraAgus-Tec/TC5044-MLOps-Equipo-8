"""
run_experiments.py
Autor: Ricardo Aguilar
Rol: Data Scientist

Descripción:
Pipeline principal de experimentación para el proyecto de eficiencia energética.
- Carga y preprocesa el dataset.
- Entrena tres modelos: Linear Regression, Random Forest, Gradient Boosting.
- Evalúa R², MAE y RMSE (Holdout + 5-Fold Cross-Validation).
- Registra métricas y modelos en MLflow.
- Exporta resultados en formato CSV y HTML.

Probado con:
Python 3.9.6 | scikit-learn 1.3.2 | pandas 2.2.2 | mlflow 2.14.1
"""
from sklearn.impute import SimpleImputer
import argparse
from pathlib import Path

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
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
)

# ========================
# CONFIGURACIÓN GLOBAL
# ========================

SEED = 42
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Energy Efficiency – Ricardo Aguilar"

DATA_PATH = Path("data/energy_efficiency_modified.csv")
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "results_metrics.csv"
OUT_HTML = OUT_DIR / "index.html"


# ========================
# FUNCIONES AUXILIARES
# ========================

def detect_targets(df: pd.DataFrame):
    """
    Detecta las columnas objetivo en el dataset.
    Retorna una lista con las columnas de Heating y Cooling Load.
    """
    candidates = [
        ("heating_load", "cooling_load"),
        ("Heating Load", "Cooling Load"),
        ("Y1", "Y2"),
    ]
    cols = set(map(str, df.columns))
    for a, b in candidates:
        if a in cols and b in cols:
            return [a, b]
    raise ValueError("No se encontraron columnas de target válidas.")


def metric_bundle(y_true, y_pred):
    """Calcula R², MAE y RMSE y devuelve un diccionario de métricas."""
    return {
        "R2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "MAE": float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average")),
        "RMSE": float(
            mean_squared_error(y_true, y_pred, multioutput="uniform_average", squared=False)
        ),
    }


def build_preprocessor(feature_cols):
    """Imputa NaN y estandariza solo columnas numéricas."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def models_zoo(seed=SEED):
    """Define los modelos base a entrenar y devuelve un diccionario."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=600, random_state=seed, n_jobs=-1)
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                learning_rate=0.1, max_depth=3, n_estimators=100, random_state=seed
            )
        ),
    }


def write_dashboard(results: dict):
    """Genera archivos CSV y HTML con los resultados finales."""
    df = pd.DataFrame([{"Model": k, **v} for k, v in results.items()])
    df = df[["Model", "R2", "MAE", "RMSE"]].round(3)
    df.to_csv(OUT_CSV, index=False)

    rows = "\n".join(
        f"<tr><td>{r['Model']}</td><td>{r['R2']:.3f}</td><td>{r['MAE']:.3f}</td><td>{r['RMSE']:.3f}</td></tr>"
        for _, r in df.iterrows()
    )

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Resultados - Energy Efficiency</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
  h1 {{ color: #003B5C; font-size: 26px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
  th {{ background: #0b3d60; color: #fff; padding: 10px; text-align: center; }}
  td {{ padding: 10px; text-align: center; border-top: 1px solid #e6e8eb; }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
</style>

<h1>Resultados de Experimentos – Energy Efficiency</h1>
<table>
  <thead><tr><th>Modelo</th><th>R²</th><th>MAE</th><th>RMSE</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p style="color:#6b7280; font-size:13px;">Generado automáticamente por MLflow – Ricardo Aguilar</p>
"""
    OUT_HTML.write_text(html, encoding="utf-8")


def log_mlflow(model_name, metrics, pipeline):
    """Registra métricas y modelo en MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags(
            {"author": "Ricardo Aguilar", "dataset": "Energy Efficiency", "role": "Data Scientist"}
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")


# ========================
# EJECUCIÓN PRINCIPAL
# ========================

def main(target_choice="both", test_size=0.2):
    """Ejecuta el flujo completo de experimentación."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    targets = detect_targets(df)
    # Mantén solo variables numéricas (evita errores por columnas mixtas)
    y = df[[c for c in df.columns if c in targets]].copy()
    X = df[[c for c in df.columns if c not in targets]].copy()

    # 1️⃣ Solo columnas numéricas
    feature_cols = X.select_dtypes(include=[np.number]).columns
    X = X[feature_cols]

    # 2️⃣ Si Y tiene NaN, los descartamos
    mask = ~y.isna().any(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Airbag por si algún NaN sobrevive (no debería, pero garantizamos)
    X = X.fillna(X.median(numeric_only=True))


    if target_choice == "heating":
        y_cols = [targets[0]]
    elif target_choice == "cooling":
        y_cols = [targets[1]]
    else:
        y_cols = targets

    y = df[y_cols].copy()
    X = df[[c for c in df.columns if c not in y_cols]].copy()

    # Mantén solo variables numéricas (evita 'mixed_type_col' con strings)
    feature_cols = X.select_dtypes(include=[np.number]).columns
    X = X[feature_cols]

    preprocessor = build_preprocessor(feature_cols)
    models = models_zoo()

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scorer_r2 = "r2"
    scorer_mae = make_scorer(mean_absolute_error, greater_is_better=False)
    scorer_rmse = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

    results = {}

    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=SEED
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        holdout_metrics = metric_bundle(y_test, y_pred)

        cv_r2 = cross_val_score(pipe, X, y, cv=kf, scoring=scorer_r2).mean()
        cv_mae = -cross_val_score(pipe, X, y, cv=kf, scoring=scorer_mae).mean()
        cv_rmse = -cross_val_score(pipe, X, y, cv=kf, scoring=scorer_rmse).mean()

        results[name] = {
            "R2": holdout_metrics["R2"],
            "MAE": holdout_metrics["MAE"],
            "RMSE": holdout_metrics["RMSE"],
        }

        log_mlflow(
            model_name=name,
            metrics={
                "R2": holdout_metrics["R2"],
                "MAE": holdout_metrics["MAE"],
                "RMSE": holdout_metrics["RMSE"],
                "cv_R2_mean": cv_r2,
                "cv_MAE_mean": cv_mae,
                "cv_RMSE_mean": cv_rmse,
            },
            pipeline=pipe,
        )

    write_dashboard(results)

    print("\nResultados promedio (Holdout):")
    for m, v in results.items():
        print(f"{m:>20s} | R2={v['R2']:.3f}  MAE={v['MAE']:.3f}  RMSE={v['RMSE']:.3f}")
    print(f"\nCSV:  {OUT_CSV.resolve()}")
    print(f"HTML: {OUT_HTML.resolve()}")
    print("MLflow UI: mlflow ui --port 5000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Energy Efficiency experiments")
    parser.add_argument(
        "--target",
        choices=["heating", "cooling", "both"],
        default="both",
        help="Qué objetivo entrenar (heating, cooling o both)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporción para el conjunto de prueba (0.0–1.0)",
    )
    args = parser.parse_args()

    # Asegura MLflow a la carpeta local y el experimento esperado
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Ejecuta
    main(target_choice=args.target, test_size=args.test_size)

