"""
runner.py
Autor: Ricardo Aguilar
Rol: Data Scientist

Descripción:
Pipeline principal de experimentación para el proyecto de eficiencia energética.
- Carga y preprocesa el dataset.
- Entrena tres modelos: Linear Regression, Random Forest, Gradient Boosting.
- Evalúa R², MAE y RMSE (Holdout + 5-Fold Cross-Validation).
- Registra métricas y modelos en MLflow (tracking en carpeta local 'mlruns').
- Exporta resultados en CSV y HTML (src/notebooks).

Probado con:
Python 3.11+ | scikit-learn 1.3–1.7 | pandas 2.2–2.3 | mlflow 2.14+
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
$
# -----------------------------
# Reproducibilidad global
# -----------------------------
import numpy as np
import random

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# -----------------------------
# Rutas y configuración
# -----------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Dataset por default (ajústalo si lo necesitas)
DEFAULT_DATA = (REPO / "data" / "energy_efficiency_final.csv").as_posix()

# Carpeta de salida
OUT_DIR = REPO / "src" / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow local (en carpeta 'mlruns' del repo)
MLRUNS_DIR = REPO / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.as_posix()}")

EXPERIMENT_NAME = "Energy Efficiency – Ricardo Aguilar"
N_SPLITS = 5
TEST_SIZE = 0.2

mlflow.set_experiment(EXPERIMENT_NAME)

# MLflow -> URI ABSOLUTA al 'mlruns' del repo (no depende del CWD)
MLRUNS_DIR = REPO / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
TRACKING_URI = f"file://{MLRUNS_DIR.as_posix()}"
EXPERIMENT_NAME = "Energy Efficiency – Ricardo Aguilar"

# Inicializa MLflow una sola vez
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# -----------------------------
# Utilidades de datos
# -----------------------------
def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path.resolve()}")
    df = pd.read_csv(path)
    # Sanitizar infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    # Eliminar columnas no relevantes
    drop_cols = [c for c in df.columns if "mixed" in c.lower()]
    if drop_cols:
        print(f"[INFO] Columnas eliminadas: {drop_cols}")
        df = df.drop(columns=drop_cols)



    return df


def split_xy(df: pd.DataFrame, target_col: str):
    """
    Soporta alias comunes: y1->Y1 (Heating), y2->Y2 (Cooling)
    """
    # Normalizar nombre de target
    aliases = {
        "y1": ["Y1", "y1", "Heating Load", "Heating_Load"],
        "y2": ["Y2", "y2", "Cooling Load", "Cooling_Load"],
    }
    tc_lower = target_col.lower()
    if tc_lower in aliases:
        for cand in aliases[tc_lower]:
            if cand in df.columns:
                target_col = cand
                break

    if target_col not in df.columns:
        raise ValueError(
            f"No existe la columna objetivo '{target_col}' en el dataset.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Filtrar filas con NaN en y
    mask_y = y.notna()
    if mask_y.sum() < len(y):
        print(f"[INFO] Filas eliminadas por NaN en y='{target_col}': {len(y) - mask_y.sum()}")
    X = X.loc[mask_y].reset_index(drop=True)
    y = y.loc[mask_y].reset_index(drop=True)
    return X, y, target_col

def make_models():
    return {
        "LinearRegression": LinearRegression(),  # sin random_state
        "RandomForest": RandomForestRegressor(
            n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        ),
    }
# --- Preprocesamiento ---
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- Modelos ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

def build_preprocessor(X_df):
    # Tomamos solo columnas numéricas
    numeric_features = [c for c in X_df.columns if np.issubdtype(X_df[c].dtype, np.number)]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pre = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
    )
    return pre

def make_pipelines(X_df):
    pre = build_preprocessor(X_df)

    lr_pipe = Pipeline([
        ("preprocessor", pre),
        ("model", LinearRegression())
    ])

    rf_pipe = Pipeline([
        ("preprocessor", pre),
        ("model", RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        ))
    ])

    gb_pipe = Pipeline([
        ("preprocessor", pre),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=4,
            random_state=RANDOM_STATE
        ))
    ])

    models = [
        ("Linear Regression", lr_pipe),
        ("Random Forest", rf_pipe),
        ("Gradient Boosting", gb_pipe),
    ]
    return models

# -----------------------------
# Entrenamiento + logging
# -----------------------------
def evaluate_and_log(
    model_name: str,
    model: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    target_name: str,
):
    """
    Entrena el pipeline, calcula métricas holdout y CV, y las registra en MLflow.
    Devuelve un dict con métricas agregadas.
    """
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name=f"{model_name} | target={target_name}"):
        # Entrenamiento
        model.fit(X_train, y_train)

        # Holdout
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred, squared=False))

        # Cross-Validation (en el set de entrenamiento)
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_r2 = cross_val_score(model, X_train, y_train, scoring="r2", cv=kf)
        cv_mae = -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_absolute_error", cv=kf
        )
        cv_rmse = -cross_val_score(
            model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf
        )

        # Tags y métricas
        mlflow.set_tags(
            {
                "author": "Ricardo Aguilar",
                "dataset": "Energy Efficiency",
                "target": target_name,
                "role": "Data Scientist",
            }
        )
        mlflow.log_metrics(
            {
                "holdout_r2": r2,
                "holdout_mae": mae,
                "holdout_rmse": rmse,
                "cv_r2_mean": float(np.mean(cv_r2)),
                "cv_r2_std": float(np.std(cv_r2)),
                "cv_mae_mean": float(np.mean(cv_mae)),
                "cv_mae_std": float(np.std(cv_mae)),
                "cv_rmse_mean": float(np.mean(cv_rmse)),
                "cv_rmse_std": float(np.std(cv_rmse)),
            }
        )
        mlflow.sklearn.log_model(model, f"{model_name}_{target_name}")

        # (Autolog ya registra el modelo y los parámetros del estimador)

        return {
            "target": target_name,
            "model": model_name,
            "holdout_r2": r2,
            "holdout_mae": mae,
            "holdout_rmse": rmse,
            "cv_r2_mean": float(np.mean(cv_r2)),
            "cv_mae_mean": float(np.mean(cv_mae)),
            "cv_rmse_mean": float(np.mean(cv_rmse)),
        }


def run_for_target(df: pd.DataFrame, target_code: str, test_size: float) -> pd.DataFrame:
    X, y, target_name = split_xy(df, target_code)

    # Split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    # Construir los pipelines con el preprocesador interno
    models = make_pipelines(X)  # <- regresa una lista de (name, pipeline)

    rows = []
    for name, model in models:   # <- SIN .items()
        metrics = evaluate_and_log(name, model, X_train, X_test, y_train, y_test, target_name)
        rows.append(metrics)

    return pd.DataFrame(rows)


def save_results(df_results: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_metrics.csv"
    html_path = out_dir / "results_metrics.html"
    df_results.to_csv(csv_path, index=False)
    df_results.to_html(html_path, index=False)
    print(f"[OK] Resultados guardados en:\n- {csv_path}\n- {html_path}")


# -----------------------------
# Main
# -----------------------------
def main(target_choice: str, data_path: str, test_size: float):
    print("[BOOT] Entré a main()")
    df = load_dataset(data_path)

    # Normalizar nombres si vienen como Heating/Cooling
    if "Y1" not in df.columns and "Heating Load" in df.columns:
        df = df.rename(columns={"Heating Load": "Y1"})
    if "Y2" not in df.columns and "Cooling Load" in df.columns:
        df = df.rename(columns={"Cooling Load": "Y2"})

    all_results = []

    if target_choice.lower() in ("y1", "y2"):
        res = run_for_target(df, target_choice.lower(), test_size)
        all_results.append(res)
    elif target_choice.lower() == "both":
        res1 = run_for_target(df, "y1", test_size)
        res2 = run_for_target(df, "y2", test_size)
        all_results.extend([res1, res2])
    else:
        raise ValueError("Parámetro --target inválido. Usa: y1, y2 o both")

    final_df = pd.concat(all_results, ignore_index=True)
    save_results(final_df, OUT_DIR)
    print("[OK] Ejecución completa.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner de experimentos – Energy Efficiency")
    parser.add_argument("--target", type=str, default="both", help="y1 | y2 | both")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Ruta al CSV de datos")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporción del test split (por defecto 20%)")
    args = parser.parse_args()
    main(target_choice=args.target, data_path=args.data, test_size=args.test_size)
