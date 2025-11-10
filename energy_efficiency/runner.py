"""
runner.py
Autor: Ricardo Aguilar
Rol: Data Scientist

Descripción:
Pipeline principal de experimentación para el proyecto de eficiencia energética.
- Carga y preprocesa el dataset.
- Entrena tres modelos: Linear Regression, Random Forest, Gradient Boosting.
- Evalúa R², MAE y RMSE (Holdout + 5-Fold Cross-Validation).
- Registra métricas y modelos en MLflow (backend SQLite para evitar el warning).
- Exporta resultados en formato CSV y HTML.

Probado con:
Python 3.9+ | scikit-learn >=1.3 | pandas >=2.2 | mlflow >=2.14
"""

from __future__ import annotations
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# -------------------------
# Configuración por defecto
# -------------------------
DEFAULT_DATA = "src/data/energy_efficiency_modified.csv"  # cámbialo si tu CSV está en otro lugar
EXPERIMENT_NAME = "Energy Efficiency – Ricardo Aguilar"
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2

# MLflow: usa SQLite para evitar el FutureWarning del filesystem backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path.resolve()}")
    df = pd.read_csv(path)
    # Normaliza infinities -> NaN (para que el Imputer los procese)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def split_xy(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        # soporta nombres comunes del dataset ENB2012
        mapping = {"y1": ["Y1", "y1", "Heating Load", "Heating_Load"],
                   "y2": ["Y2", "y2", "Cooling Load", "Cooling_Load"]}
        # intenta mapear si pidieron y1/y2
        for k, aliases in mapping.items():
            if target_col.lower() == k and any(a in df.columns for a in aliases):
                target_col = next(a for a in aliases if a in df.columns)
                break
    if target_col not in df.columns:
        raise ValueError(f"No existe la columna objetivo '{target_col}' en el dataset.\n"
                         f"Columnas disponibles: {list(df.columns)}")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Quita filas donde y es NaN (no se puede entrenar sin etiqueta)
    mask_y = y.notna()
    if mask_y.sum() < len(y):
        print(f"[INFO] Filas eliminadas por NaN en y='{target_col}': {len(y) - mask_y.sum()}")
    X = X.loc[mask_y].reset_index(drop=True)
    y = y.loc[mask_y].reset_index(drop=True)
    return X, y


def build_preprocessor(feature_names):
    # Todas las columnas se tratan como numéricas para este dataset
    numeric_features = list(feature_names)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
        n_jobs=None
    )
    return preprocessor


def make_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def evaluate_and_log(model_name, pipeline, X_train, X_test, y_train, y_test, target_name):
    with mlflow.start_run(run_name=f"{model_name} | target={target_name}"):
        # Entrena
        pipeline.fit(X_train, y_train)

        # Predicción holdout
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # CV 5-fold sobre el train
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        # Neg MAE y Neg RMSE (convertimos a positivos)
        cv_r2 = cross_val_score(pipeline, X_train, y_train, scoring="r2", cv=kf)
        cv_mae = -cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_absolute_error", cv=kf)
        cv_rmse = (-cross_val_score(pipeline, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf))

        # Log métricas
        mlflow.log_metric("holdout_r2", r2)
        mlflow.log_metric("holdout_mae", mae)
        mlflow.log_metric("holdout_rmse", rmse)
        mlflow.log_metric("cv_r2_mean", float(np.mean(cv_r2)))
        mlflow.log_metric("cv_r2_std", float(np.std(cv_r2)))
        mlflow.log_metric("cv_mae_mean", float(np.mean(cv_mae)))
        mlflow.log_metric("cv_mae_std", float(np.std(cv_mae)))
        mlflow.log_metric("cv_rmse_mean", float(np.mean(cv_rmse)))
        mlflow.log_metric("cv_rmse_std", float(np.std(cv_rmse)))

        # Log params básicos del modelo
        try:
            mlflow.log_params({k: v for k, v in pipeline.named_steps["model"].get_params().items()
                               if isinstance(v, (int, float, str, bool))})
        except Exception:
            pass

        # Guarda el modelo
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Devuelve dict con resultados
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
    # Map ea. "y1"/"y2" a nombre real de columna si aplica
    target_map = {"y1": "Y1", "y2": "Y2"}
    target_name = target_map.get(target_code.lower(), target_code)

    X, y = split_xy(df, target_name)

    # Recuento NaN inicial en X
    nan_count_before = int(np.isnan(X.to_numpy(dtype=float, copy=False)).sum()) if len(X) else 0
    print(f"[INFO] NaN en X antes del preprocesamiento (target={target_name}): {nan_count_before}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    pre = build_preprocessor(X.columns)
    models = make_m_
