"""Runner script for Energy Efficiency project."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_ERROR_MESSAGE = (
    "Dataset no encontrado. Debe estar en data/energy_efficiency_modified.csv o en "
    "data/processed/energy_efficiency_modified.csv"
)
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
EXPERIMENT_NAME = "Energy Efficiency – Ricardo Aguilar"
TRACKING_URI = "file:./mlruns"
TARGET_PRIORITIES = (["Y1", "Y2"], ["Heating Load", "Cooling Load"])

@dataclass
class ModelSpec:
    name: str
    estimator: object
    params: Dict[str, object]

def load_data(dataset_paths: Iterable[Path]) -> pd.DataFrame:
    for path in dataset_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                raise RuntimeError(f"Error al cargar el dataset en {path}: {exc}") from exc
            if df.empty:
                raise ValueError("El dataset está vacío")
            return df

    dvc_hints: List[str] = []
    for path in dataset_paths:
        dvc_candidate = path.with_suffix(path.suffix + ".dvc")
        if dvc_candidate.exists():
            dvc_hints.append(str(dvc_candidate))

    error_message = DATASET_ERROR_MESSAGE
    if dvc_hints:
        joined_hints = ", ".join(dvc_hints)
        error_message = (
            f"{DATASET_ERROR_MESSAGE}. "
            f"Se detectaron archivos DVC asociados ({joined_hints}). "
            "Ejecuta 'dvc pull' para descargarlos o coloca manualmente el CSV."
        )

    raise FileNotFoundError(error_message)

def detect_targets(df: pd.DataFrame) -> List[str]:
    columns = set(df.columns)
    for candidates in TARGET_PRIORITIES:
        if all(target in columns for target in candidates):
            return list(candidates)
    raise ValueError("No se encontraron columnas objetivo válidas. Se esperaba ['Y1','Y2'] o ['Heating Load','Cooling Load']")

def build_preprocessor(feature_names: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_names)],
        remainder="drop",
    )

def models_zoo(seed: int = DEFAULT_SEED) -> List[ModelSpec]:
    return [
        ModelSpec(
            name="Linear Regression",
            estimator=LinearRegression(),
            params={"model": "LinearRegression"},
        ),
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestRegressor(n_estimators=600, random_state=seed, n_jobs=-1),
            params={"model": "RandomForestRegressor", "n_estimators": 600, "n_jobs": -1, "random_state": seed},
        ),
        ModelSpec(
            name="Gradient Boosting",
            estimator=GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=seed),
            params={"model": "GradientBoostingRegressor", "learning_rate": 0.1, "max_depth": 3, "n_estimators": 100, "random_state": seed},
        ),
    ]

def _ensure_2d(array) -> np.ndarray:
    arr = np.asarray(array)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr

def r2_metric(y_true, y_pred) -> float:
    return r2_score(_ensure_2d(y_true), _ensure_2d(y_pred), multioutput="uniform_average")

def mae_metric(y_true, y_pred) -> float:
    return mean_absolute_error(_ensure_2d(y_true), _ensure_2d(y_pred), multioutput="uniform_average")

def rmse_metric(y_true, y_pred) -> float:
    return mean_squared_error(_ensure_2d(y_true), _ensure_2d(y_pred), squared=False)

class FailsafeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            arr = X.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float, copy=False)
        else:
            arr = np.asarray(X, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

def build_pipeline(preprocessor: ColumnTransformer, base_estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("failsafe", FailsafeTransformer()),
        ("regressor", MultiOutputRegressor(base_estimator)),
    ])

def evaluate_model(model_name, pipeline, X_train, X_test, y_train, y_test, cv_splits):
    scoring = {
        "r2": make_scorer(r2_metric),
        "neg_mae": make_scorer(mae_metric, greater_is_better=False),
        "neg_rmse": make_scorer(rmse_metric, greater_is_better=False),
    }
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=DEFAULT_SEED)
    cv_results = cross_validate(clone(pipeline), X_train, y_train, cv=cv, scoring=scoring, error_score="raise")

    fitted_pipeline = clone(pipeline)
    fitted_pipeline.fit(X_train, y_train)
    preds = fitted_pipeline.predict(X_test)
    return (
        fitted_pipeline,
        {
            "R2": r2_metric(y_test, preds),
            "MAE": mae_metric(y_test, preds),
            "RMSE": rmse_metric(y_test, preds),
        },
        {
            "cv_R2_mean": np.mean(cv_results["test_r2"]),
            "cv_MAE_mean": -np.mean(cv_results["test_neg_mae"]),
            "cv_RMSE_mean": -np.mean(cv_results["test_neg_rmse"]),
        },
    )

def write_dashboard(metrics_df, csv_path, html_path):
    metrics_df_rounded = metrics_df.round({"R2": 3, "MAE": 3, "RMSE": 3})
    metrics_df_rounded.to_csv(csv_path, index=False)
    html_content = f"""
    <html><head><meta charset='utf-8'><title>Resultados</title><style>
    body {{ font-family: sans-serif; background: #f4f4f4; padding: 2rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
    th {{ background: #0b3d91; color: #fff; }}
    </style></head><body>
    <h1>Resultados de Experimentos – Energy Efficiency</h1>
    {metrics_df_rounded.to_html(index=False)}
    </body></html>
    """
    html_path.write_text(html_content, encoding="utf-8")

def log_mlflow_run(model_name, pipeline, holdout_metrics, cv_metrics, params, feature_names, target_choice, test_size, seed, csv_path, html_path, X_sample):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment else ""
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        mlflow.log_params({**{
            "seed": seed,
            "test_size": test_size,
            "target_choice": ",".join(target_choice),
            "n_features": len(feature_names),
            "feature_names": ",".join(feature_names),
        }, **params})
        mlflow.log_metrics({**holdout_metrics, **cv_metrics})
        mlflow.set_tags({"author": "Ricardo Aguilar", "dataset": "Energy Efficiency"})
        mlflow.log_artifact(str(csv_path), artifact_path="outputs")
        mlflow.log_artifact(str(html_path), artifact_path="outputs")
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=X_sample.head(2).fillna(0),
            registered_model_name=None,
        )
    return run_id, experiment_id

def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["heating", "cooling", "both"], default="both")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE)
    return parser.parse_args(args)

def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or [])
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    csv_path = base_dir / "results_metrics.csv"
    html_path = base_dir / "index.html"
    dataset_paths = [
        project_root / "data" / "energy_efficiency_modified.csv",
        project_root / "data" / "processed" / "energy_efficiency_modified.csv",
        project_root / "data" / "raw" / "energy_efficiency_modified.csv",
    ]
    
    df = load_data(dataset_paths)

    # 🔒 Eliminar filas con cualquier valor NaN o infinito antes del procesamiento
    df = df.apply(pd.to_numeric, errors='coerce')           # Forzar todos los datos a ser numéricos
    df = df.replace([np.inf, -np.inf], np.nan)              # Reemplazar infinitos por NaN
    df = df.dropna()                                        # Eliminar filas que contengan cualquier NaN
    if df.empty:
        raise ValueError("El dataset quedó vacío tras eliminar NaNs. Verifica el archivo de entrada.")

    # Continuar normalmente
    target_candidates = detect_targets(df)


    if args.target == "heating":
        selected_targets = [target_candidates[0]]
    elif args.target == "cooling":
        selected_targets = [target_candidates[1]]
    else:
        selected_targets = target_candidates

    X = df.drop(columns=selected_targets)
    y = df[selected_targets]

    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = y.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    Xy = pd.concat([X, y], axis=1).dropna()
    X = Xy[X.columns]
    y = Xy[y.columns]

    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=DEFAULT_SEED)

    preprocessor = build_preprocessor(feature_names)
    model_specs = models_zoo()
    results = []

    for spec in model_specs:
        pipeline = build_pipeline(preprocessor, spec.estimator)
        fitted, holdout, cv = evaluate_model(spec.name, pipeline, X_train, X_test, y_train, y_test, 5)
        results.append({
            "name": spec.name,
            "pipeline": fitted,
            "holdout_metrics": holdout,
            "cv_metrics": cv,
            "params": spec.params,
        })

    metrics_df = pd.DataFrame([{
        "Model": r["name"],
        "R2": r["holdout_metrics"]["R2"],
        "MAE": r["holdout_metrics"]["MAE"],
        "RMSE": r["holdout_metrics"]["RMSE"],
    } for r in results])

    write_dashboard(metrics_df, csv_path, html_path)

    print("\nResultados de holdout:")
    print(metrics_df.round(3).to_string(index=False))

    for record in results:
        run_id, exp_id = log_mlflow_run(
            record["name"], record["pipeline"],
            record["holdout_metrics"], record["cv_metrics"],
            record["params"], feature_names,
            selected_targets, args.test_size, DEFAULT_SEED,
            csv_path, html_path, X
        )
        print(f"Run ID: {run_id} | Experiment ID: {exp_id}")

    print(f"CSV:  {csv_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")
    print("MLflow UI: mlflow ui --port 5000")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
