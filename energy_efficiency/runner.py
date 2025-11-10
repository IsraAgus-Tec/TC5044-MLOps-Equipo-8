energy_efficiency/runner.py
New
+509
-0

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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


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
    """Load dataset from preferred paths."""
    for path in dataset_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover - defensive
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
    """Detect target columns based on priority."""
    columns = set(df.columns)
    for candidates in TARGET_PRIORITIES:
        if all(target in columns for target in candidates):
            return list(candidates)
    raise ValueError(
        "No se encontraron columnas objetivo válidas. Se esperaba ['Y1','Y2'] o "
        "['Heating Load','Cooling Load']"
    )


def build_preprocessor(feature_names: List[str]) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_names)],
        remainder="drop",
    )
    return preprocessor


def models_zoo(seed: int = DEFAULT_SEED) -> List[ModelSpec]:
    """Return a list of model specifications."""
    models: List[ModelSpec] = [
        ModelSpec(
            name="Linear Regression",
            estimator=LinearRegression(),
            params={"model": "LinearRegression"},
        ),
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestRegressor(
                n_estimators=600,
                random_state=seed,
                n_jobs=-1,
            ),
            params={
                "model": "RandomForestRegressor",
                "n_estimators": 600,
                "max_depth": None,
                "n_jobs": -1,
                "random_state": seed,
            },
        ),
        ModelSpec(
            name="Gradient Boosting",
            estimator=GradientBoostingRegressor(
                learning_rate=0.1,
                max_depth=3,
                n_estimators=100,
                random_state=seed,
            ),
            params={
                "model": "GradientBoostingRegressor",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 100,
                "random_state": seed,
            },
        ),
    ]
    return models


def _ensure_2d(array) -> np.ndarray:
    """Ensure arrays are two-dimensional for metric compatibility."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def r2_metric(y_true, y_pred) -> float:
    """Compute R2 handling one-dimensional predictions."""
    y_true_2d = _ensure_2d(y_true)
    y_pred_2d = _ensure_2d(y_pred)
    return r2_score(y_true_2d, y_pred_2d, multioutput="uniform_average")


def mae_metric(y_true, y_pred) -> float:
    """Compute MAE handling one-dimensional predictions."""
    y_true_2d = _ensure_2d(y_true)
    y_pred_2d = _ensure_2d(y_pred)
    return mean_absolute_error(y_true_2d, y_pred_2d, multioutput="uniform_average")


def rmse_metric(y_true, y_pred) -> float:
    """Compute root mean squared error handling shape mismatches."""
    y_true_2d = _ensure_2d(y_true)
    y_pred_2d = _ensure_2d(y_pred)
    return mean_squared_error(y_true_2d, y_pred_2d, squared=False)


def _failsafe_replace_nan(X):
    """Replace remaining NaNs with zero after preprocessing."""
    if isinstance(X, pd.DataFrame):
        return X.fillna(0)
    return np.nan_to_num(X, nan=0.0)


def build_pipeline(preprocessor: ColumnTransformer, base_estimator) -> Pipeline:
    """Construct a full pipeline with preprocessing and estimator."""
    preprocessor_clone = clone(preprocessor)
    failsafe = FunctionTransformer(_failsafe_replace_nan, validate=False)
    model = MultiOutputRegressor(base_estimator)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_clone),
            ("failsafe", failsafe),
            ("regressor", model),
        ]
    )
    return pipeline


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    cv_splits: int,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, float]]:
    """Fit pipeline, evaluate holdout and cross-validation metrics."""
    scoring = {
        "r2": make_scorer(r2_metric),
        "neg_mae": make_scorer(mae_metric, greater_is_better=False),
        "neg_rmse": make_scorer(rmse_metric, greater_is_better=False),
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=DEFAULT_SEED)
    cv_results = cross_validate(
        clone(pipeline),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
        error_score="raise",
    )

    fitted_pipeline = clone(pipeline)
    fitted_pipeline.fit(X_train, y_train)
    predictions = fitted_pipeline.predict(X_test)

    y_test_array = _ensure_2d(y_test)
    predictions_array = _ensure_2d(predictions)

    holdout_metrics = {
        "R2": float(r2_metric(y_test_array, predictions_array)),
        "MAE": float(mae_metric(y_test_array, predictions_array)),
        "RMSE": float(rmse_metric(y_test_array, predictions_array)),
    }

    cv_metrics = {
        "cv_R2_mean": float(np.mean(cv_results["test_r2"])),
        "cv_MAE_mean": float(-np.mean(cv_results["test_neg_mae"])),
        "cv_RMSE_mean": float(-np.mean(cv_results["test_neg_rmse"])),
    }

    return fitted_pipeline, holdout_metrics, cv_metrics


def write_dashboard(metrics_df: pd.DataFrame, csv_path: Path, html_path: Path) -> None:
    """Persist metrics to CSV and HTML dashboard."""
    metrics_df_rounded = metrics_df.copy()
    metrics_df_rounded[["R2", "MAE", "RMSE"]] = metrics_df_rounded[["R2", "MAE", "RMSE"]].round(3)
    metrics_df_rounded.to_csv(csv_path, index=False)

    table_html = metrics_df_rounded.to_html(index=False, classes="results-table")
    style = """
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f7fa;
        margin: 0;
        padding: 2rem;
        color: #0b2545;
    }
    h1 {
        color: #0b3d91;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .table-container {
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(11, 61, 145, 0.1);
        padding: 2rem;
    }
    table.results-table {
        width: 100%;
        border-collapse: collapse;
    }
    table.results-table thead {
        background-color: #0b3d91;
        color: #ffffff;
        position: sticky;
        top: 0;
    }
    table.results-table th,
    table.results-table td {
        padding: 12px 16px;
        text-align: center;
    }
    table.results-table tbody tr:nth-child(odd) {
        background-color: #e6ecfa;
    }
    table.results-table tbody tr:nth-child(even) {
        background-color: #ffffff;
    }
    table.results-table tbody tr:hover {
        background-color: #d0dbf7;
    }
    </style>
    """

    html_content = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Resultados de Experimentos – Energy Efficiency</title>
        {style}
    </head>
    <body>
        <div class='table-container'>
            <h1>Resultados de Experimentos – Energy Efficiency</h1>
            {table_html}
        </div>
    </body>
    </html>
    """
    html_path.write_text(html_content, encoding="utf-8")


def log_mlflow_run(
    model_name: str,
    pipeline: Pipeline,
    holdout_metrics: Dict[str, float],
    cv_metrics: Dict[str, float],
    params: Dict[str, object],
    feature_names: List[str],
    target_choice: List[str],
    test_size: float,
    seed: int,
    csv_path: Path,
    html_path: Path,
    X_sample: pd.DataFrame,
) -> Tuple[str, str]:
    """Log run information to MLflow and return run and experiment IDs."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment else ""

    general_params = {
        "seed": seed,
        "test_size": test_size,
        "target_choice": ",".join(target_choice),
        "n_features": len(feature_names),
        "feature_names": ",".join(feature_names),
    }
    combined_params = {**general_params, **params}

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(combined_params)
        mlflow.log_metrics({**holdout_metrics, **cv_metrics})
        mlflow.set_tags(
            {
                "author": "Ricardo Aguilar",
                "dataset": "Energy Efficiency",
                "role": "Data Scientist",
            }
        )
        mlflow.log_artifact(str(csv_path), artifact_path="outputs")
        mlflow.log_artifact(str(html_path), artifact_path="outputs")
        input_example = X_sample.head(2).copy()
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=None,
        )
    return run_id, experiment_id


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Energy Efficiency runner")
    parser.add_argument(
        "--target",
        choices=["heating", "cooling", "both"],
        default="both",
        help="Selecciona el objetivo a modelar",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporción del conjunto de prueba",
    )
    return parser.parse_args(args)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or [])

    if not 0 < args.test_size < 1:
        raise ValueError("--test_size debe estar en el rango (0,1)")

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    csv_path = base_dir / "results_metrics.csv"
    html_path = base_dir / "index.html"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_paths = [
        project_root / "data" / "energy_efficiency_modified.csv",
        project_root / "data" / "processed" / "energy_efficiency_modified.csv",
        project_root / "data" / "raw" / "energy_efficiency_modified.csv",
    ]
    df = load_data(dataset_paths)
    target_candidates = detect_targets(df)

    if args.target == "heating":
        selected_targets = [target_candidates[0]]
    elif args.target == "cooling":
        selected_targets = [target_candidates[1]]
    else:
        selected_targets = target_candidates

    df_targets = df[selected_targets].apply(pd.to_numeric, errors="coerce")
    df_targets = df_targets.replace([np.inf, -np.inf], np.nan)
    valid_rows = df_targets.notna().all(axis=1)
    df_targets = df_targets.loc[valid_rows]

    X = df.drop(columns=selected_targets).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[valid_rows]
    X = X.dropna(axis=1, how="all")

    if X.empty or not X.columns.any():
        raise ValueError("No hay características numéricas disponibles para el modelado")

    feature_names = X.columns.tolist()

    if len(X) == 0:
        raise ValueError("No hay filas disponibles tras limpiar los datos")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df_targets,
        test_size=args.test_size,
        random_state=DEFAULT_SEED,
    )

    if X_train.empty or X_test.empty:
        raise ValueError("El conjunto de entrenamiento o prueba está vacío después del split")

    preprocessor = build_preprocessor(feature_names)

    model_specs = models_zoo(seed=DEFAULT_SEED)
    evaluation_records = []

    for spec in model_specs:
        pipeline = build_pipeline(preprocessor, spec.estimator)
        fitted_pipeline, holdout_metrics, cv_metrics = evaluate_model(
            spec.name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            cv_splits=5,
        )
        evaluation_records.append(
            {
                "name": spec.name,
                "pipeline": fitted_pipeline,
                "holdout_metrics": holdout_metrics,
                "cv_metrics": cv_metrics,
                "params": spec.params,
            }
        )

    metrics_rows = [
        {
            "Model": record["name"],
            "R2": record["holdout_metrics"]["R2"],
            "MAE": record["holdout_metrics"]["MAE"],
            "RMSE": record["holdout_metrics"]["RMSE"],
        }
        for record in evaluation_records
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    write_dashboard(metrics_df, csv_path, html_path)

    print("\nResultados de holdout:")
    print(metrics_df.round({"R2": 3, "MAE": 3, "RMSE": 3}).to_string(index=False))

    for record in evaluation_records:
        run_id, experiment_id = log_mlflow_run(
            model_name=record["name"],
            pipeline=record["pipeline"],
            holdout_metrics=record["holdout_metrics"],
            cv_metrics=record["cv_metrics"],
            params=record["params"],
            feature_names=feature_names,
            target_choice=selected_targets,
            test_size=args.test_size,
            seed=DEFAULT_SEED,
            csv_path=csv_path,
            html_path=html_path,
            X_sample=X,
        )
        print(f"Run ID: {run_id} | Experiment ID: {experiment_id}")

    print(f"CSV:  {csv_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")
    print("MLflow UI: mlflow ui --port 5000")
    print("Abre MLflow UI con: mlflow ui --port 5000")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
