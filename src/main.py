from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd
from joblib import dump
import mlflow
import mlflow.sklearn

try:
    from handlers.model_evaluator import ModelEvaluator
    from handlers.model_trainer import ModelTrainer
    from handlers.visual_eda import VisualEDA
    from handlers.data_loader import DataLoader
    from handlers.data_preprocessor import DataPreprocessor
except ModuleNotFoundError:  # Ejecutado como paquete: python -m src.main
    from .handlers.model_evaluator import ModelEvaluator
    from .handlers.model_trainer import ModelTrainer
    from .handlers.visual_eda import VisualEDA
    from .handlers.data_loader import DataLoader
    from .handlers.data_preprocessor import DataPreprocessor

DEFAULT_DATASET_PATH = "data/interim/energy_efficiency_modified.csv"
DEFAULT_CLEANSED_DIR = "data/interim/cleansed"
DEFAULT_RESULTS_DIR = Path("notebooks")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_MODEL_PATH = Path("models/energy_efficiency_rf.joblib")
DEFAULT_MODEL_METADATA = DEFAULT_MODEL_PATH.with_suffix(".json")
CLEANSED_FILENAME = "energy_efficiency_modified.csv"
MLFLOW_TRACKING_URI = f"file:{(Path('mlruns').resolve())}"
MLFLOW_EXPERIMENT = "EnergyEfficiency-CDS"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


def _save_metrics(results_df, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results_metrics.csv"
    html_path = output_dir / "results_metrics.html"
    results_df.to_csv(csv_path, index=False)
    results_df.to_html(html_path, index=False)
    print(f"\n\n > Metrics exported to:\n   - {csv_path}\n   - {html_path}\n")


def _generate_visuals(eda: VisualEDA, stage_name: str, figures_dir: Path, show_visuals: bool):
    stage_dir = figures_dir / stage_name
    print(f"\n\n > Exporting visual EDA ({stage_name.replace('_', ' ')}):\n")
    eda.plot_histograms(stage_dir / "histograms.png", show=show_visuals)
    eda.plot_boxplots(stage_dir / "boxplots.png", show=show_visuals)
    eda.plot_correlation_heatmap(stage_dir / "correlation_heatmap.png", show=show_visuals)


def _select_best_model(models: dict, validation_reports: dict) -> tuple[str, object]:
    if not models:
        raise ValueError("No models were trained; cannot export artifact.")
    if not validation_reports:
        name, estimator = next(iter(models.items()))
        return name, estimator
    best_name = max(validation_reports.items(), key=lambda item: item[1].get("r2", float("-inf")))[0]
    return best_name, models[best_name]


def _log_model_to_mlflow(model_name: str, model, model_path: Path, metadata_path: Path, metrics: dict | None):
    with mlflow.start_run(run_name=f"{model_name}-export", nested=False):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("artifact_path", str(model_path))
        if metrics:
            mlflow.log_metrics({f"cv_{k}": float(v) for k, v in metrics.items()})
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(model_path), artifact_path="serialized")
        mlflow.log_artifact(str(metadata_path), artifact_path="serialized")


def _export_model(model, model_name: str, feature_cols, model_path: Path, metadata_path: Path, metrics: dict | None):
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    metadata = {
        "model_name": model_name,
        "artifact_path": model_path.as_posix(),
        "feature_columns": list(feature_cols),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n\n > Model '{model_name}' exported to {model_path}\n")
    _log_model_to_mlflow(model_name, model, model_path, metadata_path, metrics)


def main(
    showVisualEDA: bool = False,
    dataset_path: str = DEFAULT_DATASET_PATH,
    cleansed_dir: str = DEFAULT_CLEANSED_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    export_metrics: bool = True,
    generate_figures: bool = True,
    export_model: bool = True,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    model_metadata_path: str | Path = DEFAULT_MODEL_METADATA,
) -> pd.DataFrame:
    """
    Execute the complete machine learning pipeline for energy efficiency analysis.
    
    Keyword arguments:
    showVisualEDA -- whether to display visual exploratory data analysis plots (default False)
    """
    data_loader = DataLoader()

    df = data_loader.getDataFrameFromFile(dataset_path)

    print(f"\n\n > Lodaded data - Rows: {df.shape[0]}, Columns: {df.shape[1]}", "\n\n")

    eda = VisualEDA(df)
    data_preprocessor = DataPreprocessor(df)

    data_preprocessor.convert_numeric()

    print(f"\n\n > Data overview before cleansing...", "\n\n")

    eda.overview()

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)

    if generate_figures:
        _generate_visuals(eda, "before_cleansing", figures_dir, showVisualEDA)

    print(f"\n\n > Initializing data cleansing...", "\n\n")

    data_preprocessor.impute_missing()
    data_preprocessor.detect_outliers()

    print(f"\n\n > Data overview after cleansing...", "\n\n")

    eda.overview()

    if generate_figures:
        _generate_visuals(eda, "after_cleansing", figures_dir, showVisualEDA)

    data_loader.saveDataFrameAsFileWithDVC(
        data_preprocessor.df,
        str(cleansed_dir),
        CLEANSED_FILENAME,
    )

    print(f"\n\n > Initializing model training...", "\n\n")

    trainer = ModelTrainer(data_preprocessor.df)
    trainer.split_data()
    trainer.train_models()

    print(f"\n\n > Initializing model evaluation...", "\n\n")

    evaluator = ModelEvaluator(
        trainer.models, trainer.X_test, trainer.Y_test, trainer.validation_reports
    )
    results_df = evaluator.evaluate_all()

    if export_metrics:
        _save_metrics(results_df, results_dir)

    if export_model:
        best_name, best_model = _select_best_model(trainer.models, trainer.validation_reports)
        _export_model(
            best_model,
            best_name,
            trainer.feature_cols,
            model_path,
            model_metadata_path,
            trainer.validation_reports.get(best_name),
        )

    return results_df


def _parse_args():
    parser = argparse.ArgumentParser(description="Energy efficiency training pipeline")
    parser.add_argument(
        "--dataset",
        dest="dataset_path",
        default=DEFAULT_DATASET_PATH,
        help="Ruta al dataset base (por defecto data/interim/energy_efficiency_modified.csv)",
    )
    parser.add_argument(
        "--cleansed-dir",
        dest="cleansed_dir",
        default=DEFAULT_CLEANSED_DIR,
        help="Directorio donde se versiona el dataset limpio (por defecto data/interim/cleansed)",
    )
    parser.add_argument(
        "--output",
        dest="results_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directorio donde se guardan los resultados y métricas (por defecto notebooks/)",
    )
    parser.add_argument(
        "--figures-dir",
        dest="figures_dir",
        default=str(DEFAULT_FIGURES_DIR),
        help="Directorio donde se guardan las imágenes de EDA (por defecto reports/figures/)",
    )
    parser.add_argument(
        "--show-eda",
        action="store_true",
        help="Habilita las gráficas de EDA antes y después del preprocesamiento",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Omite la exportación de results_metrics.{csv,html}",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Omitir la generación de imágenes de EDA (se guardan por defecto)",
    )
    parser.add_argument(
        "--skip-model-export",
        action="store_true",
        help="No serializar el modelo entrenado en la carpeta models/",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta del artefacto serializado (joblib). Por defecto models/energy_efficiency_rf.joblib",
    )
    parser.add_argument(
        "--model-metadata",
        dest="model_metadata",
        default=str(DEFAULT_MODEL_METADATA),
        help="Ruta del archivo JSON con metadatos del modelo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        showVisualEDA=args.show_eda,
        dataset_path=args.dataset_path,
        cleansed_dir=args.cleansed_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        export_metrics=not args.skip_metrics,
        generate_figures=not args.skip_figures,
        export_model=not args.skip_model_export,
        model_path=args.model_path,
        model_metadata_path=args.model_metadata,
    )
