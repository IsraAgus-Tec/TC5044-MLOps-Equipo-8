from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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
CLEANSED_FILENAME = "energy_efficiency_modified.csv"


def _save_metrics(results_df, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results_metrics.csv"
    html_path = output_dir / "results_metrics.html"
    results_df.to_csv(csv_path, index=False)
    results_df.to_html(html_path, index=False)
    print(f"\n\n > Metrics exported to:\n   - {csv_path}\n   - {html_path}\n")


def main(
    showVisualEDA: bool = False,
    dataset_path: str = DEFAULT_DATASET_PATH,
    cleansed_dir: str = DEFAULT_CLEANSED_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    export_metrics: bool = True,
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

    if showVisualEDA:
        print(f"\n\n > Initializing visual EDA before processing...", "\n\n")

        eda.plot_histograms()
        eda.plot_boxplots()
        eda.plot_correlation_heatmap()

    print(f"\n\n > Initializing data cleansing...", "\n\n")

    data_preprocessor.impute_missing()
    data_preprocessor.detect_outliers()

    print(f"\n\n > Data overview after cleansing...", "\n\n")

    eda.overview()

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

    if showVisualEDA:
        print(f"\n\n > Initializing visual EDA after processing...", "\n\n")

        eda.plot_histograms()
        eda.plot_boxplots()
        eda.plot_correlation_heatmap()

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
        "--show-eda",
        action="store_true",
        help="Habilita las gráficas de EDA antes y después del preprocesamiento",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Omite la exportación de results_metrics.{csv,html}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        showVisualEDA=args.show_eda,
        dataset_path=args.dataset_path,
        cleansed_dir=args.cleansed_dir,
        results_dir=args.results_dir,
        export_metrics=not args.skip_metrics,
    )
