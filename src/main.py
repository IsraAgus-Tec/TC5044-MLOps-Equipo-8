from handlers.model_evaluator import ModelEvaluator
from handlers.model_trainer import ModelTrainer
from handlers.visual_eda import VisualEDA
from handlers.data_loader import DataLoader
from handlers.data_preprocessor import DataPreprocessor


def main(showVisualEDA: bool):
    """
    Execute the complete machine learning pipeline for energy efficiency analysis.
    
    Keyword arguments:
    showVisualEDA -- whether to display visual exploratory data analysis plots (default False)
    """
    data_loader = DataLoader()

    df = data_loader.getDataFrameFromFile("data/energy_efficiency_modified.csv")

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
        data_preprocessor.df, "data/cleansed", "energy_efficiency_modified.csv"
    )

    print(f"\n\n > Initializing model training...", "\n\n")

    trainer = ModelTrainer(data_preprocessor.df)
    trainer.split_data()
    trainer.train_models()

    print(f"\n\n > Initializing model evaluation...", "\n\n")

    evaluator = ModelEvaluator(
        trainer.models, trainer.X_test, trainer.Y_test, trainer.validation_reports
    )
    evaluator.evaluate_all()

    if showVisualEDA:
        print(f"\n\n > Initializing visual EDA after processing...", "\n\n")

        eda.plot_histograms()
        eda.plot_boxplots()
        eda.plot_correlation_heatmap()

if __name__ == "__main__":
    main(showVisualEDA=True)
