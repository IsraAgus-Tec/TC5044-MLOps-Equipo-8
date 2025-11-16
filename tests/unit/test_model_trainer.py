# src/tests/test_model_trainer.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from handlers.data_loader import DataLoader
from handlers.data_preprocessor import DataPreprocessor
from handlers.model_trainer import ModelTrainer


def _get_cleansed_df():
    loader = DataLoader()
    df = loader.getDataFrameFromFile("data/interim/energy_efficiency_modified.csv")
    preprocessor = DataPreprocessor(df)
    preprocessor.convert_numeric()
    preprocessor.impute_missing()
    preprocessor.detect_outliers()
    return preprocessor.df


def test_model_trainer_split_and_train():
    df = _get_cleansed_df()
    trainer = ModelTrainer(df)

    trainer.split_data()
    trainer.train_models()

    # Verificar que las particiones no estén vacías
    assert trainer.X_train is not None
    assert trainer.X_test is not None
    assert trainer.Y_train is not None
    assert trainer.Y_test is not None
    assert len(trainer.X_train) > 0
    assert len(trainer.X_test) > 0

    # Debe haber modelos entrenados (dict no vacío)
    assert hasattr(trainer, "models")
    assert isinstance(trainer.models, dict)
    assert len(trainer.models) >= 1
