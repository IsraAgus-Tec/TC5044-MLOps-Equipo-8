import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from handlers.data_loader import DataLoader
from handlers.data_preprocessor import DataPreprocessor
from handlers.model_trainer import ModelTrainer
from src.config import RANDOM_STATE


def _train_and_get_preds(df, random_state, n_samples=10):
    """
    Ejecuta el pipeline básico y regresa predicciones de un modelo
    para un subconjunto fijo de X_test.
    """
    trainer = ModelTrainer(df, random_state=random_state)
    trainer.split_data()
    trainer.train_models()

    # Usamos el modelo de RandomForest como referencia
    rf_model = trainer.models["RandomForest"]
    X_sample = trainer.X_test.iloc[:n_samples].copy()
    preds = rf_model.predict(X_sample)
    return preds


def test_reproducible_with_same_seed():
    """
    Con la MISMA semilla, el modelo debe producir las mismas predicciones.
    """
    loader = DataLoader()
    df = loader.getDataFrameFromFile("data/interim/energy_efficiency_modified.csv")

    preds_run1 = _train_and_get_preds(df, RANDOM_STATE)
    preds_run2 = _train_and_get_preds(df, RANDOM_STATE)

    assert np.allclose(
        preds_run1, preds_run2
    ), "Predictions differ between runs with the same RANDOM_STATE."


def test_different_seeds_change_predictions():
    """
    Con semillas DISTINTAS, esperamos al menos alguna diferencia en predicciones.
    """
    loader = DataLoader()
    df = loader.getDataFrameFromFile("data/interim/energy_efficiency_modified.csv")

    preds_seed_1 = _train_and_get_preds(df, 42)
    preds_seed_2 = _train_and_get_preds(df, 99)

    # Es posible que sean muy parecidos, pero en general al menos uno debería diferir
    assert not np.allclose(
        preds_seed_1, preds_seed_2
    ), "Predictions are identical even with different seeds, which is unexpected."
