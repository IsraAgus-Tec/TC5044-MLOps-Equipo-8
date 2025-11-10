"""
Evaluación sobre el conjunto de prueba – Equipo 8.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict
from sklearn.pipeline import Pipeline


def evaluate_models(models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula R², RMSE y MAE por modelo (promedio multi-output).
    """
    rows = []
    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
        mae = mean_absolute_error(y_test, y_pred, multioutput="uniform_average")
        rmse = mean_squared_error(y_test, y_pred, multioutput="uniform_average", squared=False)
        rows.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(rows)
