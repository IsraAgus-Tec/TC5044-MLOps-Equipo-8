# energy_efficiency/modeling/predict.py
# Team 8 — Minimal evaluator for holdout performance

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


class Team8Evaluator:
    """
    Evalúa modelos entrenados sobre el conjunto de prueba (holdout).

    Uso:
        evaluator = Team8Evaluator(models, X_test, Y_test, cv_report)
        df_results = evaluator.evaluate_all()
    """

    def __init__(
        self,
        models: Dict[str, Pipeline],
        X_test: pd.DataFrame,
        Y_test: pd.DataFrame,
        cv_report: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.models = models
        self.X_test = X_test
        self.Y_test = Y_test
        self.cv_report = cv_report or {}

    @staticmethod
    def _aggregate_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula R2, RMSE y MAE promediados de forma uniforme entre las salidas.
        """
        r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
        rmse = mean_squared_error(y_true, y_pred, multioutput="uniform_average", squared=False)
        mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
        return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae)}

    def evaluate_all(self, include_cv: bool = True) -> pd.DataFrame:
        """
        Evalúa todos los modelos y devuelve un DataFrame compacto con métricas de holdout.
        Si include_cv=True y existen valores, añade columnas cv_R2, cv_RMSE y cv_MAE.
        """
        rows = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            hold = self._aggregate_metrics(self.Y_test, y_pred)

            row = {"Model": name, **hold}
            if include_cv and name in self.cv_report:
                row.update(
                    {
                        "cv_R2": float(self.cv_report[name].get("r2", np.nan)),
                        "cv_RMSE": float(self.cv_report[name].get("rmse", np.nan)),
                        "cv_MAE": float(self.cv_report[name].get("mae", np.nan)),
                    }
                )
            rows.append(row)

        cols = ["Model", "R2", "MAE", "RMSE", "cv_R2", "cv_MAE", "cv_RMSE"]
        df = pd.DataFrame(rows)
        # orden de columnas si existen
        df = df[[c for c in cols if c in df.columns]].sort_values("R2", ascending=False).reset_index(drop=True)
        return df
