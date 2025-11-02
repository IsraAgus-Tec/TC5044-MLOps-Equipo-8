import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    def __init__(self, models, X_test, Y_test, validation_reports=None):
        """
        Evaluate trained pipelines on a held-out test set.

        Keyword arguments:
        models -- dictionary with model name as key and trained Pipeline as value
        X_test -- features reserved for testing (unseen during training)
        Y_test -- target values corresponding to X_test
        validation_reports -- optional dictionary with cross-validation summaries
        """
        self.models = models
        self.X_test = X_test
        self.Y_test = Y_test
        self.validation_reports = validation_reports or {}

    def list_models(self):
        """
        Print the names of all trained models available for evaluation.
        """
        print("\nTrained models:")
        for name in self.models.keys():
            print(f"- {name}")

    def evaluate_model(self, y_true, y_pred, name, target):
        """
        Calculate and display regression metrics for a single model.

        Keyword arguments:
        y_true -- actual target values
        y_pred -- predicted target values
        name -- name of the model being evaluated
        target -- target column name being evaluated

        Returns tuple of (R², RMSE, MAE) metrics.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(
            f"{name:<20} | Target: {target:<15} | "
            f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}"
        )
        return r2, rmse, mae

    def evaluate_all(self):
        """
        Evaluate all trained models and return results as a DataFrame.

        Calculates R², RMSE, and MAE for each model and returns
        a pandas DataFrame with all evaluation metrics.
        """
        self.list_models()
        print("\nModel evaluation:\n")
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_df = pd.DataFrame(
                y_pred, columns=self.Y_test.columns, index=self.Y_test.index
            )
            for target in self.Y_test.columns:
                y_true_col = self.Y_test[target]
                y_pred_col = y_pred_df[target]
                r2, rmse, mae = self.evaluate_model(
                    y_true_col, y_pred_col, name, target
                )
                results.append([name, target, r2, rmse, mae])
            if name in self.validation_reports:
                val = self.validation_reports[name]
                print(
                    f"  -> CV mean scores | R²: {val['r2']:.4f} | "
                    f"RMSE: {val['rmse']:.4f} | MAE: {val['mae']:.4f}"
                )
        return pd.DataFrame(
            results, columns=["Model", "Target", "R2", "RMSE", "MAE"]
        )
