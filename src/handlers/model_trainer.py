import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class ModelTrainer:
    def __init__(
        self,
        df,
        target_cols=["heating_load", "cooling_load"],
        test_size=0.2,
        random_state=42,
    ):
        self.df = df.copy()
        self.target_cols = target_cols
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = self.X_test = self.Y_train = self.Y_test = None
        self.feature_cols = [col for col in self.df.columns if col not in self.target_cols]
        self.models = {}
        self.validation_reports = {}

    def list_models(self):
        """
        Print the names of all currently available models for training.
        """
        print("\nCurrent models to train:")
        for name in self.models.keys():
            print(f"- {name}")

    def split_data(self):
        """
        Split the dataset into training and testing sets.
        """
        X = self.df[self.feature_cols]
        Y = self.df[self.target_cols]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state
        )
        print(
            f"Dataset split completed -> train: {self.X_train.shape[0]} rows | "
            f"test: {self.X_test.shape[0]} rows"
        )

    def _build_preprocessor(self):
        """
        Create the preprocessing pipeline applied to every estimator.

        Steps:
        - Cast incoming feature columns to float (robust to mixed types).
        - Impute missing values using the median to avoid leakage from outliers.
        - Standardize features to zero mean and unit variance.
        """
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "cast_to_float",
                    FunctionTransformer(
                        lambda data: data.apply(pd.to_numeric, errors="coerce"),
                        validate=False,
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            transformers=[("numeric", numeric_pipeline, self.feature_cols)],
            remainder="drop",
        )

    def _wrap_estimator(self, estimator):
        """
        Ensure the estimator can handle multi-output regression problems.
        """
        # Some regressors (e.g., GradientBoostingRegressor) need explicit multi-output support.
        if getattr(estimator, "_get_tags", None) and estimator._get_tags().get("multioutput", False):
            return estimator
        return MultiOutputRegressor(estimator)

    def train_models(self):
        """
        Train a set of scikit-learn Pipelines that encapsulate preprocessing and modeling.

        For each estimator, the resulting pipeline performs:
        1. Automated preprocessing (casting, imputing, scaling).
        2. Model fitting on the training split.
        3. Cross-validation using r2, RMSE, and MAE to provide quick feedback.
        """
        if any(
            split is None
            for split in [self.X_train, self.X_test, self.Y_train, self.Y_test]
        ):
            raise RuntimeError(
                "split_data must be executed before calling train_models."
            )

        base_estimators = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=600,
                max_depth=12,
                min_samples_split=4,
                random_state=self.random_state,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=4,
                random_state=self.random_state,
            ),
        }

        preprocessor = self._build_preprocessor()
        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }

        for name, estimator in base_estimators.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", self._wrap_estimator(estimator)),
                ]
            )
            pipeline.fit(self.X_train, self.Y_train)
            self.models[name] = pipeline

            cv = cross_validate(
                pipeline,
                self.X_train,
                self.Y_train,
                cv=5,
                scoring=scoring,
            )
            metrics_summary = {}
            for metric in scoring.keys():
                metric_scores = cv[f"test_{metric}"]
                score_mean = metric_scores.mean()
                if metric in {"rmse", "mae"}:
                    metrics_summary[metric] = abs(score_mean)
                else:
                    metrics_summary[metric] = score_mean
            self.validation_reports[name] = metrics_summary

        print("Pipelines trained and validated successfully.")
