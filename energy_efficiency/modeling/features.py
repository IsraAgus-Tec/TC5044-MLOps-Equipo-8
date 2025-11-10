"""
Transformaciones de features – Equipo 8.
Define el preprocesador para pipelines de scikit-learn.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
