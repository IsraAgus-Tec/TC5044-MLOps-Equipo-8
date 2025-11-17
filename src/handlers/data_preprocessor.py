import numpy as np
import pandas as pd


class DataPreprocessor:
    NUMERIC_COLS = [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
        "overall_height",
        "orientation",
        "glazing_area",
        "glazing_area_distribution",
        "heating_load",
        "cooling_load",
        "mixed_type_col",
    ]

    def __init__(self, df):
        self.df = df
        self.outliers = []

    def convert_numeric(self):
        """
        Convert specified columns to numeric float type.
        
        Converts all columns in NUMERIC_COLS to float, handling errors by coercing to NaN.
        """
        print(f"\nConverting numeric values...", "\n")
        for col in self.NUMERIC_COLS:
            if col not in self.df.columns:
                continue
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(float)

    def impute_missing(self):
        """
        Fill missing values in numeric columns with their median values.
        
        Prints the number of missing values before and after imputation.
        """
        print(f"\nInitializing imputation of values...", "\n")

        missing_before = self.df.isna().sum().sum()
        print(f"\nMissing values before imputation: {missing_before}", "\n")

        for col in self.NUMERIC_COLS:
            if col not in self.df.columns:
                continue
            self.df[col] = self.df[col].fillna(self.df[col].median())
        missing_after = self.df.isna().sum().sum()

        print(f"\nMissing values after imputation: {missing_after}", "\n")

    def detect_outliers(self):
        """
        Detect and remove outliers using the Interquartile Range (IQR) method.
        
        Identifies outliers as values beyond 1.5 * IQR from Q1 and Q3.
        Removes entire rows containing outliers and stores their indices.
        """
        print(f"\nInitializing outlier analysis...", "\n")
        outlier_rows = set()
        for col in self.NUMERIC_COLS:
            if col not in self.df.columns:
                continue
            column_data = self.df[col]
            valid_data = column_data.dropna().values
            if valid_data.size == 0:
                continue
            Q1 = np.percentile(valid_data, 25)
            Q3 = np.percentile(valid_data, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = column_data[(column_data < lower) | (column_data > upper)].index
            outlier_rows.update(outliers)
        self.outliers = list(sorted(outlier_rows))
        print(f"\nRows detected as outliers: {len(self.outliers)}", "\n")
        self.df.drop(index=self.outliers, inplace=True)
        print(f"\nOutliers removed", "\n")
