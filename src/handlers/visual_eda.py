import matplotlib.pyplot as plt
import seaborn as sns


class VisualEDA:
    def __init__(self, df):
        self.df = df

    def overview(self):
        """
        Display comprehensive DataFrame information including structure, statistics, and missing values.
        """
        print(f"\nCurrent dataframe data:", "\n")
        print(f"\n\tDataframe information:", "\n")
        print(self.df.info(), "\n")
        print(f"\n\tDescriptive statistics:", "\n")
        print(self.df.describe(), "\n")
        print(f"\n\tMissing values:", "\n")
        print(self.df.isna().sum(), "\n")

    def plot_histograms(self):
        """
        Generate and display histograms for all numeric columns in the DataFrame.
        """
        print(f"\nPlotting histograms...", "\n")
        numeric_cols = self.df.select_dtypes(include="number").columns
        self.df[numeric_cols].hist(bins=15, figsize=(15, 10))
        plt.suptitle("Histogramas de variables numéricas", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self):
        """
        Create boxplots for all numeric columns to visualize distributions and outliers.
        """
        print(f"\nPlotting boxplots...", "\n")
        numeric_cols = self.df.select_dtypes(include="number").columns
        n_cols = 3
        n_rows = int(len(numeric_cols) / n_cols) + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.boxplot(y=self.df[col], ax=axes[i])
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            plt.suptitle("Boxplots de variables numéricas", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Display a correlation heatmap showing relationships between numeric variables.
        """
        print(f"\nPlotting correlation heatmap...", "\n")
        plt.figure(figsize=(12, 8))
        corr = self.df.corr()
        plt.title("Mapa de correlación", fontsize=16)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.show()
