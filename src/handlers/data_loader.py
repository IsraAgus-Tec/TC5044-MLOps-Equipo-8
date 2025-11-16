import pandas as pd
from pathlib import Path
import subprocess
import shutil


class DataLoader:
    NAMING_MAP = {
        "X1": "relative_compactness",
        "X2": "surface_area",
        "X3": "wall_area",
        "X4": "roof_area",
        "X5": "overall_height",
        "X6": "orientation",
        "X7": "glazing_area",
        "X8": "glazing_area_distribution",
        "Y1": "heating_load",
        "Y2": "cooling_load",
    }

    def getBaseDir(self):
        return Path(__file__).resolve().parent

    def get_repo_root(self) -> Path:
        # handlers/ -> src/ -> repo root
        return self.getBaseDir().parents[1]

    def getDataFrameFromFile(self, route_to_file: str):
        file_path = self.getBaseDir().parent / route_to_file
        df = pd.read_csv(file_path)
        print(f"\nSuccesfully loaded DF from {file_path}...", "\n")
        return df.rename(columns=self.NAMING_MAP, inplace=False)

    def saveDataFrameAsFileWithDVC(self, df: pd.DataFrame, route: str, file_name: str):
        folder_path = self.getBaseDir().parent / route
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / file_name

        df.to_csv(file_path, index=False)
        print(f"\nSuccesfully saved DF in {file_path}...", "\n")

        print("\nInitializing DVC versioning...", "\n")

        repo_root = self.get_repo_root()

        # Solo intentamos usar DVC si está instalado y el repo tiene .dvc
        if shutil.which("dvc") and (repo_root / ".dvc").exists():
            try:
                subprocess.run(
                    ["dvc", "add", str(file_path.relative_to(repo_root))],
                    cwd=repo_root,
                    check=True,
                )
                print("File successfully tracked with DVC.\n")
            except subprocess.CalledProcessError as e:
                print(f"Warning: DVC tracking failed: {e}\n")
        else:
            print("DVC not available or repo not initialized, skipping DVC tracking.\n")