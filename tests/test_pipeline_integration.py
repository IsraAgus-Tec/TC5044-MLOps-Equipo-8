# tests/test_pipeline_integration.py
from pathlib import Path
import subprocess

# Rutas usadas en tu pipeline (según README_Fase2 y runner.py)
DATA_PATH = Path("data/energy_efficiency_final.csv")
RESULTS_DIR = Path("src/notebooks")
CSV_METRICS = RESULTS_DIR / "results_metrics.csv"
HTML_METRICS = RESULTS_DIR / "results_metrics.html"


def test_end_to_end_pipeline_creates_metrics_files():
    """
    Prueba de integración:
    - Ejecuta el pipeline completo mediante `python -m energy_efficiency.runner`.
    - Verifica que se generen los archivos de métricas esperados.
    """

    # Asegurarnos de que el dataset exista antes de correr el pipeline
    assert DATA_PATH.exists(), f"No se encontró el archivo de datos: {DATA_PATH}"

    # Si ya existen los archivos de métricas, los eliminamos para empezar "limpio"
    if CSV_METRICS.exists():
        CSV_METRICS.unlink()
    if HTML_METRICS.exists():
        HTML_METRICS.unlink()

    # Ejecutar el módulo como script, igual que tú lo corres
    result = subprocess.run(
        [
            "python",
            "-m",
            "energy_efficiency.runner",
            "--target",
            "both",
            "--data",
            str(DATA_PATH),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Si falla, mostramos stdout/stderr para entender por qué
    assert (
        result.returncode == 0
    ), f"El pipeline falló.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Verificar que los archivos de métricas se hayan generado
    assert CSV_METRICS.exists(), f"No se generó el archivo: {CSV_METRICS}"
    assert HTML_METRICS.exists(), f"No se generó el archivo: {HTML_METRICS}"
