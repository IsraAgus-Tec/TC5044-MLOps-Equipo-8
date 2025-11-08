"""
Subpaquete de modelado (Equipo 8).

Incluye módulos para entrenamiento, evaluación y experimentación.
"""

from .train import ModelTrainer
from .predict import ModelEvaluator

__all__ = ["ModelTrainer", "ModelEvaluator"]
