"""ML tier: dataset loading, training and evaluation. Imports torch; only import inside the
Modal GPU image or a dev environment with torch installed."""

from lenscraft.ml.train import evaluate_model, train_model

__all__ = ["evaluate_model", "train_model"]
