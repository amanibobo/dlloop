"""ML tier. ``lenscraft.ml.train`` imports torch and is only imported inside the Modal GPU image
or a dev environment; ``lenscraft.ml.uncertainty`` is pure numpy and runs anywhere."""

__all__ = ["evaluate_model", "sample_uncertainty", "train_model"]


def __getattr__(name: str):
    if name in ("train_model", "evaluate_model"):
        from lenscraft.ml import train

        return getattr(train, name)
    if name == "sample_uncertainty":
        from lenscraft.ml.uncertainty import sample_uncertainty

        return sample_uncertainty
    raise AttributeError(name)
