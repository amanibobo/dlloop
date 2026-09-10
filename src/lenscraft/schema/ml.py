"""Schemas for the ML tier (DESIGN.md Sections 6-7): training and evaluation specs and results.

Like ``LensCard``, a ``TrainSpec`` is what the agent proposes and a human approves before GPU
time is spent.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelKind = Literal["classifier", "anomaly"]
ClassifierArch = Literal["resnet18", "alexnet"]
AnomalyArch = Literal["dcae", "vae", "aae"]
CLASSIFIER_ARCHS: tuple[str, ...] = ("resnet18", "alexnet")
ANOMALY_ARCHS: tuple[str, ...] = ("dcae", "vae", "aae")

CLASS_NAMES: tuple[str, ...] = ("none", "subhalo", "vortex")
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def eval_scores_filename(run_ids: list[str]) -> str:
    """Name of the per-image scores file ``evaluate_model`` writes for a set of runs."""
    return f"eval_{'-'.join(run_ids)[:60]}.json"


class TrainSpec(BaseModel):
    """One training job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,39}$", description="Unique name for the trained model.")
    kind: ModelKind = Field(description="'classifier' (3-class, Alexander et al. 2019) or 'anomaly' (Alexander et al. 2021).")
    arch: str = Field(description="resnet18 | alexnet for classifiers; dcae | vae | aae for anomaly detectors.")
    run_ids: list[str] = Field(min_length=1, description="Runs whose images form the dataset.")
    epochs: int = Field(default=10, ge=1, le=500)
    batch_size: int = Field(default=64, ge=1, le=1024)
    learning_rate: float = Field(default=1e-3, gt=0)
    val_fraction: float = Field(default=0.2, gt=0, lt=1)
    input_size: int = Field(default=150, ge=96, le=512, description="Images are resized to this; the 2021 autoencoders were specified at 150.")
    latent_dim: int = Field(default=1000, ge=8, description="Anomaly-detector latent size (paper: 1000).")
    kl_weight: float = Field(default=1e-3, ge=0, description="VAE KL term weight.")
    adv_weight: float = Field(default=0.1, ge=0, description="AAE adversarial (generator) loss weight.")
    seed: int = Field(default=0, ge=0)
    max_images: Optional[int] = Field(default=None, ge=1, description="Subsample the dataset (quick tests).")

    @model_validator(mode="after")
    def _check_arch(self) -> "TrainSpec":
        allowed = CLASSIFIER_ARCHS if self.kind == "classifier" else ANOMALY_ARCHS
        if self.arch not in allowed:
            raise ValueError(f"arch {self.arch!r} is not valid for kind {self.kind!r}; expected one of {allowed}")
        return self


class TrainResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "failed"]
    model_id: str
    kind: ModelKind
    arch: str
    run_ids: list[str] = Field(default_factory=list, description="Runs the model was trained on.")
    checkpoint_path: str = ""
    n_train: int = 0
    n_val: int = 0
    epochs_run: int = 0
    metrics: dict[str, float] = Field(default_factory=dict, description="Final validation metrics (val_loss, val_acc, auc, ...).")
    history: list[dict[str, float]] = Field(default_factory=list, description="Per-epoch metrics.")
    elapsed_seconds: float = 0.0
    message: str = ""


class EvalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    run_ids: list[str] = Field(min_length=1)
    max_images: Optional[int] = Field(default=None, ge=1)


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "failed"]
    model_id: str
    kind: ModelKind = "classifier"
    n_images: int = 0
    auc: Optional[float] = Field(default=None, description="Classifier: macro one-vs-rest AUC. Anomaly: substructure-vs-none AUC.")
    metrics: dict[str, float] = Field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict, description="Per substructure class: n, and mean confidence / anomaly score.")
    scores_path: str = Field(default="", description="JSON file with per-image predictions/scores (used by sample_uncertainty).")
    elapsed_seconds: float = 0.0
    message: str = ""
