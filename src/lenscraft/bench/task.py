"""The benchmark task and its prompt for each arm (DESIGN.md Section 9)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class BenchTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "three_class_classifier"
    classes: list[str] = Field(default_factory=lambda: ["none", "subhalo", "vortex"])
    n_train_per_class: int = 24
    n_test_per_class: int = 8
    image_size: int = 32
    mass_fraction: float = 0.03
    arch: str = "resnet18"
    epochs: int = 1
    input_size: int = 96
    auc_threshold: float = 0.5
    train_prefix: str = "train"
    test_prefix: str = "test"
    model_id: str = "clf"

    def train_run(self, cls: str) -> str:
        return f"{self.train_prefix}_{cls}"

    def test_run(self, cls: str) -> str:
        return f"{self.test_prefix}_{cls}"

    def goal(self) -> str:
        cls = ", ".join(self.classes)
        return (
            f"Build a strong-lensing substructure dataset and classifier.\n"
            f"1. Simulate {self.n_train_per_class} training images for each class ({cls}) with run ids "
            f"{', '.join(self.train_run(c) for c in self.classes)}, and {self.n_test_per_class} held-out test images per class with run ids "
            f"{', '.join(self.test_run(c) for c in self.classes)}. Use {self.image_size} px Euclid-like images, a 1e12 M_sun SIE lens at z=0.5 "
            f"with a Sersic source at z=1.0, substructure mass fraction {self.mass_fraction} for the substructure classes, and a distinct seed per run.\n"
            f"2. Train a {self.arch} classifier over the three classes for {self.epochs} epoch(s) on the training runs only, model id '{self.model_id}'.\n"
            f"3. Evaluate it on the test runs (never on training data) and report the macro one-vs-rest AUC.\n"
        )

    def prompt(self, arm: str, workspace: Path | None = None) -> str:
        if arm == "tool":
            return self.goal() + "Use the available tools for every step. Finish by stating the test AUC."
        ws = str(workspace) if workspace else "."
        return (
            self.goal()
            + f"\nYou are in the workspace {ws}. Required output layout (graded automatically):\n"
            f"- {ws}/data/<run_id>/records.lensjsonl: one JSON object per line with keys image_id, substructure_type "
            f"(one of {self.classes}), image_path (relative to the run directory, a float32 .npy file that exists).\n"
            f"- {ws}/data/_models/{self.model_id}/train_result.json: {{\"status\": \"ok\", \"run_ids\": [training run ids]}}.\n"
            f"- {ws}/data/_models/{self.model_id}/eval_test.json: {{\"auc\": <float>, \"run_ids\": [test run ids]}}.\n"
            "Use lenstronomy for the simulation (SIE + SHEAR main lens, POINT_MASS subhalos, SERSIC_ELLIPSE source, "
            "lenstronomy.SimulationAPI with the Euclid config) and torch/torchvision for the classifier. Finish by stating the test AUC."
        )
