"""Training and evaluation for the ported classifiers and anomaly detectors.

Both entry points take the same ``data_root`` / ``models_root`` layout locally and on the Modal
Volume, never raise (failures come back as ``status="failed"``), and write:

    <models_root>/<model_id>/model.pt            checkpoint (spec + state_dict)
    <models_root>/<model_id>/train_result.json
    <models_root>/<model_id>/eval_<tag>.json     per-image scores from evaluate_model
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lenscraft.ml.data import LensImageDataset, load_run_records
from lenscraft.models import build_anomaly_model, build_classifier
from lenscraft.schema.lensjsonl import LensRecord
from lenscraft.schema.ml import CLASS_INDEX, CLASS_NAMES, EvalResult, EvalSpec, TrainResult, TrainSpec, eval_scores_filename

CHECKPOINT = "model.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build(spec: TrainSpec) -> nn.Module:
    if spec.kind == "classifier":
        return build_classifier(spec.arch, n_classes=len(CLASS_NAMES))
    return build_anomaly_model(spec.arch, spec.latent_dim, spec.input_size)


def _loader(records: list[LensRecord], data_root: Path, spec: TrainSpec, shuffle: bool) -> DataLoader:
    return DataLoader(LensImageDataset(records, data_root, spec.input_size), batch_size=spec.batch_size, shuffle=shuffle)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Binary AUC (scores = P(positive)) or macro one-vs-rest AUC (scores shape (N, C)); None if undefined."""
    from sklearn.metrics import roc_auc_score

    present = np.unique(labels)
    if len(present) < 2:
        return None
    if scores.ndim == 1:
        return float(roc_auc_score(labels, scores))
    if len(present) == scores.shape[1]:
        return float(roc_auc_score(labels, scores, multi_class="ovr", average="macro"))
    aucs = []
    for c in present:  # classes missing from the labels have no defined AUC
        aucs.append(roc_auc_score((labels == c).astype(int), scores[:, c]))
    return float(np.mean(aucs))


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def score_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def score_anomaly(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Per-image reconstruction MSE (higher = more anomalous) and labels."""
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        x = x.to(device)
        target = x * 2 - 1  # decoder ends in tanh
        recon = model.reconstruct(x)
        scores.append(((recon - target) ** 2).flatten(1).mean(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------


def _train_classifier_epoch(model, loader, opt, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def _train_anomaly_epoch(model, loader, opts, spec: TrainSpec, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        target = x * 2 - 1
        if spec.arch == "dcae":
            opts["ae"].zero_grad()
            loss = F.mse_loss(model(x), target)
            loss.backward()
            opts["ae"].step()
        elif spec.arch == "vae":
            opts["ae"].zero_grad()
            recon, mu, logvar = model(x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = F.mse_loss(recon, target) + spec.kl_weight * kl
            loss.backward()
            opts["ae"].step()
        else:  # aae: reconstruction, then discriminator, then generator (encoder) step
            opts["ae"].zero_grad()
            recon, z = model(x)
            loss = F.mse_loss(recon, target)
            loss.backward()
            opts["ae"].step()

            opts["d"].zero_grad()
            z_fake = model.encoder(x).detach()
            z_real = torch.randn_like(z_fake)
            d_loss = F.binary_cross_entropy(model.discriminator(z_real), torch.ones(len(x), 1, device=device)) + F.binary_cross_entropy(
                model.discriminator(z_fake), torch.zeros(len(x), 1, device=device)
            )
            d_loss.backward()
            opts["d"].step()

            opts["g"].zero_grad()
            g_loss = spec.adv_weight * F.binary_cross_entropy(model.discriminator(model.encoder(x)), torch.ones(len(x), 1, device=device))
            g_loss.backward()
            opts["g"].step()
        total += loss.item() * len(x)
        n += len(x)
    return total / max(n, 1)


def _split(records: list[LensRecord], spec: TrainSpec) -> tuple[list[LensRecord], list[LensRecord]]:
    rng = np.random.default_rng(spec.seed)
    idx = rng.permutation(len(records))
    if spec.max_images is not None:
        idx = idx[: spec.max_images]
    n_val = max(1, int(round(len(idx) * spec.val_fraction))) if len(idx) > 1 else 0
    val = [records[i] for i in idx[:n_val]]
    train = [records[i] for i in idx[n_val:]]
    return train, val


def train_model(spec: TrainSpec, data_root: str | Path, models_root: str | Path) -> TrainResult:
    t0 = time.perf_counter()
    data_root, models_root = Path(data_root), Path(models_root)
    out_dir = models_root / spec.model_id
    try:
        torch.manual_seed(spec.seed)
        device = get_device()
        records = load_run_records(data_root, spec.run_ids)

        if spec.kind == "anomaly":
            # Unsupervised: train on the 'none' class only; anything with substructure is held out
            # and used purely to measure how well reconstruction error separates it.
            normal = [r for r in records if r.substructure_type == "none"]
            anomalous = [r for r in records if r.substructure_type != "none"]
            if not normal:
                raise ValueError("anomaly detectors train on substructure='none' images, but the given runs contain none")
            train_recs, val_recs = _split(normal, spec)
        else:
            train_recs, val_recs = _split(records, spec)
            anomalous = []
        if not train_recs:
            raise ValueError("no training images after the train/val split")

        model = _build(spec).to(device)
        if spec.kind == "classifier":
            opts = {"ae": torch.optim.Adam(model.parameters(), lr=spec.learning_rate)}
        elif spec.arch == "aae":
            opts = {
                "ae": torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=spec.learning_rate),
                "d": torch.optim.Adam(model.discriminator.parameters(), lr=spec.learning_rate),
                "g": torch.optim.Adam(model.encoder.parameters(), lr=spec.learning_rate),
            }
        else:
            opts = {"ae": torch.optim.Adam(model.parameters(), lr=spec.learning_rate)}

        train_loader = _loader(train_recs, data_root, spec, shuffle=True)
        val_loader = _loader(val_recs, data_root, spec, shuffle=False) if val_recs else None
        anom_loader = _loader(anomalous, data_root, spec, shuffle=False) if anomalous else None
        # Cosine decay to ~0 over the run damps the epoch-to-epoch swings of a constant-LR Adam
        # run from scratch; the checkpoint kept is the best validation epoch, not the last one.
        schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=spec.epochs) for o in opts.values()]

        history: list[dict[str, float]] = []
        metrics: dict[str, float] = {}
        best_key, best_value, best_state, best_epoch = _selection_key(spec), None, None, 0
        for epoch in range(spec.epochs):
            if spec.kind == "classifier":
                train_loss = _train_classifier_epoch(model, train_loader, opts["ae"], device)
            else:
                train_loss = _train_anomaly_epoch(model, train_loader, opts, spec, device)
            for s in schedulers:
                s.step()
            row = {"epoch": float(epoch + 1), "train_loss": train_loss}
            if val_loader is not None:
                row.update(_validate(model, spec, val_loader, anom_loader, device))
            history.append(row)
            value = row.get(best_key[0])
            if value is None:
                value = -train_loss  # no validation split: fall back to training loss
            elif best_key[1] == "min":
                value = -value
            if best_value is None or value > best_value:
                best_value, best_epoch = value, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                metrics = {k: v for k, v in row.items() if k != "epoch"}
        metrics["best_epoch"] = float(best_epoch)

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"spec": spec.model_dump(), "state_dict": best_state, "class_names": list(CLASS_NAMES)}, out_dir / CHECKPOINT)
        result = TrainResult(
            status="ok", model_id=spec.model_id, kind=spec.kind, arch=spec.arch, run_ids=list(spec.run_ids), checkpoint_path=str(out_dir / CHECKPOINT),
            n_train=len(train_recs), n_val=len(val_recs), epochs_run=spec.epochs, metrics=metrics, history=history,
            elapsed_seconds=time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001 - structured failure for the agent
        result = TrainResult(
            status="failed", model_id=spec.model_id, kind=spec.kind, arch=spec.arch, run_ids=list(spec.run_ids),
            elapsed_seconds=time.perf_counter() - t0, message=f"{type(exc).__name__}: {exc}",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _selection_key(spec: TrainSpec) -> tuple[str, str]:
    """(metric, direction) used to pick the checkpoint epoch."""
    if spec.kind == "classifier":
        return "val_acc", "max"
    return "val_loss", "min"


def _validate(model, spec: TrainSpec, val_loader, anom_loader, device) -> dict[str, float]:
    out: dict[str, float] = {}
    if spec.kind == "classifier":
        probs, labels = score_classifier(model, val_loader, device)
        out["val_loss"] = float(F.nll_loss(torch.log(torch.from_numpy(probs) + 1e-12), torch.from_numpy(labels)).item())
        out["val_acc"] = float((probs.argmax(1) == labels).mean())
        auc = _auc(labels, probs)
        if auc is not None:
            out["auc"] = auc
    else:
        scores, _ = score_anomaly(model, val_loader, device)
        out["val_loss"] = float(scores.mean())
        if anom_loader is not None:
            a_scores, _ = score_anomaly(model, anom_loader, device)
            labels = np.concatenate([np.zeros(len(scores)), np.ones(len(a_scores))])
            auc = _auc(labels, np.concatenate([scores, a_scores]))
            if auc is not None:
                out["auc"] = auc
    return out


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


def load_checkpoint(models_root: str | Path, model_id: str, device: torch.device) -> tuple[nn.Module, TrainSpec]:
    path = Path(models_root) / model_id / CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(f"no trained model {model_id!r} under {models_root}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    spec = TrainSpec.model_validate(ckpt["spec"])
    model = _build(spec).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, spec


def evaluate_model(spec: EvalSpec, data_root: str | Path, models_root: str | Path) -> EvalResult:
    t0 = time.perf_counter()
    data_root, models_root = Path(data_root), Path(models_root)
    try:
        device = get_device()
        model, train_spec = load_checkpoint(models_root, spec.model_id, device)
        records = load_run_records(data_root, spec.run_ids)
        if spec.max_images is not None:
            records = records[: spec.max_images]
        if not records:
            raise ValueError("no images in the given runs")
        loader = _loader(records, data_root, train_spec, shuffle=False)

        per_image: list[dict] = []
        per_class: dict[str, dict[str, float]] = {}
        metrics: dict[str, float] = {}
        if train_spec.kind == "classifier":
            probs, labels = score_classifier(model, loader, device)
            pred = probs.argmax(1)
            conf = probs.max(1)
            metrics["accuracy"] = float((pred == labels).mean())
            metrics["mean_confidence"] = float(conf.mean())
            auc = _auc(labels, probs)
            for c, name in enumerate(CLASS_NAMES):
                m = labels == c
                if m.any():
                    per_class[name] = {"n": float(m.sum()), "accuracy": float((pred[m] == c).mean()), "mean_confidence": float(conf[m].mean())}
                    ca = _auc((labels == c).astype(int), probs[:, c])
                    if ca is not None:
                        per_class[name]["auc"] = ca
            for rec, p, l, pr, cf in zip(records, probs, labels, pred, conf):
                per_image.append({"image_id": rec.image_id, "label": CLASS_NAMES[l], "pred": CLASS_NAMES[pr], "confidence": float(cf),
                                  "probs": [float(v) for v in p], "mass_fraction": rec.mass_fraction, "snr": rec.snr})
        else:
            scores, labels = score_anomaly(model, loader, device)
            metrics["mean_score"] = float(scores.mean())
            auc = _auc((labels != CLASS_INDEX["none"]).astype(int), scores)
            for c, name in enumerate(CLASS_NAMES):
                m = labels == c
                if m.any():
                    per_class[name] = {"n": float(m.sum()), "mean_score": float(scores[m].mean()), "max_score": float(scores[m].max())}
            for rec, s, l in zip(records, scores, labels):
                per_image.append({"image_id": rec.image_id, "label": CLASS_NAMES[l], "score": float(s),
                                  "mass_fraction": rec.mass_fraction, "snr": rec.snr})

        scores_path = models_root / spec.model_id / eval_scores_filename(spec.run_ids)
        scores_path.write_text(
            json.dumps({"model_id": spec.model_id, "kind": train_spec.kind, "run_ids": spec.run_ids,
                        "auc": None if auc is None or math.isnan(auc) else auc, "metrics": metrics, "images": per_image}),
            encoding="utf-8",
        )
        return EvalResult(
            status="ok", model_id=spec.model_id, kind=train_spec.kind, n_images=len(records),
            auc=None if auc is None or math.isnan(auc) else auc, metrics=metrics, per_class=per_class,
            scores_path=str(scores_path), elapsed_seconds=time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return EvalResult(status="failed", model_id=spec.model_id, elapsed_seconds=time.perf_counter() - t0, message=f"{type(exc).__name__}: {exc}")
