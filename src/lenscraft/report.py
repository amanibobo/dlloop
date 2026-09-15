"""Report generator (DESIGN.md Section 6, docs tier): a markdown report with figures for a set of
runs, models, an optional active-learning history and an optional benchmark summary.

Works against any ``ComputeBackend``; figures are matplotlib PNGs written next to ``report.md``.
Colour is by role: the three substructure classes take the first three categorical slots of the
validated reference palette (all-pairs safe), magnitude uses a single blue ramp, arms are blue /
orange, and text never wears a series colour.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lenscraft.compute.backend import ComputeBackend
from lenscraft.schema.lensjsonl import LensRecord, summarize_records
from lenscraft.schema.ml import CLASS_NAMES

CLASS_COLORS = {"none": "#2a78d6", "subhalo": "#eb6834", "vortex": "#1baf7a"}
ARM_COLORS = {"tool": "#2a78d6", "core": "#eb6834"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ_CMAP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


class ReportResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    path: str = ""
    figures: list[str] = Field(default_factory=list)
    message: str = ""


# --------------------------------------------------------------------------------------------
# matplotlib helpers
# --------------------------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "legend.frameon": False,
    })
    return plt


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, out_dir: Path, name: str, figures: list[str]) -> str:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    figures.append(path.name)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path.name


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------


def _dataset_section(backend: ComputeBackend, run_ids: list[str], out_dir: Path, figures: list[str], n_samples: int) -> tuple[str, dict[str, list[LensRecord]]]:
    plt = _plt()
    per_run: dict[str, list[LensRecord]] = {}
    rows = ["| run | class | n | mass fraction | SNR median | residual_rms median | seed |", "|---|---|---|---|---|---|---|"]
    for run_id in run_ids:
        recs = backend.read_records(run_id)
        per_run[run_id] = recs
        s = summarize_records(recs)
        cls = recs[0].substructure_type if recs else "?"
        rows.append(
            f"| `{run_id}` | {cls} | {len(recs)} | {s['mass_fraction']['median']:.3g} | {s['snr']['median']:.0f} | "
            f"{statistics.median(r.moment_stats.residual_rms for r in recs):.3g} | {recs[0].lens_card.seed if recs else ''} |"
        )
    all_recs = [r for rs in per_run.values() for r in rs]

    # SNR and residual distributions per class: small multiples, one hue per class, legend + direct label
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key, label in ((axes[0], "snr", "SNR"), (axes[1], "residual_rms", "substructure residual (fraction of peak)")):
        # 'none' has residual 0 by construction; it would collapse the residual axis, so it is left off that panel
        classes = [c for c in CLASS_NAMES if not (key == "residual_rms" and c == "none")]
        pooled = [r.snr if key == "snr" else r.moment_stats.residual_rms for r in all_recs if r.substructure_type in classes]
        if not pooled:
            continue
        bins = np.linspace(min(pooled), max(pooled) or 1.0, 31)  # shared bins so the classes are comparable
        for cls in classes:
            vals = [r.snr if key == "snr" else r.moment_stats.residual_rms for r in all_recs if r.substructure_type == cls]
            if vals:
                ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=CLASS_COLORS[cls], label=f"{cls} (n={len(vals)})")
        ax.set_xlabel(label)
        ax.set_ylabel("images")
        ax.legend()
        _style(ax)
    axes[0].set_title("Dataset composition")
    fig_name = _save(fig, out_dir, "dataset_distributions", figures)

    # sample images, one row per class, sqrt stretch
    grid_name = ""
    classes = [c for c in CLASS_NAMES if any(r.substructure_type == c for r in all_recs)]
    if classes and hasattr(backend, "read_image"):
        fig, axes = plt.subplots(len(classes), n_samples, figsize=(1.6 * n_samples, 1.7 * len(classes)), squeeze=False)
        for i, cls in enumerate(classes):
            recs = [r for r in all_recs if r.substructure_type == cls][:n_samples]
            for j in range(n_samples):
                ax = axes[i][j]
                ax.axis("off")
                if j < len(recs):
                    img = backend.read_image(recs[j].image_id.rsplit("_", 1)[0], recs[j].image_path)
                    if img is not None:
                        ax.imshow(np.sqrt(np.clip(img, 0, None)), cmap="gray")
                if j == 0:
                    ax.set_title(cls, fontsize=9, loc="left")
        grid_name = _save(fig, out_dir, "sample_images", figures)

    md = ["## Dataset", "", *rows, "", f"![dataset distributions]({fig_name})", ""]
    if grid_name:
        md += [f"![sample images (sqrt stretch)]({grid_name})", ""]
    return "\n".join(md), per_run


def _model_section(backend: ComputeBackend, model_id: str, eval_runs: list[str], out_dir: Path, figures: list[str]) -> str:
    plt = _plt()
    md = [f"## Model `{model_id}`", ""]
    tr = backend.read_train_result(model_id) if hasattr(backend, "read_train_result") else None
    if tr:
        md.append(f"- kind **{tr.get('kind')}**, arch **{tr.get('arch')}**, trained on {', '.join(f'`{r}`' for r in tr.get('run_ids', []))} "
                  f"({tr.get('n_train', '?')} train / {tr.get('n_val', '?')} val images, {tr.get('epochs_run', '?')} epochs, best epoch {tr.get('metrics', {}).get('best_epoch', '?')})")
        hist = tr.get("history") or []
        if hist:
            fig, ax = plt.subplots(figsize=(6, 3))
            ep = [h["epoch"] for h in hist]
            ax.plot(ep, [h["train_loss"] for h in hist], color=CLASS_COLORS["none"], linewidth=2, label="train loss")
            if "val_loss" in hist[0]:
                ax.plot(ep, [h.get("val_loss", np.nan) for h in hist], color=CLASS_COLORS["subhalo"], linewidth=2, label="val loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title(f"Training curves, {model_id}")
            ax.legend()
            _style(ax)
            md.append(f"\n![training curves]({_save(fig, out_dir, f'train_{model_id}', figures)})\n")
            key = "val_acc" if "val_acc" in hist[-1] else ("auc" if "auc" in hist[-1] else None)
            if key:
                fig, ax = plt.subplots(figsize=(6, 2.6))
                ax.plot(ep, [h.get(key, np.nan) for h in hist], color=CLASS_COLORS["vortex"], linewidth=2)
                ax.set_ylim(0, 1)
                ax.set_xlabel("epoch")
                ax.set_ylabel(key)
                ax.set_title(f"Validation {key}, {model_id}")
                _style(ax)
                md.append(f"![validation {key}]({_save(fig, out_dir, f'val_{model_id}', figures)})\n")
    else:
        md.append("- (no train_result.json found)")

    scores = backend.read_eval_scores(model_id, eval_runs) if eval_runs else None
    if scores:
        images = scores.get("images", [])
        md.append(f"\nEvaluated on {', '.join(f'`{r}`' for r in eval_runs)}: **{len(images)} images**, "
                  f"AUC **{scores.get('auc'):.3f}**" if isinstance(scores.get("auc"), (int, float)) else f"\nEvaluated on {', '.join(eval_runs)}: {len(images)} images")
        if scores.get("kind") == "classifier" and images:
            labels = [im["label"] for im in images]
            preds = [im["pred"] for im in images]
            md += ["", "| true \\ predicted | " + " | ".join(CLASS_NAMES) + " | accuracy |", "|---|" + "---|" * (len(CLASS_NAMES) + 1)]
            for c in CLASS_NAMES:
                n = sum(1 for l in labels if l == c)
                if n == 0:
                    continue
                counts = [sum(1 for l, p in zip(labels, preds) if l == c and p == p2) for p2 in CLASS_NAMES]
                md.append(f"| {c} | " + " | ".join(str(x) for x in counts) + f" | {counts[CLASS_NAMES.index(c)] / n:.3f} |")
        elif images:
            fig, ax = plt.subplots(figsize=(6, 3))
            for cls in CLASS_NAMES:
                vals = [im["score"] for im in images if im["label"] == cls]
                if vals:
                    ax.hist(vals, bins=30, histtype="step", linewidth=2, color=CLASS_COLORS[cls], label=f"{cls} (n={len(vals)})")
            ax.set_xlabel("reconstruction error")
            ax.set_ylabel("images")
            ax.set_title(f"Anomaly scores by class, {model_id}")
            ax.legend()
            _style(ax)
            md.append(f"\n![anomaly scores]({_save(fig, out_dir, f'scores_{model_id}', figures)})\n")

        # uncertainty cells for this model on the evaluated runs
        try:
            from lenscraft.ml.uncertainty import sample_uncertainty

            rep = sample_uncertainty(backend, model_id, eval_runs)
        except Exception as exc:  # noqa: BLE001
            rep = None
            md.append(f"\n(uncertainty analysis unavailable: {exc})")
        if rep is not None and rep.status == "ok" and rep.cells:
            cells = rep.cells[:10]
            fig, ax = plt.subplots(figsize=(7, 0.38 * len(cells) + 1.2))
            names = [f"{c.substructure} · {c.axis} [{c.low:.3g}, {c.high:.3g}]  n={c.n}" for c in cells][::-1]
            vals = [c.mean_uncertainty for c in cells][::-1]
            cols = [CLASS_COLORS.get(c.substructure, MUTED) for c in cells][::-1]
            ax.barh(names, vals, color=cols, height=0.6)
            ax.set_xlim(0, 1)
            ax.set_xlabel("mean uncertainty")
            ax.set_title(f"Where {model_id} is weakest")
            _style(ax)
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            md.append(f"\n![uncertainty cells]({_save(fig, out_dir, f'uncertainty_{model_id}', figures)})\n")
            md.append(f"{rep.rationale}\n")
    elif eval_runs:
        md.append(f"\n(not yet evaluated on {', '.join(eval_runs)})")
    return "\n".join(md) + "\n"


def _loop_section(history: list[dict[str, Any]]) -> str:
    md = ["## Active-learning rounds", "", "| round | weakest cell | new run | images | new model | before | after |", "|---|---|---|---|---|---|---|"]
    for h in history:
        w = h.get("weakest") or {}
        cell = f"{w.get('substructure')} {w.get('axis')} [{w.get('low', 0):.3g}, {w.get('high', 0):.3g}] acc {w.get('score', 0):.2f}" if w else "-"
        fmt = lambda m: ", ".join(f"{k} {v:.3f}" for k, v in m.items() if k in ("accuracy", "auc"))  # noqa: E731
        md.append(f"| {h.get('round')} | {cell} | `{h.get('new_run_id') or '-'}` | {h.get('n_new_images', 0)} | `{h.get('new_model_id') or '-'}` | "
                  f"{fmt(h.get('metrics_before', {}))} | {fmt(h.get('metrics_after', {}))} {h.get('message', '')} |")
    return "\n".join(md) + "\n"


def _bench_section(summary: dict[str, Any], out_dir: Path, figures: list[str]) -> str:
    plt = _plt()
    arms = summary.get("arms", [])
    md = ["## Benchmark", "", "| arm | n | pass | pass@1 | pass@3 | reach | mean requests | mean tokens | mean time |", "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        md.append(f"| {a['arm']} | {a['n']} | {a['successes']} | {a['pass_at_k'].get('1', 0):.2f} | {a['pass_at_k'].get('3', float('nan')):.2f} | "
                  f"{a['reach_mean']:.2f} ± {a['reach_std']:.2f} | {a['mean_requests']:.0f} | {a['mean_tokens']:.0f} | {a['mean_duration_seconds']:.0f} s |")
    if arms:
        stages = list(arms[0]["stage_pass_rate"])
        fig, ax = plt.subplots(figsize=(6.5, 3))
        width = 0.8 / len(arms)
        x = np.arange(len(stages))
        for i, a in enumerate(arms):
            ax.bar(x + i * width - 0.4 + width / 2, [a["stage_pass_rate"][s] for s in stages], width * 0.92, color=ARM_COLORS.get(a["arm"], MUTED), label=a["arm"])
        ax.set_xticks(x, stages)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("stage pass rate")
        ax.set_title("Stagewise pass rate by arm")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(arms))
        _style(ax)
        md.append(f"\n![stage pass rates]({_save(fig, out_dir, 'bench_stages', figures)})\n")

        # cost per trial: two measures of different scale -> two panels, one axis each
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
        for ax, key, label in ((axes[0], "mean_requests", "model requests"), (axes[1], "mean_tokens", "tokens"), (axes[2], "mean_duration_seconds", "wall time (s)")):
            names = [a["arm"] for a in arms]
            vals = [a[key] for a in arms]
            ax.bar(names, vals, color=[ARM_COLORS.get(n, MUTED) for n in names], width=0.6)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
            ax.set_ylabel(label)
            ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
            _style(ax)
        axes[0].set_title("Mean cost per trial")
        md.append(f"![cost per trial]({_save(fig, out_dir, 'bench_cost', figures)})\n")
    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def generate_report(
    backend: ComputeBackend,
    *,
    run_ids: list[str],
    model_ids: list[str] | None = None,
    eval_runs: list[str] | None = None,
    out_dir: str | Path = "reports/report",
    title: str = "LensCraft run report",
    loop_history: list[dict[str, Any]] | None = None,
    bench_summary: dict[str, Any] | None = None,
    n_samples: int = 6,
) -> ReportResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    try:
        parts = [f"# {title}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from backend `{backend.name}`.", ""]
        ds_md, _ = _dataset_section(backend, run_ids, out_dir, figures, n_samples)
        parts.append(ds_md)
        for model_id in model_ids or []:
            parts.append(_model_section(backend, model_id, list(eval_runs or []), out_dir, figures))
        if loop_history:
            parts.append(_loop_section(loop_history))
        if bench_summary:
            parts.append(_bench_section(bench_summary, out_dir, figures))
        path = out_dir / "report.md"
        path.write_text("\n".join(parts), encoding="utf-8")
        return ReportResult(path=str(path), figures=figures)
    except Exception as exc:  # noqa: BLE001
        return ReportResult(status="failed", figures=figures, message=f"{type(exc).__name__}: {exc}")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
