# LensCraft

Agentic strong-gravitational-lensing simulation pipeline. See [DESIGN.md](DESIGN.md) for the full
design. This README tracks what is actually built.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | `LensCard`, `lensjsonl`, standalone `simulate_lens_batch` on Lenstronomy | **built** |
| 2 | Pydantic AI agent with approval-gated simulation tool; local + Modal compute backends | **built**, verified live (Kimi K3 on Fireworks; batches on Modal) |
| 3 | ResNet-18/AlexNet classifier, DCAE/VAE/AAE anomaly detectors, `evaluate_model`; GPU training on Modal | **built** |
| 4 | `sample_uncertainty` tool + `lenscraft loop` driver (uncertainty → approve → simulate → retrain → evaluate) | **built** |
| 5 | PyAutoLens backend + `cross_backend_validate` (stretch) | **built**: engines agree to ~2% of peak on identical systems |
| 6a | pass@k benchmark harness: tool-assisted vs core-only (shell + files) arms, stagewise rubric | **built** |
| 6b | report generator (`lenscraft report`, agent tool `generate_report`) | **built** |

## Setup

```sh
uv sync                 # pydantic, lenstronomy, pydantic-ai, modal (no torch on the laptop)
uv run pytest           # unit + integration tests, all offline
```

### LLM provider

`lenscraft agent` works with any Pydantic AI provider. Put the key and default model in a
gitignored `.env` at the repo root (the CLI loads it automatically):

```sh
# Fireworks (open-weight models, OpenAI-compatible endpoint)
FIREWORKS_API_KEY=fw_...
LENSCRAFT_MODEL=fireworks:accounts/fireworks/models/kimi-k3

# or Anthropic
ANTHROPIC_API_KEY=sk-ant-...
LENSCRAFT_MODEL=anthropic:claude-opus-5
```

`--model` on the command line overrides `LENSCRAFT_MODEL`. Pick a model that supports function
calling; the agent has to emit a well-formed `LensCard` as tool arguments. Kimi K3 on Fireworks
has been verified end to end (propose → approve → run → summarize, and a human denial).

### Modal (remote compute + storage)

Simulation batches, the datasets they produce, and later the model training all run on Modal so
nothing large lives on your machine. Images and records go to the `lenscraft-data` Volume.

```sh
uv run modal setup                                   # once: log in, writes ~/.modal.toml
uv run modal deploy -m lenscraft.compute.modal_app   # once per code change
uv run modal run -m lenscraft.compute.modal_app --substructure vortex --n 50 --run-id smoke  # remote smoke test
```

Then add `--backend modal` to any command below. Batches are sharded 500 images per container,
so a 30k-image class renders in parallel; per-image seeding makes the output identical to a
local run.

## Usage

```sh
# natural-language request -> proposed LensCard -> your y/n/edit -> run -> summary
uv run lenscraft agent "300 axion vortex images at mass fraction 0.03 and 300 with no substructure" --backend modal

# the same tool call, no LLM
uv run lenscraft simulate --substructure vortex --n 100 --seed 0 --run-id run001 --backend modal

uv run lenscraft runs --backend modal                    # list runs on the Volume
uv run lenscraft summarize run001 --backend modal        # composition summary (what the agent reads)
uv run lenscraft fetch run001 --backend modal --out data # copy records (not images) locally

# ML tier (GPU on Modal; torch is never installed on the laptop by `uv sync`)
uv run lenscraft train --backend modal --kind classifier --arch resnet18 --model-id clf_a \
    --runs train_none train_sub train_vor --epochs 10
uv run lenscraft train --backend modal --kind anomaly --arch aae --model-id aae_a \
    --runs train_none train_vor --epochs 10          # trains on 'none' only; vortex held out for AUC
uv run lenscraft evaluate --backend modal --model-id clf_a --runs test_none test_sub test_vor
uv run lenscraft models --backend modal
```

The agent exposes the same as tools: `train_classifier`, `train_anomaly_detector` (both
approval-gated), `evaluate_model`, `list_models`.

From Python:

```python
from lenscraft import LensCard
from lenscraft.compute import get_backend

backend = get_backend("modal")
job = backend.submit(LensCard(n_images=100, substructure="subhalo", seed=0), run_id="run003")
print(backend.wait(job).result)
```

Each run produces `<root>/<run_id>/card.json`, `records.lensjsonl`, and `images/*.npy` (float32),
where `<root>` is `data/` locally or `/data` on the Modal Volume.

## How the agent works

One Pydantic AI agent (`lenscraft.agent`) with three tools: `simulate_lens_batch`
(approval-gated), `summarize_dataset`, `list_runs`. When the model calls the simulate tool the
run pauses; `run_with_approval` hands the proposed card to a human callback which returns
approve / approve-with-edits / deny, and the run resumes with that decision. Denials are shown to
the model with the reason. The `--core-only` flag builds the same agent with no domain tools,
which is the benchmark baseline arm from DESIGN.md §9. Tools talk only to a `ComputeBackend`,
so `--backend local|modal` changes where work happens without touching agent code.

## The active-learning loop

`sample_uncertainty` (tool, CLI `lenscraft uncertainty`) joins a model's per-image scores on
held-out runs with the simulation records and bins each substructure class along physical axes:
mass fraction, SNR, axion mass (vortex), subhalo count (CDM). Per-image uncertainty is
1 − P(true class) for classifiers, and for anomaly detectors the fraction of no-substructure
images whose reconstruction error is at least as large (how ordinary the anomaly looks). The
weakest cell becomes a `suggested_card` that adds images there, which goes through the same
approval gate as any other batch.

```sh
uv run lenscraft uncertainty --backend modal --model-id clf_resnet_5k --runs te1k_none te1k_sub te1k_vor
uv run lenscraft loop --backend modal --model-id clf_resnet_5k \
    --train-runs tr5k_none tr5k_sub tr5k_vor --test-runs te1k_none te1k_sub te1k_vor --rounds 2
```

`lenscraft loop` is the deterministic driver: uncertainty → your approval → simulate → retrain on
the enlarged set under `<model_id>_r<n>` → re-evaluate on the *same* held-out runs, printing
before/after metrics per round. The agent does the same through its tools when asked; the driver
exists so the loop is demonstrable, testable offline, and usable as a scripted benchmark arm.

## Cross-backend validation (PyAutoLens)

`lenscraft.sim.pyautolens_backend` renders the *same* `LensSystem` the lenstronomy backend builds
through PyAutoLens, and `cross_backend_validate` (tool, CLI `lenscraft crosscheck`) compares the
unconvolved noiseless models image by image. The convention mapping was established empirically
by minimising the residual over every orientation and sign combination: PyAutoLens takes
`ell_comps = (e2, e1)`, shear unchanged, centres as `(y, x)`, its native array is flipped
vertically, and it returns surface brightness (lenstronomy: counts per pixel, i.e. × pixel area).

```sh
uv run lenscraft crosscheck --backend modal --substructure vortex --n 5     # PyAutoLens lives in its own Modal image
uv run --python .venv-autolens/bin/python -m lenscraft.cli crosscheck --substructure subhalo   # or a local venv with autolens
```

Result on Modal (96 px, 0.05"/px, 5 systems per class, seed 7):

| class | shape rel. rms (mean / max, fraction of peak) | flux ratio | centroid shift |
|---|---|---|---|
| none | 0.0063 / 0.0099 | 0.996 | ≤ 0.05 px |
| subhalo | 0.0074 / 0.0118 | 0.996 | ≤ 0.04 px |
| vortex | 0.0057 / 0.0065 | 0.996 | ≤ 0.04 px |

The residual per-cent level is the two packages' slightly different elliptical-radius
conventions, not a bug on either side; the point-mass substructure agrees as well as the smooth
lens does.

PyAutoLens is not a core dependency (it pins an older scikit-learn); install it into a separate
venv (`uv venv .venv-autolens && uv pip install --python .venv-autolens/bin/python autolens -e .`)
or use the Modal backend. `LensCard(backend="pyautolens")` renders full datasets with it
(`custom` instrument only).

## Benchmark (HEPTAPOD-style)

```sh
uv run lenscraft bench --trials 10 --arms tool core            # default tiny task, ~1-3 min per trial
uv run lenscraft bench --trials 5 --arms core --keep --out reports/bench_core
```

Two arms, same model, same task prompt, fresh sandbox per trial, local backend:

- **tool**: the LensCraft agent with its domain tools (batches and training auto-approved).
- **core**: the same model with only `run_shell` / `write_file` / `read_file` / `list_files` in
  the sandbox, Python with lenstronomy and torch on PATH, the required output layout spelled out
  in the prompt, and the `lenscraft` package blocked. It has to build the pipeline itself.

The task: simulate train and held-out test runs for the three classes, train a ResNet-18, report
the held-out macro AUC. Grading reads the sandbox, so it is identical for both arms. Stages are
cumulative and weighted: simulate 0.25 → label 0.15 → train 0.35 (training runs only) → AUC 0.25
(evaluated on the held-out runs, and the AUC the agent *reports* matches the file). *Reach* is the
summed weight of passed stages; pass@k and pass^k use the unbiased estimators. Results go to
`reports/bench/results.jsonl` and `summary.json`.

### Result (Kimi K3 on Fireworks, 3 trials per arm, 2026-09-15, `reports/bench_kimi3`)

| arm | pass | pass@1 | reach | mean requests | mean tokens | mean wall time | held-out AUC reached |
|---|---|---|---|---|---|---|---|
| tool | 3/3 | 1.00 | 1.00 | 6 | 36k | 103 s | 0.61, 0.61, 0.61 |
| core | 3/3 | 1.00 | 1.00 | 30 | 857k | 894 s | 0.85, 0.96, 0.96 |

With a strong model both arms complete the task every time, so the value of the domain tools
shows up as cost and reliability: 24× fewer tokens, 9× less wall time, 5× fewer model
requests, and a deterministic, reproducible pipeline (identical AUC across trials). Two
caveats worth stating plainly. First, n = 3 per arm; the estimators are exact for the sample
but the sample is small. Second, the core arm reached *higher* AUCs on this tiny task by
engineering its own preprocessing (an asinh stretch, dihedral augmentation, a small-image ResNet
stem), which the tool arm's fixed training recipe does not do. The first two are data-pipeline
choices the ML tier should adopt; the stem is an architecture change and stays out of the port.
An earlier run with a weaker task config (`reports/bench_kimi2`) was cut short by a network
outage, which is why the harness now retries infrastructure errors instead of scoring them.

## Reports

```sh
uv run lenscraft report --backend modal --runs tr5k_none tr5k_sub tr5k_vor te1k_none te1k_sub te1k_vor al_r1 \
    --models clf_base_b clf_al_b aae_5k_v2 --eval-runs te1k_none te1k_sub te1k_vor \
    --loop-json loop.out --bench-summary reports/bench/summary.json --out reports/run_5k
```

Writes `report.md` plus PNG figures: dataset composition and a sample-image grid, per-model
training curves, held-out confusion matrix or anomaly-score distributions, the uncertainty
cells with the proposed next batch, the active-learning rounds table, and the benchmark's
stagewise pass rates. The agent's `generate_report` tool produces the same under `reports/<name>`.

## ML notes

`lenscraft.models` ports the architectures verbatim: torchvision ResNet-18 / AlexNet with a
single-channel stem and 3-class head (Alexander et al. 2019), and the DCAE / VAE / AAE encoder,
shared decoder and discriminator from Alexander et al. 2021 Appendix B. The paper's autoencoders
are specified at 150×150 (5184 conv features), so `TrainSpec.input_size` defaults to 150 and images
are resized on load; the flattened size and decoder output paddings are derived from `input_size`
rather than hard-coded so other sizes where the arithmetic closes (96, 204) also work. Anomaly
detectors train on the `none` class only and score images by reconstruction MSE; any substructure
images passed in are held out and used to report a separation AUC. Checkpoints live under
`_models/<model_id>/` next to the runs (locally under `data/`, on the Volume under `/data`), and
`evaluate_model` writes per-image scores there for the Phase 4 uncertainty loop.

### Reference results (2026-09-09, Modal L4, Euclid 64 px, mass fraction 0.03, 10 epochs)

Runs on the Volume: `tr5k_{none,sub,vor}` (5,000 each, seeds 100-102) and `te1k_{none,sub,vor}`
(1,000 each, seeds 200-202). Evaluation is on the 3,000 held-out test images.

| Model | Train images | Wall time | Test accuracy | Test AUC |
|---|---|---|---|---|
| `clf_resnet_5k` (ResNet-18) | 15,000 | ~5 min | 0.919 (none 1.00 / subhalo 0.92 / vortex 0.84) | 0.977 macro |
| `clf_resnet_a` (ResNet-18) | 900 | ~3 min | 0.373 (collapses to "none") | 0.814 macro |
| `aae_5k_v2` (AAE) | 4,000 none | 44 s | – | ~0.57 substructure-vs-none |

**One active-learning round** (same recipe, seed 0, best-validation checkpoint, cosine LR).
`sample_uncertainty` on `clf_base_b` flagged vortices with axion mass in [1e-24, 2.8e-24] eV
(the longest vortex lines, 1.8-3") at 29% accuracy. The loop simulated 2,000 vortex images at
1.7e-24 eV (`al_r1`) and retrained:

| Model | Training data | Test acc | Test AUC | none / subhalo / vortex acc | Weak cell acc |
|---|---|---|---|---|---|
| `clf_base_b` | 15,000 | 0.928 | 0.979 | 1.00 / 0.96 / 0.82 | 0.29 |
| `clf_uni_b` | 15,000 + 2,000 **uniform** (`uni_*`, 667/class, default axion sampling) | 0.930 | 0.978 | 1.00 / 0.97 / 0.82 | 0.28 |
| `clf_al_b` | 15,000 + 2,000 **targeted** (`al_r1`) | 0.900 | 0.980 | 1.00 / 0.79 / 0.91 | 0.62 |

Same budget, same recipe, same seed: the uniform batch is indistinguishable from no batch at
all, while the targeted batch doubles accuracy in the weak region and lifts vortex accuracy
overall, at the cost of subhalo accuracy (the added light-axion vortices are the ones that look
most like subhalos, and the classes are now unbalanced). A second round would flag subhalos as
the weakest class and propose the balancing batch. That is the loop working as intended: the
uncertainty signal finds a region uniform sampling cannot, and the remaining question is how
to trade the targeted gain against the imbalance it introduces (e.g. class-balanced batches, or
re-weighting).

The classifier matches the ballpark of Alexander et al. 2019. The AAE is far from the 2021 paper's
0.93: at this mass fraction and resolution the substructure perturbs the arcs by ~1% of the peak,
and 10 epochs is a small fraction of what the paper trained for. Longer training, larger mass
fractions, and the 150 px `custom` instrument are the obvious levers, and are exactly what the
Phase 4 uncertainty loop is meant to explore.

## Simulation engine notes

`lenscraft.sim.lenstronomy_backend` is a port of
[DeepLenseSim](https://github.com/mwt5345/DeepLenseSim)'s `deeplense/lens.py` driven by a
`LensCard` instead of hard-coded constants. The lens (SIE + shear), source (Sersic), CDM subhalo
draw, axion-vortex construction, and both observing setups (Euclid VIS via lenstronomy's
`SimAPI`; a generic 0.05"/px setup) follow DeepLenseSim. pyHalo and colossus are not needed and
are not dependencies.

Deliberate deviations from DeepLenseSim, all fixing bugs on its side:

1. Einstein radii use angular-diameter distances rather than luminosity distances
   (1e12 M_sun at z=0.5 → 1.66" instead of 1.35").
2. The CDM subhalo count follows its Poisson draw (DeepLenseSim draws it and then ignores it).
3. The Euclid-path image is model + noise (DeepLenseSim adds the model to itself).
4. Subhalo masses are rescaled so their total equals `substructure_mass_fraction * halo_mass`,
   giving the mass-fraction field the same meaning for both substructure classes.
5. All randomness comes from a per-image `numpy` generator seeded from `card.seed` and the image
   index, so a seeded card is bit-for-bit reproducible regardless of batch size or sharding.

A card whose field of view is narrower than the Einstein radius is rejected with a structured
failure instead of silently producing pure-noise images.

Deviations from the `LensCard` sketch in DESIGN.md §4.1: `psf_sigma`/`snr_target` were replaced
by `psf_fwhm`, `source_magnitude`, and instrument settings, because that is how DeepLenseSim
controls depth. The *measured* SNR is recorded per image in `lensjsonl` instead.
