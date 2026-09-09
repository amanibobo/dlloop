# LensCraft

Agentic strong-gravitational-lensing simulation pipeline. See [DESIGN.md](DESIGN.md) for the full
design. This README tracks what is actually built.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | `LensCard`, `lensjsonl`, standalone `simulate_lens_batch` on Lenstronomy | **built** |
| 2 | Pydantic AI agent with approval-gated simulation tool; local + Modal compute backends | **built**, verified live (Kimi K3 on Fireworks; batches on Modal) |
| 3 | ResNet-18/AlexNet classifier, DCAE/VAE/AAE anomaly detectors, `evaluate_model`; GPU training on Modal | **built** |
| 4 | `sample_uncertainty` active-learning loop | not started |
| 5 | PyAutoLens cross-backend validation (stretch) | not started |
| 6 | pass@k benchmark harness + report generator | not started |

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
