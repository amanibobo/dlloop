# LensCraft — Agentic Gravitational Lensing Simulation Pipeline

**Product requirements & technical design document**
**Status:** draft / pre-implementation — **v3** (revised: Pydantic AI for orchestration, ported ML architectures)

> **Revision history:**
> - v1: build everything from scratch (orchestration + ML architectures).
> - v2: build on Orchestral AI (the framework HEPTAPOD itself uses) + port ML architectures.
> - v3 (this version): swap Orchestral AI for **Pydantic AI** as the orchestration layer. Ported
>   ML architectures from v2 are unchanged. Pydantic AI pairs naturally with the Pydantic-based
>   `LensCard`/`lensjsonl` schemas already in this doc, and is a widely used, well-documented
>   framework, which lowers install/maturity risk relative to Orchestral AI.

---

## 0. One-liner

An agent, orchestrated with Pydantic AI, that plans, proposes, and (with human sign-off) executes
strong-gravitational-lensing simulation batches, trains classifiers and anomaly detectors ported
from the group's own prior papers on the results, and uses what those models are uncertain about
to decide what to simulate next. Lenstronomy/DeepLenseSim is the domain engine being orchestrated,
exactly as MadGraph/Pythia are orchestrated (not reimplemented) in HEPTAPOD.

---

## 1. Motivation & context

Gravitational lensing simulation pipelines built on Lenstronomy (e.g. DeepLenseSim) require
manual parameter configuration, job submission, output validation, and iteration on failures.
This bottlenecks large-scale dataset generation and consumes researcher time on engineering
rather than analysis.

This project draws its orchestration philosophy from **HEPTAPOD** (Menzo, Roman, Gleyzer, Matchev
et al., 2026) — an agentic framework for high-energy-physics workflows built on schema-validated
tools, run-card-driven coordination, and LLM-legible intermediate data formats — and applies the
same philosophy to dark matter substructure classification, which was itself pioneered by the
same lab in two prior papers:

- **Alexander, Gleyzer, McDonough, Toomey, Usai (2019)** — *Deep Learning the Morphology of Dark
  Matter Substructure*: supervised CNN classification (ResNet-18, AlexNet) of strong-lensing
  images into no-substructure / spherical-subhalo / vortex classes, using PyAutoLens simulations.
- **Alexander, Gleyzer, Parul, Reddy, Toomey, Usai, Von Klar (2021)** — *Decoding Dark Matter
  Substructure without Supervision*: unsupervised anomaly detection (deep convolutional
  autoencoder, VAE, adversarial autoencoder, restricted Boltzmann machine) for the same
  classification task, plus a combined unsupervised+supervised pipeline. This paper publishes
  full layer-by-layer architecture specs (Appendix B, Tables IV–VI), which this project ports
  directly rather than re-deriving.

HEPTAPOD itself runs on the Orchestral AI orchestration engine — this project uses **Pydantic AI**
instead for the same role. The schema-validated-tools idea translates directly either way: Pydantic
AI's tools and structured outputs are Pydantic models natively, which is exactly the schema layer
this doc already specifies for `LensCard`/`lensjsonl`.

The novel contribution is not "wrap a simulator with an LLM," and it's also not "reimplement an
agent framework." It's closing the loop between simulation, model training, and
*uncertainty-driven resimulation* — using tools that already exist for the surrounding
infrastructure so the effort goes into that loop.

---

## 2. Goals / non-goals

**Goals**
- An agent, built on Pydantic AI, that accepts a natural-language simulation request, proposes a
  structured `LensCard`, waits for human approval, and executes it against Lenstronomy/DeepLenseSim.
- Training of a substructure classifier (ResNet-18 / AlexNet, ported via torchvision) and an
  anomaly detector (DCAE / VAE / AAE, ported from the 2021 paper's Appendix B) on generated data.
- An uncertainty-sampling loop: the agent inspects model confidence/reconstruction-loss across
  the current dataset and proposes follow-up simulation batches targeting weak regions.
- A quantitative benchmark comparing the tool-assisted agent against a core-tools-only baseline
  (same Pydantic AI setup, no domain tool registry), using HEPTAPOD's pass@k / stagewise-rubric
  methodology.
- A generated technical report/notebook documenting a full run.

**Non-goals**
- Re-deriving agent orchestration machinery that Pydantic AI already provides.
- Re-designing classifier/anomaly-detector architectures the 2019/2021 papers already specified
  and benchmarked — port them, don't redesign them.
- A full RL training loop — uncertainty sampling is the planning signal, not a learned policy.
- SLURM integration is optional/stretch, not required for v1.

---

## 3. System overview

```
 user prompt
     |
     v
 planner agent  --------------------------+
     |                                    |
     v                                    | (uncertainty summary
 human checkpoint                         |  feeds back into next
     | (approves LensCard)                |  proposed LensCard)
     v                                    |
 +----------------+----------------+      |
 | simulation     | ML tools       | docs |
 | tools          |                | tool |
 +----------------+----------------+------+
     |                |              |
     v                v              v
        dataset (lensjsonl) + model checkpoints + report
```

Three tool tiers, one shared sandbox, one human approval gate before any batch executes —
unchanged since v1. What changes across revisions is *how* each tier is implemented.

---

## 4. Core data structures

### 4.1 LensCard

The structured config the agent proposes and the human approves before execution — analogous to
HEPTAPOD's run cards. A plain Pydantic model, which Pydantic AI uses directly for both tool
arguments and structured agent output:

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class LensCard(BaseModel):
    n_images: int = Field(gt=0, le=100_000)
    substructure: Literal["none", "subhalo", "vortex"]
    halo_mass: float                       # M_sun
    redshift_lens: float = 0.5
    redshift_source: float = 1.0
    substructure_mass_fraction: float = Field(default=0.01, ge=0.0, le=1.0)
    psf_sigma: float = 1.0                 # arcsec
    snr_target: float = 20.0
    backend: Literal["lenstronomy", "pyautolens"] = "lenstronomy"
    seed: Optional[int] = None
```

### 4.2 lensjsonl

One JSON record per generated image — the LLM-legible intermediate format the agent reasons over
instead of raw image tensors, mirroring HEPTAPOD's `evtjsonl`. Format-level, unaffected by the
orchestration framework choice:

```json
{
  "image_id": "run003_00042",
  "lens_card": { "...": "..." },
  "substructure_type": "vortex",
  "mass_fraction": 0.012,
  "snr": 18.4,
  "image_path": "data/run003/images/00042.npy",
  "moment_stats": {
    "residual_rms": 0.0031,
    "second_moment": 0.87,
    "arc_ellipticity": 0.21
  }
}
```

`moment_stats` are cheap, computed-at-generation-time image statistics — enough for the agent to
reason about dataset composition and quality without needing multimodal vision. A vision-capable
inspection tool can be added later as a separate, optional tool that reads `image_path` directly.

---

## 5. Orchestration layer — built on Pydantic AI

Pydantic AI centers on an `Agent`, model-provider-agnostic (Anthropic, OpenAI, Google, Groq, and
local models via an OpenAI-compatible endpoint such as Ollama — covering the "local inference
tooling" requirement from the original brief with no extra work), with tools registered as plain
Python functions and structured output enforced via Pydantic models.

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class SimBatchResult(BaseModel):
    status: str
    lensjsonl_path: str
    n_images: int

execution_agent = Agent(
    "anthropic:claude-sonnet-4-6",   # or "openai:gpt-4o", or an Ollama model via provider config
    system_prompt=(
        "You are an agent that orchestrates gravitational lensing simulation workflows. "
        "Given an approved LensCard, call the appropriate tools to execute it."
    ),
)

@execution_agent.tool_plain
def simulate_lens_batch(card: LensCard) -> SimBatchResult:
    """Generate a batch of strong gravitational lensing images using Lenstronomy.

    Args:
        card: simulation configuration (mass, substructure type, redshifts, etc.)
    """
    result = run_deeplensesim(card)
    return SimBatchResult(**result)
```

**Human-in-the-loop checkpoint.** Rather than relying on framework-level pre-execution hooks,
split planning from execution into two agents and let your own driving code sit in between —
which also makes the approval boundary explicit and easy to unit-test:

```python
planning_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    output_type=LensCard,
    system_prompt="Given the user's request, propose a LensCard for the simulation batch.",
)

# 1. Agent proposes a config, no tools involved yet.
proposal = planning_agent.run_sync(user_prompt).output

# 2. Human approves or edits — your own code, not the framework's.
approved_card = get_human_approval(proposal)

# 3. Only the approved card reaches the execution agent's tools.
result = execution_agent.run_sync(
    f"Execute this approved simulation batch: {approved_card.model_dump_json()}"
)
```

This propose → approve → execute split maps directly onto HEPTAPOD's run-card approval boundary
(Section 3.4 of that paper) without depending on a specific framework feature for it.

*(Caveat: check Pydantic AI's current docs before implementing — the library has been actively
adding native support for deferred/approval-gated tool calls, which may let you collapse the
two-agent split above into a single agent with an approval-gated tool. Worth checking at build
time rather than assuming the split above is the only option.)*

---

## 6. Domain tool tiers

| Tier | Tools | Implementation | External dependency |
|---|---|---|---|
| Simulation | `simulate_lens_batch`, `cross_backend_validate` | Pydantic AI tools (`@agent.tool_plain`), `LensCard` validation, lensjsonl writer | Lenstronomy / DeepLenseSim (and optionally PyAutoLens for cross-validation) |
| ML | `train_classifier`, `train_anomaly_detector`, `evaluate_auc`, `sample_uncertainty` | Pydantic AI tools wrapping ported model training/eval code | PyTorch, torchvision (ResNet-18/AlexNet) |
| Docs | `generate_report` | Pydantic AI tool assembling lensjsonl + eval metrics | matplotlib, Marimo (optional) |

---

## 7. ML components — ported architectures

### 7.1 Classifier: ResNet-18 / AlexNet (Alexander et al. 2019)

Standard torchvision architectures, untrained, resized for a 3-class head (no substructure /
subhalo / vortex) — exactly the two architectures compared in the first paper:

```python
import torchvision.models as tvm
import torch.nn as nn

def build_classifier(arch: str = "resnet18", n_classes: int = 3):
    if arch == "resnet18":
        model = tvm.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale input
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "alexnet":
        model = tvm.alexnet(weights=None)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return model
```

### 7.2 Anomaly detectors: DCAE / VAE / AAE (Alexander et al. 2021, Appendix B, Tables IV–VI)

Ported layer-for-layer from the paper's published specs (150×150 grayscale input):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCAEEncoder(nn.Module):
    """Table IV encoder."""
    def __init__(self, latent_dim: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, stride=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 7, stride=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)
        self.fc = nn.Linear(5184, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.fc(x.flatten(1))
        return self.bn(x)

class DCAEDecoder(nn.Module):
    """Table IV decoder."""
    def __init__(self, latent_dim: int = 1000):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 5184)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 7)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 7, stride=3, padding=1, output_padding=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 6, stride=3, padding=1, output_padding=2)

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 9, 9)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return torch.tanh(self.deconv3(x))

class DCAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = DCAEEncoder(), DCAEDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAEEncoder(nn.Module):
    """Table V encoder — same conv stack as DCAE, two linear heads instead of one."""
    def __init__(self, latent_dim: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, stride=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 7, stride=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)
        self.fc_mu = nn.Linear(5184, latent_dim)
        self.fc_logvar = nn.Linear(5184, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = DCAEDecoder()  # decoder architecture is shared with DCAE

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar


class AAEEncoder(nn.Module):
    """Table VI encoder — same conv stack, no BatchNorm."""
    def __init__(self, latent_dim: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, stride=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 7, stride=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)
        self.fc = nn.Linear(5184, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return self.fc(x.flatten(1))

class AAEDiscriminator(nn.Module):
    """Table VI discriminator."""
    def __init__(self, latent_dim: int = 1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)

class AAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AAEEncoder()
        self.decoder = DCAEDecoder()  # decoder architecture is shared with DCAE
        self.discriminator = AAEDiscriminator()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
```

**Which one to actually run in the active-learning loop:** the 2021 paper found AAE the strongest
anomaly detector (AUC ≈ 0.932, near an "optimal" trained detector at ≈ 0.934), with VAE close
behind and DCAE noticeably weaker — so AAE is the natural default for `sample_uncertainty`, with
DCAE/VAE kept around as comparison baselines for the benchmark section rather than the production
loop. RBM is skipped (weakest performer in the paper, not worth the extra implementation for a
supporting role).

---

## 8. Active-learning / uncertainty loop

`sample_uncertainty` runs the current classifier and/or AAE over a held-out slice of the dataset
and returns the LensCard region — which substructure type, which mass-fraction range — where
classifier confidence is lowest or reconstruction loss is highest. That summary is appended to
the planning agent's context; it drafts the next LensCard targeting the gap.

This is the project's central novel contribution: uncertainty-sampling-driven simulation-parameter
search in place of a full RL policy — a defensible, finishable stand-in that's still honestly
"planning under a feedback signal." Nothing about this loop changes with the orchestration
framework choice — it's bespoke logic sitting on top of ported model outputs either way.

---

## 9. Benchmark & evaluation methodology

Adopted directly from HEPTAPOD Section 5 / Appendix A:

- **Tool-assisted arm:** full tool registry (simulation, ML, docs tiers) registered on the
  Pydantic AI execution agent.
- **Core-only arm:** same Pydantic AI setup, agent given only a raw shell/file tool and the
  Lenstronomy install path — no domain tool registry.
- **Stagewise rubric:** simulate → label/cluster → train → correct AUC (each stage weighted,
  cumulative — a stage only counts if every prior stage passed).
- **Metrics:** pass@k (probability at least one of k attempts succeeds) and passk (probability
  all k attempts succeed), computed via the standard unbiased estimators; reach (rubric-weighted
  partial credit) for runs that don't fully complete.
- Run N ≥ 10 trials per arm, same task, same seed sandbox scaffold each time.

---

## 10. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| LLM backend | Pydantic AI's model routing | Anthropic / OpenAI / Google / Groq / local Ollama via OpenAI-compatible provider |
| Orchestration | Pydantic AI (`Agent`, `@agent.tool_plain`, `output_type`) | Two-agent propose/approve/execute split for the human checkpoint |
| Schema validation | Pydantic models (`LensCard`, tool args, tool outputs) | Native to Pydantic AI |
| Simulation | Lenstronomy via DeepLenseSim; PyAutoLens optional for cross-validation | External domain engine, not reimplemented |
| ML | PyTorch + torchvision (ResNet-18/AlexNet); DCAE/VAE/AAE ported from paper Appendix B | See Section 7 |
| Compute | Local subprocess execution behind a `submit()/status()` interface | SLURM implementation optional/stretch, same interface |
| Reporting | matplotlib + markdown, or a Marimo notebook for a reactive report | Optional interactive upgrade |

---

## 11. Build phases

1. **Data model.** `LensCard`, `lensjsonl`, and a standalone `simulate_lens_batch` function
   calling DeepLenseSim directly — get this working before wiring up the agent.
2. **Orchestration setup.** Install Pydantic AI; define the planning agent (`output_type=LensCard`)
   and execution agent (`@agent.tool_plain` for the simulation/ML/docs tools); implement the
   propose → human-approve → execute flow.
3. **ML tier.** Port ResNet-18/AlexNet classifier and DCAE/VAE/AAE anomaly detectors from Section
   7; `evaluate_auc` tool.
4. **Active-learning loop.** `sample_uncertainty` — closes the loop. Primary demo moment.
5. **Cross-backend validation** (Lenstronomy vs PyAutoLens) — stretch, high credibility payoff.
6. **Benchmark harness** (pass@k, tool-assisted vs core-only) + report generator.

---

## 12. Suggested repo structure

```
lenscraft/
  agent/
    tools.py           # Pydantic AI tool functions (@agent.tool_plain), sim/ML/docs tiers
    config.py           # planning_agent + execution_agent setup, provider config
  schema/
    lens_card.py          # LensCard (Pydantic)
    lensjsonl.py            # record read/write helpers
  models/
    classifier.py              # ResNet-18 / AlexNet builders (torchvision)
    autoencoders.py               # DCAE, VAE, AAE (ported from paper Appendix B)
  compute/
    backend.py                      # submit()/status() interface, local subprocess impl
  bench/
    rubric.py                         # stagewise rubric, pass@k / passk estimators
    run_benchmark.py                    # N-trial driver, tool-assisted vs core-only
  data/                                    # generated lensjsonl + images (gitignored)
  reports/                                   # generated reports/notebooks (gitignored)
```

---

## 13. Open questions / risks

- **Human-in-the-loop mechanics:** confirm whether Pydantic AI's current release supports
  approval-gated tool calls natively, or whether the two-agent propose/approve/execute split in
  Section 5 is the right pattern going in.
- **Compute cost of trials:** benchmark runs (N ≥ 10 per arm) plus training runs could be slow on
  a laptop — worth profiling a single simulate+train+evaluate cycle early to size N realistically.
- **Cross-backend validation** depends on getting both Lenstronomy and PyAutoLens installed and
  configured consistently — treat as stretch, not core path.
- **Vision-capable inspection tool** (agent looking at actual images, not just `moment_stats`) is
  explicitly out of scope for v1 but worth flagging as a natural v2 extension.

---

## 14. References

1. Menzo, Roman, Gleyzer, Matchev, Fleming, Höche, Mrenna, Shyamsundar. *HEPTAPOD: Orchestrating
   High Energy Physics Workflows Towards Autonomous Agency.* arXiv:2512.15867 (2026).
2. Alexander, Gleyzer, McDonough, Toomey, Usai. *Deep Learning the Morphology of Dark Matter
   Substructure.* arXiv:1909.07346 (2019).
3. Alexander, Gleyzer, Parul, Reddy, Toomey, Usai, Von Klar. *Decoding Dark Matter Substructure
   without Supervision.* arXiv:2008.12731 (2021). — Appendix B, Tables IV–VI (DCAE/VAE/AAE specs).
4. DeepLenseSim repository — https://github.com/mwt5345/DeepLenseSim
5. Pydantic AI documentation — https://ai.pydantic.dev/
