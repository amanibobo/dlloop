export const REPO = "https://github.com/amanibobo/lensgrav";

// Interactive marimo apps, all WASM exports served from /marimo/... on this site. The playground
// uses a precomputed grid of simulations, so nothing runs on a server when someone plays with it.
export const embeds = {
  explorer: "/marimo/uncertainty/index.html",
  passk: "/marimo/passk/index.html",
  playground: "/marimo/playground/index.html",
};

export const papers = {
  heptapod: { title: "HEPTAPOD: Orchestrating High Energy Physics Workflows Towards Autonomous Agency", authors: "Menzo, Roman, Gleyzer, Matchev, Fleming, Höche, Mrenna, Shyamsundar", year: "2026", href: "https://arxiv.org/pdf/2512.15867" },
  morphology: { title: "Deep Learning the Morphology of Dark Matter Substructure", authors: "Alexander, Gleyzer, McDonough, Toomey, Usai", year: "2019", href: "https://arxiv.org/pdf/1909.07346" },
  decoding: { title: "Decoding Dark Matter Substructure without Supervision", authors: "Alexander, Gleyzer, Parul, Reddy, Toomey, Usai, Von Klar", year: "2021", href: "https://arxiv.org/pdf/2008.12731" },
  deeplensesim: { title: "DeepLenseSim", authors: "Toomey", year: "GitHub", href: "https://github.com/mwt5345/DeepLenseSim" },
};

export const nav = [
  { id: "overview", label: "Overview" },
  { id: "problem", label: "Problem" },
  { id: "idea", label: "Idea" },
  { id: "decisions", label: "Decisions" },
  { id: "architecture", label: "Architecture" },
  { id: "build", label: "Build" },
  { id: "results", label: "Results" },
  { id: "demos", label: "Demos" },
  { id: "limits", label: "Limits" },
  { id: "reproduce", label: "Reproduce" },
];

export const stats = [
  { value: "0.93", label: "held-out accuracy", sub: "ResNet-18, 15k simulated images, 3 classes" },
  { value: "0.29 → 0.62", label: "accuracy in the weakest region", sub: "after one uncertainty-driven round; a uniform batch of the same size: 0.28" },
  { value: "24×", label: "fewer tokens with domain tools", sub: "36k vs 857k per task, both arms 3/3 passing" },
  { value: "≤ 1.2%", label: "two-engine disagreement", sub: "lenstronomy vs PyAutoLens on identical systems" },
];

export type Decision = { question: string; options: string; chosen: string; why: string };
export const decisions: Decision[] = [
  {
    question: "How to orchestrate the agent?",
    options: "Write the loop from scratch (v1) · Orchestral AI, the framework HEPTAPOD uses (v2) · Pydantic AI (v3)",
    chosen: "Pydantic AI",
    why: "The run cards and dataset records were already Pydantic models, so tool arguments and structured outputs came for free. It also shipped approval-gated tools natively, which collapsed the planned two-agent propose/approve/execute split into one agent with one gated tool.",
  },
  {
    question: "Where does the compute and data live?",
    options: "Laptop · RunPod or Lambda VMs · Modal",
    chosen: "Modal functions and a Modal Volume",
    why: "The pipeline is function-shaped, so each tool became a remote function behind the same submit/status interface as the local backend. Batches shard across containers, training gets an L4 on demand, and nothing large ever touches the laptop, which had 300 MB free the day the project started.",
  },
  {
    question: "Should the images go in a vector database?",
    options: "Pinecone · object storage · the Modal Volume",
    chosen: "The Modal Volume",
    why: "A vector database stores embeddings for similarity search; it cannot hold float32 image arrays. The volume mounts straight into the GPU containers where training reads it.",
  },
  {
    question: "Which LLM drives the agent?",
    options: "Anthropic · OpenAI · an open model on Fireworks",
    chosen: "Kimi K3 on Fireworks (any Pydantic AI provider works)",
    why: "The provider is a model string. Kimi K3 handled structured tool calls well enough to pass the benchmark from scratch, which made the tool-vs-no-tool comparison more interesting than a weaker model would have.",
  },
  {
    question: "Port or redesign the ML models?",
    options: "Re-derive architectures · port them from the two papers",
    chosen: "Port them layer for layer",
    why: "The lab's 2019 and 2021 papers already specified and benchmarked ResNet-18, AlexNet and the DCAE/VAE/AAE stack. The novelty of this project is the loop around them, not the models.",
  },
  {
    question: "Human-in-the-loop: framework hook or own code?",
    options: "Framework pre-execution hooks · two agents with a human between them · one agent with gated tools",
    chosen: "One agent, gated tools, approval loop in our own code",
    why: "The approval boundary is a plain callback that returns approve, approve-with-edits, or deny. It is unit-tested without an LLM and the same callback drives the terminal prompt, the auto-approver for benchmarks, and the loop driver.",
  },
];

export type Phase = {
  n: number;
  title: string;
  when: string;
  delivered: string;
  broke: { what: string; fix: string }[];
};
export const phases: Phase[] = [
  {
    n: 1,
    title: "Data model and simulator",
    when: "Sep 8",
    delivered:
      "LensCard (the run card a human approves), lensjsonl (one JSON record per image with cheap statistics the agent can reason over), and a port of DeepLenseSim's lenstronomy wrapper driven by a card instead of hard-coded constants. About 50 ms per image on a laptop.",
    broke: [
      { what: "DeepLenseSim had four bugs: Einstein radii from luminosity instead of angular-diameter distances, a Poisson subhalo count drawn and then ignored, the Euclid image adding the model to itself, and a mass fraction that meant different things per class.", fix: "Fixed and documented rather than copied. The correct Einstein radius for the default lens is 1.66 arcsec, not 1.35." },
      { what: "A 32-pixel image at 0.05 arcsec per pixel silently produced pure noise: the arcs sat outside the field of view.", fix: "A guard rejects any card whose field is narrower than the Einstein radius, as a structured failure the agent can read." },
      { what: "lenstronomy had renamed its keyword arguments since DeepLenseSim was written.", fix: "Two renames, found by inspecting the installed signatures instead of trusting the old code." },
    ],
  },
  {
    n: 2,
    title: "Agent, approval gate, compute backends",
    when: "Sep 9",
    delivered:
      "One Pydantic AI agent with simulation tools, an approval loop that pauses on expensive calls, and a ComputeBackend interface with a local implementation and a Modal implementation. Batches shard 500 images per container; per-image seeding makes a sharded remote run bit-identical to a local one.",
    broke: [
      { what: "Pydantic AI could not resolve a tool's dependency type that was imported only for type checking.", fix: "The dependency class moved to its own module and is imported for real." },
      { what: "The shell running the automation was zsh, which does not word-split variables the way bash does.", fix: "Six simulation runs got empty arguments before that was noticed. Explicit commands from then on." },
    ],
  },
  {
    n: 3,
    title: "ML tier on Modal GPUs",
    when: "Sep 9",
    delivered:
      "ResNet-18 and AlexNet with a single-channel stem, and the DCAE, VAE and AAE encoder/decoder/discriminator from the 2021 paper's appendix. Training and evaluation as remote functions; torch exists only in the Modal image.",
    broke: [
      { what: "The 2021 autoencoders only close at 150 by 150 pixels; 64-pixel Euclid images produce a zero-size feature map.", fix: "Images are resized on load, and the flattened size and decoder paddings are derived from the input size so 96 and 204 also work and 64 fails loudly." },
      { what: "The first 15k-image training run died of container memory exhaustion while preloading.", fix: "16 GB per training container." },
      { what: "A 44-second training job took 23 minutes because it read 15,000 files one at a time from a network volume.", fix: "Threaded preloading. Same job, 44 seconds." },
    ],
  },
  {
    n: 4,
    title: "The uncertainty loop",
    when: "Sep 9",
    delivered:
      "sample_uncertainty joins a model's per-image scores with the simulation records, bins each class along physical axes (mass fraction, SNR, axion mass, subhalo count), ranks the cells, and proposes the next LensCard. A loop driver runs uncertainty → approve → simulate → retrain → evaluate without an LLM so the loop is testable offline.",
    broke: [
      { what: "The first real round made the model worse: accuracy 0.92 to 0.43.", fix: "The training code kept the last epoch's weights, and validation accuracy swung between 0.61 and 0.93 from epoch to epoch. Best-validation checkpointing and a cosine learning-rate schedule made rounds comparable." },
      { what: "Float noise in mass fractions produced duplicate degenerate cells.", fix: "Values are rounded before binning and single-valued axes are skipped." },
    ],
  },
  {
    n: 5,
    title: "PyAutoLens cross-check",
    when: "Sep 15",
    delivered:
      "A second engine renders the identical lens systems. The convention mapping between the two packages was found empirically by minimising the residual over every orientation and sign combination, then frozen in code.",
    broke: [
      { what: "The first comparison was off by a factor of 94.", fix: "One engine returns surface brightness, the other counts per pixel: the pixel area." },
      { what: "A 'peak shift' metric reported 66-pixel offsets.", fix: "On an Einstein ring the brightest pixel is ambiguous between two sides. Flux-weighted centroids agree to a twentieth of a pixel." },
      { what: "PyAutoLens pins an older scikit-learn.", fix: "It lives in its own virtual environment and its own Modal image." },
    ],
  },
  {
    n: 6,
    title: "Benchmark and report generator",
    when: "Sep 9 to 15",
    delivered:
      "A HEPTAPOD-style benchmark: the same model and prompt, one arm with the domain tools and one with only a sandboxed shell and file tools, graded by a cumulative four-stage rubric read off the sandbox. Plus a report generator that produced the figures on this page.",
    broke: [
      { what: "The last rubric stage thresholded AUC, and a ResNet-18 trained from scratch on under 200 images sits at chance whatever the settings.", fix: "The stage now checks that a held-out evaluation exists and that the number the agent reports matches the file. Correct reporting, not luck." },
      { what: "The sandbox path check raised on macOS because temp directories resolve through a /private symlink.", fix: "Resolve the workspace once." },
    ],
  },
];

export const loopTable = [
  { model: "Base", data: "15,000", acc: "0.928", auc: "0.979", none: "1.00", sub: "0.96", vor: "0.82", weak: "0.29" },
  { model: "+ 2,000 uniform", data: "17,000", acc: "0.930", auc: "0.978", none: "1.00", sub: "0.97", vor: "0.82", weak: "0.28" },
  { model: "+ 2,000 targeted", data: "17,000", acc: "0.900", auc: "0.980", none: "1.00", sub: "0.79", vor: "0.91", weak: "0.62" },
];

export const benchTable = [
  { arm: "tool", pass: "3 / 3", req: "6", tok: "36k", time: "103 s", auc: "0.61, 0.61, 0.61" },
  { arm: "core", pass: "3 / 3", req: "30", tok: "857k", time: "894 s", auc: "0.85, 0.96, 0.96" },
];

export const crossTable = [
  { cls: "none", rms: "0.0063 / 0.0099", flux: "0.996", centroid: "≤ 0.05 px" },
  { cls: "subhalo", rms: "0.0074 / 0.0118", flux: "0.996", centroid: "≤ 0.04 px" },
  { cls: "vortex", rms: "0.0057 / 0.0065", flux: "0.996", centroid: "≤ 0.04 px" },
];

export const videos = [
  { file: "agent-approve.mp4", title: "An agent session, approve path", caption: "" },
  { file: "agent-edit-deny.mp4", title: "Edit and deny", caption: "" },
  { file: "loop-modal.mp4", title: "One active-learning round", caption: "" },
  { file: "benchmark.mp4", title: "Tool arm vs core arm", caption: "" },
  { file: "report.mp4", title: "Generating the report", caption: "" },
];

export const reproduce = [
  { cmd: "git clone https://github.com/amanibobo/lensgrav && cd lensgrav && uv sync", note: "core dependencies; torch is never installed on the laptop" },
  { cmd: "uv run modal setup && uv run modal deploy -m lenscraft.compute.modal_app", note: "once; builds the simulation, GPU and PyAutoLens images" },
  { cmd: "uv run lenscraft simulate --backend modal --substructure vortex --n 5000 --seed 0 --run-id vor5k", note: "18k images rendered in about a minute across shards" },
  { cmd: "uv run lenscraft train --backend modal --kind classifier --arch resnet18 --model-id clf --runs none5k sub5k vor5k", note: "about 2 minutes on an L4" },
  { cmd: "uv run lenscraft loop --backend modal --model-id clf --train-runs none5k sub5k vor5k --test-runs none1k sub1k vor1k --rounds 1", note: "uncertainty → approve → simulate → retrain → evaluate" },
  { cmd: "uv run lenscraft bench --trials 10 --arms tool core", note: "needs an LLM key in .env; the core arm is slow" },
];
