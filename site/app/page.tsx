import { Nav } from "@/components/nav";
import { VideoSlot } from "@/components/video-slot";
import { ApprovalDiagram, ArchitectureDiagram, DataFlowDiagram, LoopDiagram } from "@/components/diagrams";
import { ClassDot, Code, Details, Figure, Prose, Section, Table } from "@/components/ui";
import { benchTable, crossTable, decisions, loopTable, phases, REPO, reproduce, stats, videos } from "@/content/site";

export default function Page() {
  return (
    <>
      <a id="top" />
      <Nav />
      <main className="flex-1">
        {/* ---------------------------------------------------------------- hero */}
        <section className="mx-auto w-full max-w-5xl px-5 pb-16 pt-16 sm:px-8 sm:pt-24">
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-muted">Case study · September 2026</p>
          <h1 className="mt-4 max-w-4xl text-4xl font-semibold leading-[1.08] tracking-tight text-ink sm:text-6xl">
            An agent that decides what to simulate next.
          </h1>
          <p className="prose mt-6 text-xl text-ink-2">
            LensCraft simulates strong gravitational lenses, trains dark-matter substructure detectors on them, and uses what those detectors
            can&apos;t classify to choose the next batch of simulations. Built on Pydantic AI, Lenstronomy and Modal, with a human approving every
            expensive step.
          </p>
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {stats.map((s) => (
              <div key={s.label} className="rounded-lg border border-line bg-surface p-5">
                <div className="text-3xl font-semibold tracking-tight text-ink">{s.value}</div>
                <div className="mt-1 text-sm font-medium text-ink">{s.label}</div>
                <div className="mt-1 text-xs leading-relaxed text-ink-2">{s.sub}</div>
              </div>
            ))}
          </div>
          <Figure
            wide
            src="/figures/hero_classes.png"
            alt="Three simulated Einstein rings, one per substructure class, with the perturbation each substructure adds to the arcs"
            caption="The three classes the models learn to tell apart. Top: the observed image. Bottom: what the substructure adds to the arcs, once the smooth lens is subtracted. CDM subhalos leave a local dent; an axion vortex leaves a coherent distortion along a line."
          />
        </section>

        {/* ------------------------------------------------------------- problem */}
        <Section
          id="problem"
          kicker="01 · The problem"
          title="Dark matter leaves fingerprints on lensed galaxies. Finding them needs data nobody has."
          lede="A massive galaxy bends the light of one behind it into arcs. Small clumps of dark matter in the lens perturb those arcs by about one percent. What the clumps look like depends on what dark matter is."
        >
          <Prose>
            <p>
              Cold dark matter predicts point-like subhalos. Ultralight axion dark matter predicts vortex lines. Both are subtle, and real
              strong lenses are rare, so the only way to train a detector is to simulate: thousands of images per hypothesis, across every
              plausible halo mass, redshift, mass fraction and axion mass.
            </p>
            <p>
              Two papers from the same group established the machine-learning side. Alexander et al. (2019) classified simulated lenses into
              no-substructure, subhalo and vortex classes with ResNet-18 and AlexNet. Alexander et al. (2021) did it without labels, using
              autoencoders trained on smooth lenses to flag anything they could not reconstruct. Both spent most of their effort on
              generating and curating simulations by hand.
            </p>
            <p>
              HEPTAPOD (Menzo et al., 2026) showed the pattern for high-energy physics: wrap the simulation engines in schema-validated
              tools, coordinate them with run cards a human signs off on, and let an agent do the orchestration. This project applies that
              pattern to lensing and adds the thing the pattern was missing: a feedback signal.
            </p>
          </Prose>
        </Section>

        {/* ---------------------------------------------------------------- idea */}
        <Section
          id="idea"
          kicker="02 · The idea"
          title="Close the loop."
          lede="Train on what you have, find where the model is uncertain, simulate more of exactly that, retrain. Repeat with a human approving each batch."
        >
          <LoopDiagram />
          <Prose>
            <p>
              The claim the project tests is simple: a batch of simulations chosen by where the current model is weak should help more than
              a batch of the same size chosen uniformly. If it does, the agent is doing planning under a real feedback signal rather than
              running a script.
            </p>
            <p>
              Everything else, the orchestration framework, the compute backend, the ported models, exists to make that comparison cheap
              enough to run repeatedly.
            </p>
          </Prose>
        </Section>

        {/* ----------------------------------------------------------- decisions */}
        <Section
          id="decisions"
          kicker="03 · Ideation"
          title="Six decisions that shaped the build."
          lede="The design went through three revisions before any code existed. These are the forks that mattered, with what was considered and why each path was taken."
        >
          <ol className="grid gap-4">
            {decisions.map((d, i) => (
              <li key={d.question} className="rounded-lg border border-line bg-surface p-5 sm:p-6">
                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-xs text-muted">{String(i + 1).padStart(2, "0")}</span>
                  <h3 className="text-lg font-semibold text-ink">{d.question}</h3>
                </div>
                <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-[7rem_1fr]">
                  <dt className="text-muted">Considered</dt>
                  <dd className="text-ink-2">{d.options}</dd>
                  <dt className="text-muted">Chosen</dt>
                  <dd className="font-medium text-ink">{d.chosen}</dd>
                  <dt className="text-muted">Why</dt>
                  <dd className="leading-relaxed text-ink-2">{d.why}</dd>
                </dl>
              </li>
            ))}
          </ol>
        </Section>

        {/* -------------------------------------------------------- architecture */}
        <Section
          id="architecture"
          kicker="04 · Architecture"
          title="One agent, three tool tiers, one backend interface."
          lede="Tools never talk to the simulator or to PyTorch directly. They talk to a ComputeBackend, so a command-line flag moves the whole pipeline from the laptop to Modal."
        >
          <ArchitectureDiagram />
          <h3 className="mt-14 text-xl font-semibold text-ink">What flows through it</h3>
          <p className="prose mt-2 text-ink-2">
            The agent never sees pixels. It reasons over one JSON record per image with cheap statistics computed at generation time, and
            over per-image model scores produced at evaluation time. That is what makes the uncertainty step a pure-numpy function rather
            than another model.
          </p>
          <div className="mt-6">
            <DataFlowDiagram />
          </div>
          <h3 className="mt-14 text-xl font-semibold text-ink">The approval gate</h3>
          <p className="prose mt-2 text-ink-2">
            Simulation batches and training jobs cost money, so those three tools require approval. Pydantic AI pauses the run and returns
            the pending calls; a callback in our own code decides; the run resumes. The same callback drives the terminal prompt, the
            auto-approver used by benchmarks, and the loop driver.
          </p>
          <div className="mt-6">
            <ApprovalDiagram />
          </div>
          <Details summary="What a LensCard looks like">
            <Code>{`{
  "n_images": 2000,
  "substructure": "vortex",
  "halo_mass": 1e12,
  "redshift_lens": 0.5,
  "redshift_source": 1.0,
  "substructure_mass_fraction": 0.03,
  "axion_mass": 1.69e-24,
  "instrument": "euclid",
  "image_size": 64,
  "seed": 1001
}`}</Code>
            <p>
              A frozen Pydantic model with physics validation: the source must sit behind the lens, the mass fraction must be positive for
              substructure classes, and a card whose field of view cannot contain the Einstein radius is rejected before any pixel is
              rendered.
            </p>
          </Details>
        </Section>

        {/* --------------------------------------------------------------- build */}
        <Section
          id="build"
          kicker="05 · The build"
          title="Six phases, and what broke in each."
          lede="Built in order over a week, each phase tested and verified with a real run before the next started. The things that broke are the useful part."
        >
          <ol className="relative grid gap-8 border-l border-line pl-6 sm:pl-8">
            {phases.map((p) => (
              <li key={p.n} className="relative">
                <span className="absolute -left-[1.95rem] top-1.5 flex h-6 w-6 items-center justify-center rounded-full border border-line bg-surface font-mono text-[11px] text-ink sm:-left-[2.45rem]">
                  {p.n}
                </span>
                <div className="flex flex-wrap items-baseline gap-x-3">
                  <h3 className="text-xl font-semibold text-ink">{p.title}</h3>
                  <span className="font-mono text-xs text-muted">{p.when}</span>
                </div>
                <p className="prose mt-2 text-ink-2">{p.delivered}</p>
                <Details summary={`What broke (${p.broke.length})`}>
                  <ul className="!mt-0">
                    {p.broke.map((b) => (
                      <li key={b.what}>
                        <span className="text-ink">{b.what}</span> <span className="text-ink-2">{b.fix}</span>
                      </li>
                    ))}
                  </ul>
                </Details>
              </li>
            ))}
          </ol>
        </Section>

        {/* ------------------------------------------------------------- results */}
        <Section id="results" kicker="06 · Results" title="What the numbers say." lede="All evaluations are on 3,000 held-out images (1,000 per class) that no model ever trained on.">
          <h3 className="text-xl font-semibold text-ink">The dataset</h3>
          <p className="prose mt-2 text-ink-2">
            5,000 training and 1,000 test images per class at 64 px in a Euclid-like setup, plus the batches added by the loop. 18,000 images
            rendered in about a minute of wall time, sharded across Modal containers.
          </p>
          <Figure wide src="/figures/dataset_distributions.png" alt="SNR and substructure residual distributions per class" caption="Signal-to-noise is matched across classes by construction. The substructure perturbs the arcs by two to three percent of peak brightness at mass fraction 0.03." />
          <Figure wide src="/figures/sample_images.png" alt="A grid of sample images per class" caption="Six samples per class, square-root stretched." />

          <h3 className="mt-14 text-xl font-semibold text-ink">The classifier</h3>
          <p className="prose mt-2 text-ink-2">
            ResNet-18 with a single-channel stem, trained from scratch for ten epochs on the 15,000 training images. It is essentially
            perfect on smooth lenses and misses about one vortex in five, and the uncertainty analysis says exactly which ones.
          </p>
          <Table
            head={["true \\ predicted", "none", "subhalo", "vortex", "accuracy"]}
            rows={[
              [<ClassDot key="n" cls="none" />, "1000", "0", "0", "1.000"],
              [<ClassDot key="s" cls="subhalo" />, "2", "963", "35", "0.963"],
              [<ClassDot key="v" cls="vortex" />, "0", "178", "822", "0.822"],
            ]}
            className="max-w-2xl"
          />
          <Figure
            src="/figures/uncertainty_clf_base_b.png"
            alt="Bar chart of mean uncertainty per region of parameter space"
            caption="Where the base classifier is weakest. The lightest axions produce vortex lines two to three arcseconds long; the same mass spread along a longer line perturbs the arcs less. Accuracy in that cell: 29%."
          />

          <h3 className="mt-14 text-xl font-semibold text-ink">The loop, with a control</h3>
          <p className="prose mt-2 text-ink-2">
            The agent simulated 2,000 more vortex images at the axion mass of the weakest cell and retrained under an identical recipe. As a
            control, 2,000 images drawn uniformly across the three classes were added instead.
          </p>
          <Table
            head={["training data", "images", "accuracy", "macro AUC", "none", "subhalo", "vortex", "weak cell"]}
            rows={loopTable.map((r) => [r.model, r.data, r.acc, r.auc, r.none, r.sub, r.vor, r.weak])}
          />
          <Prose>
            <p>
              The uniform batch is indistinguishable from adding nothing. The targeted batch doubles accuracy in the weak region and lifts
              vortex accuracy from 0.82 to 0.91, at the cost of subhalos, which the added light-axion vortices most resemble and which are
              now outnumbered. A second round would flag subhalos as weakest and propose the balancing batch. That is the loop working as
              intended: the uncertainty signal finds a region uniform sampling cannot.
            </p>
          </Prose>

          <h3 className="mt-14 text-xl font-semibold text-ink">Tools versus no tools</h3>
          <p className="prose mt-2 text-ink-2">
            Same model, same prompt, same sandbox. One arm has the domain tools; the other has only a shell and file tools, Python with
            lenstronomy and torch on the path, and the required output layout spelled out. Graded by a cumulative rubric read off the
            sandbox: simulate, label, train on training data only, report the held-out AUC correctly.
          </p>
          <Table head={["arm", "pass", "mean requests", "mean tokens", "mean time", "AUC reached per trial"]} rows={benchTable.map((r) => [r.arm, r.pass, r.req, r.tok, r.time, r.auc])} className="max-w-3xl" />
          <Figure wide src="/figures/bench_cost.png" alt="Bar charts of requests, tokens and wall time per arm" caption="Mean cost per trial. Kimi K3 completes the task either way; the tools change what it costs and whether the result is reproducible." />
          <Prose>
            <p>
              A strong model builds the whole pipeline from scratch and even reaches a higher AUC by engineering its own preprocessing, an
              asinh stretch, dihedral augmentation and a small-image ResNet stem. The domain tools trade that ceiling for 24 times fewer
              tokens, nine times less wall time, and an identical result on every trial.
            </p>
          </Prose>

          <h3 className="mt-14 text-xl font-semibold text-ink">Two engines, one answer</h3>
          <p className="prose mt-2 text-ink-2">
            The same lens systems rendered through PyAutoLens agree with the lenstronomy port to about one percent of peak brightness. The
            residual is the packages&apos; slightly different elliptical-radius conventions, and the point-mass substructure agrees as well as
            the smooth lens does.
          </p>
          <Table head={["class", "shape residual, mean / max", "flux ratio", "centroid shift"]} rows={crossTable.map((r) => [<ClassDot key={r.cls} cls={r.cls as "none" | "subhalo" | "vortex"} />, r.rms, r.flux, r.centroid])} className="max-w-2xl" />
        </Section>

        {/* --------------------------------------------------------------- demos */}
        <Section id="demos" kicker="07 · Demos" title="See it run." lede="Short recordings of the product. Each under two minutes.">
          <div className="grid gap-8 md:grid-cols-2">
            {videos.map((v) => (
              <VideoSlot key={v.file} {...v} />
            ))}
          </div>
        </Section>

        {/* -------------------------------------------------------------- limits */}
        <Section id="limits" kicker="08 · Limits" title="What this does not show yet." lede="The results above are real and reproducible. They are also early, and these are the caveats that matter.">
          <Prose>
            <ul>
              <li>
                <strong>Three trials per arm.</strong> The pass@k estimators are exact for the sample, but the sample is small. Ten or more
                trials and a second model are the obvious next benchmark.
              </li>
              <li>
                <strong>The anomaly detector is at chance.</strong> The AAE reconstructs well but separates poorly, AUC 0.54, far from the
                2021 paper&apos;s 0.93. At this resolution and mass fraction the substructure signal is one to three percent of peak, and ten
                epochs is a fraction of what the paper trained for. Longer training, larger mass fractions, and the 150-pixel instrument are
                the levers.
              </li>
              <li>
                <strong>Targeting has a cost.</strong> The loop&apos;s gain in the weak region came with a subhalo regression from class imbalance.
                Class-balanced batches or re-weighting is the natural fix and the natural second round.
              </li>
              <li>
                <strong>The core arm out-engineered the fixed recipe.</strong> Its asinh stretch and augmentation are data-pipeline choices the
                ML tier should adopt. Its architecture change stays out, because the port is meant to match the papers.
              </li>
            </ul>
          </Prose>
        </Section>

        {/* ----------------------------------------------------------- reproduce */}
        <Section id="reproduce" kicker="09 · Reproduce it" title="Clone to trained model in five commands." lede="Everything on this page came from the repository, a Modal account, and an LLM key.">
          <ol className="grid gap-3">
            {reproduce.map((r) => (
              <li key={r.cmd}>
                <Code>{r.cmd}</Code>
                <p className="-mt-2 text-sm text-ink-2">{r.note}</p>
              </li>
            ))}
          </ol>
          <div className="mt-10">
            <Prose>
              <p>
                Source, design document, tests and the generated report are at{" "}
              <a href={REPO} target="_blank" rel="noreferrer">
                github.com/amanibobo/lensgrav
              </a>
              . References: Menzo et al., <em>HEPTAPOD</em> (2026); Alexander et al., <em>Deep Learning the Morphology of Dark Matter
              Substructure</em> (2019); Alexander et al., <em>Decoding Dark Matter Substructure without Supervision</em> (2021).
              </p>
            </Prose>
          </div>
        </Section>
      </main>
      <footer className="border-t border-line py-10">
        <div className="mx-auto max-w-5xl px-5 text-sm text-ink-2 sm:px-8">LensCraft · built September 2026 · figures generated by the project&apos;s own report tool.</div>
      </footer>
    </>
  );
}
