import { Embed } from "@/components/embed";
import { LensAnimation } from "@/components/lens-animation";
import { MediaSlot } from "@/components/media-slot";
import { Rail } from "@/components/nav";
import { Stack } from "@/components/stack";
import { VideoSlot } from "@/components/video-slot";
import { ClassDot, Code, Figure, Meta, Prose, Section, Sub, Table } from "@/components/ui";
import { benchTable, crossTable, embeds, loopTable, papers, phases, REPO, reproduce, videos } from "@/content/site";

export default function Page() {
  return (
    <>
      <a id="top" />
      <Rail />
      <main className="flex-1">
        {/* ------------------------------------------------------------- hero */}
        <section className="mx-auto w-full max-w-195 px-6 pb-6 pt-20 sm:pt-28">
          <h1 className="text-[1.85rem] font-semibold leading-[1.2] tracking-[-0.02em] text-ink sm:text-[2.1rem]">
            Dark Matter by Feedback: Uncertainty-Driven Simulation for Strong-Lensing Detectors
          </h1>
          <p className="mt-5 text-[1.05rem] leading-snug text-ink-2 sm:text-[1.1rem]">
            An agent that simulates gravitational lenses, trains dark-matter detectors on them, and uses what those detectors get wrong to choose the next batch of simulations
          </p>
          <div className="mt-8">
            <LensAnimation />
          </div>
          <div className="mt-10">
            <Stack />
          </div>
        </section>

        {/* ---------------------------------------------------------- overview */}
        <Section id="overview" kicker="Overview" title="Closing the loop between simulation and learning">
          <Prose>
            <p>
              LensCraft is an agent that plans and runs strong gravitational lensing simulation batches, trains classifiers and anomaly
              detectors on the results, and uses what those models are uncertain about to decide what to simulate next. A human approves
              every batch before it runs.
            </p>
            <p>
              It was built in six phases over a week on Pydantic AI, Lenstronomy and Modal, with the machine-learning architectures ported
              from two papers by the same research group.
            </p>
          </Prose>
          <div className="mt-10">
            <Meta
              items={[
                { label: "Accuracy", lines: ["0.93 held-out", "ResNet-18, 15k images"] },
                { label: "Weak region", lines: ["0.29 → 0.62", "after one round"] },
                { label: "Tool cost", lines: ["24× fewer tokens", "than no tools"] },
                { label: "Engine check", lines: ["≤ 1.2% disagreement", "vs PyAutoLens"] },
              ]}
            />
          </div>
        </Section>

        {/* ----------------------------------------------------------- problem */}
        <Section id="problem" kicker="Problem" title="Dark matter leaves fingerprints on lensed galaxies, and finding them needs data nobody has">
          <Prose>
            <p>
              A massive galaxy bends the light of one behind it into arcs. Small clumps of dark matter in the lens perturb those arcs by
              about one percent, and what the clumps look like depends on what dark matter is. Cold dark matter predicts point-like
              subhalos; ultralight axion dark matter predicts vortex lines.
            </p>
            <p>
              Both signatures are subtle and real strong lenses are rare, so the only way to train a detector is to simulate: thousands of
              images per hypothesis, across every plausible halo mass, redshift, mass fraction and axion mass.{" "}
              <a href={papers.morphology.href} target="_blank" rel="noreferrer">
                Alexander et al. (2019)
              </a>{" "}
              classified such simulations with ResNet-18 and AlexNet;{" "}
              <a href={papers.decoding.href} target="_blank" rel="noreferrer">
                Alexander et al. (2021)
              </a>{" "}
              did it without labels using autoencoders. Both spent most of their effort generating and curating simulations by hand.
            </p>
            <p>
              <a href={papers.heptapod.href} target="_blank" rel="noreferrer">
                HEPTAPOD
              </a>{" "}
              (Menzo et al., 2026) showed the pattern for high-energy physics: wrap the simulation engines in schema-validated
              tools, coordinate them with run cards a human signs off on, and let an agent orchestrate. This project applies that pattern
              to lensing and adds the piece it was missing, a feedback signal.
            </p>
          </Prose>
        </Section>

        {/* -------------------------------------------------------------- idea */}
        <Section id="idea" kicker="Idea" title="The idea">
          <Prose>
            <p>
              The claim the project tests is simple: a batch of simulations chosen by where the current model is uncertain should help more
              than a batch of the same size chosen uniformly. If it does, the agent is planning under a real feedback signal rather than
              running a script. Everything else exists to make that comparison cheap enough to run repeatedly.
            </p>
          </Prose>
          <MediaSlot file="sketch-loop.png" alt="Hand-drawn sketch of the simulate, train, find-weak-region loop" />
          <Sub id="simulator">Try the simulator</Sub>
          <Prose>
            <p>
              The same engine the agent drives. Switch to the vortex class and slide the axion mass down: the vortex line stretches across
              the image and its perturbation of the arcs fades. That region is where the classifier turned out to be weakest.
            </p>
          </Prose>
          <Embed src={embeds.playground} title="Simulate a lens" height={620} />
        </Section>

        {/* --------------------------------------------------------- decisions */}
        <Section id="decisions" title="Key decisions">
          <Prose>
            <p>
              The design went through three versions before any code existed. The first built everything by hand, including the agent
              loop. The second sat on Orchestral AI, the framework HEPTAPOD itself uses. The third, and the one that shipped, uses
              Pydantic AI, mostly because the run cards and dataset records were already Pydantic models, so tool arguments and structured
              outputs came for free. It also turned out to support approval-gated tools natively, which collapsed a planned two-agent
              propose-then-execute design into a single agent with one gated tool and an approval callback in our own code. That callback
              is a plain function returning approve, approve with edits, or deny, so it is unit-tested without an LLM, and the same
              function drives the terminal prompt, the benchmark&apos;s auto-approver, and the loop driver.
            </p>
            <p>
              Compute and data went to Modal rather than a rented VM or the laptop. The
              pipeline is function-shaped, so each tool became a remote function behind the same submit-and-status interface as the local
              backend; batches shard across containers and training gets an L4 on demand. The images live on a Modal volume that mounts
              straight into the GPU containers, after a brief detour considering a vector database. The agent runs on Kimi K3 through Fireworks, though any Pydantic AI provider is a one-line change, and the ML
              models were ported layer for layer from the group&apos;s 2019 and 2021 papers rather than redesigned, since the contribution
              here is the loop around them, not the networks.
            </p>
          </Prose>
          <MediaSlot file="sketch-versions.png" alt="Hand-drawn sketch of how the design changed from v1 to v3" />
        </Section>

        {/* ------------------------------------------------------ architecture */}
        <Section id="architecture" kicker="Architecture" title="Architecture">
          <Prose>
            <p>
              Tools never talk to the simulator or to PyTorch directly. They talk to a compute backend, so a command-line flag moves the
              whole pipeline from the laptop to Modal.
            </p>
          </Prose>
          <MediaSlot file="sketch-architecture.png" alt="Hand-drawn sketch of the agent, tool tiers, compute backend and Modal volume" />
          <Sub id="data-flow">What flows through it</Sub>
          <Prose>
            <p>
              The agent never sees pixels. It reasons over one JSON record per image with cheap statistics computed at generation time,
              and over per-image model scores produced at evaluation time. That is what makes the uncertainty step a small numpy function
              rather than another model.
            </p>
          </Prose>
          <MediaSlot file="sketch-dataflow.png" alt="Hand-drawn sketch of the data flowing from LensCard to lensjsonl to checkpoint to scores to the next card" />
          <Sub id="approval">The approval gate</Sub>
          <Prose>
            <p>
              Simulation batches and training jobs cost money, so those tools require approval. Pydantic AI pauses the run and returns the
              pending calls; a callback in our own code decides; the run resumes. The same callback drives the terminal prompt, the
              auto-approver used by benchmarks, and the loop driver.
            </p>
          </Prose>
          <MediaSlot file="sketch-approval.png" alt="Hand-drawn sketch of the approval gate: the run pauses, the human answers y, e or n, the run resumes" />
          <MediaSlot kind="screenshot" file="shot-terminal-approval.png" alt="Terminal screenshot of the approval prompt showing a proposed LensCard" />
          <Sub id="run-card">What a run card looks like</Sub>
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
          <Prose>
            <p>
              A frozen Pydantic model with physics validation: the source must sit behind the lens, the mass fraction must be positive for
              substructure classes, and a card whose field of view cannot contain the Einstein radius is rejected before any pixel is
              rendered.
            </p>
          </Prose>
        </Section>

        {/* ------------------------------------------------------------- build */}
        <Section id="build" kicker="Process" title="The build">
          <ol className="mt-4 divide-y divide-line">
            {phases.map((p) => (
              <li key={p.n} className="py-10">
                <h3 className="text-[1.15rem] font-semibold tracking-[-0.01em] text-ink">{p.title}</h3>
                <p className="prose mt-3">{p.delivered}</p>
                {p.n === 2 ? (
                  <MediaSlot kind="screenshot" file="shot-modal-shards.png" alt="Modal dashboard showing simulation shards running in parallel containers" />
                ) : null}
                {p.n === 3 ? (
                  <>
                    <MediaSlot kind="screenshot" file="shot-modal-gpu.png" alt="Modal dashboard showing the L4 training container" />
                    <MediaSlot kind="screenshot" file="shot-modal-volume.png" alt="Modal volume browser showing runs and models" />
                  </>
                ) : null}
                {p.n === 4 ? <MediaSlot file="sketch-uncertainty.png" alt="Hand-drawn sketch of how the weak-cell search bins parameter space" /> : null}
                <p className="kicker mt-5">What broke</p>
                <div className="prose text-[0.95rem]">
                  <ul className="mt-2!">
                    {p.broke.map((b) => (
                      <li key={b.what}>
                        <span className="text-ink">{b.what}</span> {b.fix}
                      </li>
                    ))}
                  </ul>
                </div>
              </li>
            ))}
          </ol>
        </Section>

        {/* ----------------------------------------------------------- results */}
        <Section id="results" kicker="Results" title="What the numbers say">
          <Prose>
            <p>
              The models were trained on 15,000 simulated images. Every number below was measured on a separate set of 3,000 images, 1,000 per
              class, generated with different seeds and kept out of training, so the scores reflect what the models learned rather than what they memorised.
            </p>
          </Prose>

          <Sub id="dataset">The dataset</Sub>
          <Prose>
            <p>
              5,000 training and 1,000 test images per class at 64 pixels in a Euclid-like setup, plus the batches added by the loop.
              18,000 images rendered in about a minute of wall time, sharded across Modal containers.
            </p>
          </Prose>
          <Figure src="/figures/dataset_distributions.png" alt="SNR and substructure residual distributions per class" />
          <Figure src="/figures/sample_images.png" alt="A grid of sample images per class" />

          <Sub id="classifier">The classifier</Sub>
          <Prose>
            <p>
              ResNet-18 with a single-channel stem, trained from scratch for ten epochs on the 15,000 training images. It is essentially
              perfect on smooth lenses and misses about one vortex in five, and the uncertainty analysis says exactly which ones.
            </p>
          </Prose>
          <Table
            head={["True \\ predicted", "none", "subhalo", "vortex", "accuracy"]}
            rows={[
              [<ClassDot key="n" cls="none" />, "1000", "0", "0", "1.000"],
              [<ClassDot key="s" cls="subhalo" />, "2", "963", "35", "0.963"],
              [<ClassDot key="v" cls="vortex" />, "0", "178", "822", "0.822"],
            ]}
            className="max-w-xl"
          />
          <Figure
            src="/figures/uncertainty_clf_base_b.png"
            alt="Bar chart of mean uncertainty per region of parameter space"
           
          />
          <Sub id="explorer">Explore it yourself</Sub>
          <Prose>
            <p>
              The same analysis, live on the 3,000 held-out scores. Pick a class and an axis, change the binning, and click a bar to see the
              most uncertain images in that cell.
            </p>
          </Prose>
          <Embed src={embeds.explorer} title="Where is the classifier weak?" height={760} />

          <Sub id="loop">The loop, with a control</Sub>
          <Prose>
            <p>
              The agent simulated 2,000 more vortex images at the axion mass of the weakest cell and retrained under an identical recipe.
              As a control, 2,000 images drawn uniformly across the three classes were added instead.
            </p>
          </Prose>
          <Table head={["Training data", "images", "accuracy", "macro AUC", "none", "subhalo", "vortex", "weak cell"]} rows={loopTable.map((r) => [r.model, r.data, r.acc, r.auc, r.none, r.sub, r.vor, r.weak])} />
          <Prose>
            <p>
              The uniform batch is indistinguishable from adding nothing. The targeted batch doubles accuracy in the weak region and lifts
              vortex accuracy from 0.82 to 0.91, at the cost of subhalos, which the added light-axion vortices most resemble and which are
              now outnumbered. A second round would flag subhalos as weakest and propose the balancing batch. That is the loop working as
              intended: the uncertainty signal finds a region uniform sampling cannot.
            </p>
          </Prose>

          <Sub id="benchmark">Tools versus no tools</Sub>
          <Prose>
            <p>
              Same model, same prompt, same sandbox. One arm has the domain tools; the other has only a shell and file tools, Python with
              lenstronomy and torch on the path, and the required output layout spelled out. Graded by a cumulative rubric read off the
              sandbox: simulate, label, train on training data only, report the held-out AUC correctly.
            </p>
          </Prose>
          <Table head={["Arm", "pass", "mean requests", "mean tokens", "mean time", "AUC reached per trial"]} rows={benchTable.map((r) => [r.arm, r.pass, r.req, r.tok, r.time, r.auc])} />
          <Embed src={embeds.passk} title="pass@k" height={430} />
          <Figure src="/figures/bench_cost.png" alt="Bar charts of requests, tokens and wall time per arm" caption="Mean cost per trial. Kimi K3 completes the task either way; the tools change what it costs and whether the result is reproducible." />
          <MediaSlot kind="screenshot" file="shot-bench-sandbox.png" alt="The core arm's sandbox directory full of scripts it wrote" />
          <Prose>
            <p>
              A strong model builds the whole pipeline from scratch and even reaches a higher AUC by engineering its own preprocessing: an
              asinh stretch, dihedral augmentation and a small-image ResNet stem. The domain tools trade that ceiling for 24 times fewer
              tokens, nine times less wall time, and an identical result on every trial.
            </p>
          </Prose>

          <Sub id="cross-check">Two engines, one answer</Sub>
          <Prose>
            <p>
              The same lens systems rendered through PyAutoLens agree with the lenstronomy port to about one percent of peak brightness.
              The residual is the packages&apos; slightly different elliptical-radius conventions, and the point-mass substructure agrees as
              well as the smooth lens does.
            </p>
          </Prose>
          <Table head={["Class", "shape residual, mean / max", "flux ratio", "centroid shift"]} rows={crossTable.map((r) => [<ClassDot key={r.cls} cls={r.cls as "none" | "subhalo" | "vortex"} />, r.rms, r.flux, r.centroid])} className="max-w-xl" />
        </Section>

        {/* ------------------------------------------------------------- demos */}
        <Section id="demos" kicker="Demos" title="See it run">
          {videos.map((v) => (
            <VideoSlot key={v.file} {...v} />
          ))}
        </Section>

        {/* ------------------------------------------------------------ limits */}
        <Section id="limits" kicker="Limitations" title="Limitations">
          <Prose>
                        <ul>
              <li>
                <strong>Three trials per arm.</strong> The pass@k estimators are exact for the sample, but the sample is small. Ten or more
                trials and a second model are the obvious next benchmark.
              </li>
              <li>
                <strong>The anomaly detector is at chance.</strong> The AAE reconstructs well but separates poorly, AUC 0.54, far from the
                2021 paper&apos;s 0.93. At this resolution and mass fraction the substructure signal is one to three percent of peak, and ten
                epochs is a fraction of what the paper trained for.
              </li>
              <li>
                <strong>Targeting has a cost.</strong> The loop&apos;s gain in the weak region came with a subhalo regression from class
                imbalance. Class-balanced batches or re-weighting is the natural fix and the natural second round.
              </li>
              <li>
                <strong>The core arm out-engineered the fixed recipe.</strong> Its asinh stretch and augmentation are data-pipeline choices the
                ML tier should adopt. Its architecture change stays out, because the port is meant to match the papers.
              </li>
            </ul>
          </Prose>
        </Section>

        {/* --------------------------------------------------------- reproduce */}
        <Section id="reproduce" kicker="Reproduce" title="Clone to trained model in five commands">
          <Prose>
            <p>Everything on this page came from the repository.</p>
          </Prose>
          <ol className="mt-8 grid gap-5">
            {reproduce.map((r) => (
              <li key={r.cmd} className="min-w-0">
                <Code>{r.cmd}</Code>
                <p className="text-[0.95rem] text-ink-2">{r.note}</p>
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
                .
              </p>
            </Prose>
          </div>
          <Sub id="references">References</Sub>
          <ol className="mt-4 divide-y divide-line border-t border-line">
            {Object.values(papers).map((p) => (
              <li key={p.href} className="py-3.5 text-[0.95rem] leading-relaxed">
                <a href={p.href} target="_blank" rel="noreferrer" className="text-ink underline decoration-line underline-offset-4 hover:decoration-ink">
                  {p.title}
                </a>
                <span className="block text-ink-2">
                  {p.authors}, {p.year}
                </span>
              </li>
            ))}
          </ol>
        </Section>
      </main>
      <footer className="py-12">
        <div className="mx-auto max-w-195 px-6 text-[0.82rem] text-muted"></div>
      </footer>
    </>
  );
}
