/* Inline SVG diagrams. Colours come from CSS variables so they follow the theme. */

const ink = "var(--ink)";
const ink2 = "var(--ink-2)";
const line = "var(--line)";
const surface = "var(--surface)";
const accent = "var(--accent)";
const font = "var(--font-geist-sans), system-ui, sans-serif";
const mono = "var(--font-geist-mono), ui-monospace, monospace";

function Box({ x, y, w, h, title, sub, strong = false }: { x: number; y: number; w: number; h: number; title: string; sub?: string; strong?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={surface} stroke={strong ? accent : line} strokeWidth={strong ? 2 : 1.5} />
      <text x={x + w / 2} y={y + (sub ? h / 2 - 4 : h / 2 + 5)} textAnchor="middle" fontFamily={font} fontSize={14} fontWeight={600} fill={ink}>
        {title}
      </text>
      {sub ? (
        <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontFamily={mono} fontSize={11} fill={ink2}>
          {sub}
        </text>
      ) : null}
    </g>
  );
}

function Arrow({ d, label, lx, ly }: { d: string; label?: string; lx?: number; ly?: number }) {
  return (
    <g>
      <path d={d} fill="none" stroke={ink2} strokeWidth={1.5} markerEnd="url(#arrow)" />
      {label ? (
        <text x={lx} y={ly} textAnchor="middle" fontFamily={mono} fontSize={11} fill={ink2}>
          {label}
        </text>
      ) : null}
    </g>
  );
}

function Defs() {
  return (
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill={ink2} />
      </marker>
    </defs>
  );
}

export function LoopDiagram() {
  return (
    <svg viewBox="0 0 760 300" role="img" aria-label="The closed loop: simulate, train, find weak regions, simulate there" className="w-full max-w-3xl">
      <Defs />
      <Box x={20} y={40} w={170} h={64} title="Simulate" sub="LensCard → lensjsonl" strong />
      <Box x={295} y={40} w={170} h={64} title="Train" sub="ResNet-18 · AAE" />
      <Box x={570} y={40} w={170} h={64} title="Evaluate" sub="held-out runs" />
      <Box x={295} y={200} w={170} h={64} title="Find weak region" sub="sample_uncertainty" strong />
      <Arrow d="M 190 72 H 293" />
      <Arrow d="M 465 72 H 568" />
      <Arrow d="M 655 104 V 232 H 467" label="per-image scores" lx={560} ly={252} />
      <Arrow d="M 295 232 H 105 V 106" label="next LensCard" lx={200} ly={252} />
      <text x={105} y={170} textAnchor="middle" fontFamily={mono} fontSize={11} fill={accent}>
        human approves
      </text>
      <text x={380} y={150} textAnchor="middle" fontFamily={font} fontSize={12} fill={ink2}>
        same held-out runs every round
      </text>
    </svg>
  );
}

export function ArchitectureDiagram() {
  return (
    <svg viewBox="0 0 860 420" role="img" aria-label="System architecture" className="w-full">
      <Defs />
      <Box x={20} y={30} w={190} h={60} title="User" sub="natural language" />
      <Box x={20} y={150} w={190} h={60} title="Approval gate" sub="approve · edit · deny" strong />
      <Box x={280} y={30} w={230} h={60} title="Agent" sub="Pydantic AI · any provider" />
      <rect x={280} y={130} width={230} height={190} rx={10} fill="none" stroke={line} strokeDasharray="4 4" />
      <text x={395} y={150} textAnchor="middle" fontFamily={mono} fontSize={11} fill={ink2}>
        tool tiers
      </text>
      <Box x={295} y={160} w={200} h={44} title="simulation" sub="simulate · cross_check" />
      <Box x={295} y={212} w={200} h={44} title="ML" sub="train · evaluate · uncertainty" />
      <Box x={295} y={264} w={200} h={44} title="docs" sub="summarize · report" />
      <Box x={590} y={130} w={250} h={60} title="ComputeBackend" sub="submit · status · wait" strong />
      <Box x={590} y={230} w={115} h={60} title="Local" sub="in-process" />
      <Box x={725} y={230} w={115} h={60} title="Modal" sub="shards · L4 GPU" />
      <Box x={590} y={330} w={250} h={60} title="Volume" sub="runs · models · scores" />
      <Arrow d="M 210 60 H 278" />
      <Arrow d="M 395 90 V 128" />
      <Arrow d="M 280 60 L 210 150" label="gated calls pause" lx={160} ly={118} />
      <Arrow d="M 510 226 H 588 V 192" label="every tool" lx={560} ly={215} />
      <Arrow d="M 650 190 V 228" />
      <Arrow d="M 780 190 V 228" />
      <Arrow d="M 780 290 V 328" />
      <Arrow d="M 650 290 V 328" />
    </svg>
  );
}

export function DataFlowDiagram() {
  const items = [
    ["LensCard", "class · mass · seed"],
    ["lensjsonl", "label · snr · stats"],
    ["checkpoint", "best-val epoch"],
    ["scores", "P(class) / recon err"],
    ["uncertainty", "weakest cell → card"],
  ];
  return (
    <svg viewBox="0 0 860 130" role="img" aria-label="Data flow from LensCard to the next LensCard" className="w-full">
      <Defs />
      {items.map(([t, s], i) => (
        <g key={t}>
          <Box x={10 + i * 170} y={20} w={150} h={64} title={t} sub={s} strong={i === 0 || i === 4} />
          {i < items.length - 1 ? <Arrow d={`M ${160 + i * 170} 52 H ${178 + i * 170}`} /> : null}
        </g>
      ))}
      <Arrow d="M 765 84 V 108 H 85 V 86" label="closes the loop" lx={425} ly={122} />
    </svg>
  );
}

export function ApprovalDiagram() {
  return (
    <svg viewBox="0 0 760 240" role="img" aria-label="How the approval gate works" className="w-full max-w-3xl">
      <Defs />
      <Box x={20} y={20} w={200} h={56} title="model proposes" sub="simulate_lens_batch(card)" />
      <Box x={280} y={20} w={200} h={56} title="run pauses" sub="DeferredToolRequests" strong />
      <Box x={540} y={20} w={200} h={56} title="human decides" sub="y · e · n" />
      <Box x={20} y={130} w={200} h={56} title="tool runs" sub="approved or edited card" />
      <Box x={280} y={130} w={200} h={56} title="model sees denial" sub="with the reason, no retry" />
      <Box x={540} y={130} w={200} h={56} title="run resumes" sub="DeferredToolResults" strong />
      <Arrow d="M 220 48 H 278" />
      <Arrow d="M 480 48 H 538" />
      <Arrow d="M 640 76 V 128" />
      <Arrow d="M 540 150 H 482" label="denied" lx={511} ly={143} />
      <Arrow d="M 600 186 V 214 H 120 V 188" label="approved" lx={360} ly={230} />
    </svg>
  );
}
