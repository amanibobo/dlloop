/* Generates the four hand-drawn diagrams as Excalidraw-style SVGs (rough.js strokes, Excalifont
   handwriting, pastel fills, tool logos) plus matching .excalidraw scene files you can open and
   edit at excalidraw.com.   Run: node scripts/sketches.cjs && node scripts/rasterize.cjs */

const fs = require("node:fs");
const path = require("node:path");
const rough = require("roughjs/bundled/rough.cjs.js");
const si = require("simple-icons");

const OUT = path.join(__dirname, "..", "public", "media");
const FONT = fs.readFileSync(path.join(OUT, "src", "Excalifont.woff2")).toString("base64");

// Excalidraw's default palette
const INK = "#1e1e1e";
const MUTED = "#495057";
const FILL = { blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", orange: "#ffd8a8", violet: "#d0bfff", grey: "#e9ecef", red: "#ffc9c9", teal: "#99e9f2" };
const ICON = { python: si.siPython, pytorch: si.siPytorch, pydantic: si.siPydantic, modal: si.siModal, numpy: si.siNumpy };

const gen = rough.generator();
const opts = (extra = {}) => ({ stroke: INK, strokeWidth: 1.5, roughness: 1.3, bowing: 1, seed: 7, fillStyle: "hachure", hachureGap: 6, fillWeight: 1, ...extra });

function ops(drawable) {
  return gen.toPaths(drawable).map((p) => `<path d="${p.d}" stroke="${p.stroke}" stroke-width="${p.strokeWidth}" fill="${p.fill || "none"}" stroke-linecap="round" stroke-linejoin="round"/>`).join("");
}
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

class Sketch {
  constructor(w, h) { this.w = w; this.h = h; this.parts = []; this.elements = []; this.seed = 1; }

  rect(x, y, w, h, { fill, label, sub, icon, size = 20, dashed = false } = {}) {
    const d = gen.rectangle(x, y, w, h, opts({ fill, seed: this.seed++, strokeLineDash: dashed ? [8, 7] : undefined }));
    this.parts.push(ops(d));
    this.elements.push({ type: "rectangle", x, y, width: w, height: h, backgroundColor: fill || "transparent", roundness: { type: 3 }, strokeStyle: dashed ? "dashed" : "solid" });
    let ty = y + h / 2;
    if (icon) { this.icon(icon, x + w / 2 - 13, y + 9, 26); ty = y + h / 2 + 6; }
    if (label) this.text(x + w / 2, sub ? ty - 8 : ty, label, { size, bold: true });
    if (sub) this.text(x + w / 2, ty + 14, sub, { size: 13, color: MUTED });
    return { x, y, w, h, cx: x + w / 2, cy: y + h / 2, r: x + w, b: y + h };
  }
  ellipse(cx, cy, w, h, { fill, label, sub } = {}) {
    this.parts.push(ops(gen.ellipse(cx, cy, w, h, opts({ fill, seed: this.seed++ }))));
    this.elements.push({ type: "ellipse", x: cx - w / 2, y: cy - h / 2, width: w, height: h, backgroundColor: fill || "transparent" });
    if (label) this.text(cx, sub ? cy - 8 : cy, label, { size: 20, bold: true });
    if (sub) this.text(cx, cy + 14, sub, { size: 13, color: MUTED });
    return { cx, cy, x: cx - w / 2, r: cx + w / 2, y: cy - h / 2, b: cy + h / 2 };
  }
  diamond(cx, cy, w, h, { fill, label, sub } = {}) {
    const pts = [[cx, cy - h / 2], [cx + w / 2, cy], [cx, cy + h / 2], [cx - w / 2, cy]];
    this.parts.push(ops(gen.polygon(pts, opts({ fill, seed: this.seed++ }))));
    this.elements.push({ type: "diamond", x: cx - w / 2, y: cy - h / 2, width: w, height: h, backgroundColor: fill || "transparent" });
    if (label) this.text(cx, sub ? cy - 8 : cy, label, { size: 18, bold: true });
    if (sub) this.text(cx, cy + 14, sub, { size: 12, color: MUTED });
    return { cx, cy, x: cx - w / 2, r: cx + w / 2, y: cy - h / 2, b: cy + h / 2 };
  }
  text(x, y, s, { size = 16, color = INK, bold = false, anchor = "middle" } = {}) {
    const lines = String(s).split("\n"), lh = size * 1.25, y0 = y - ((lines.length - 1) * lh) / 2;
    lines.forEach((ln, i) => this.parts.push(`<text x="${x}" y="${y0 + i * lh}" font-size="${size}" fill="${color}" text-anchor="${anchor}" dominant-baseline="central" font-weight="${bold ? 600 : 400}">${esc(ln)}</text>`));
    this.elements.push({ type: "text", x: anchor === "start" ? x : x - s.length * size * 0.28, y: y - size * 0.7, width: s.length * size * 0.56, height: lh * lines.length, text: s, originalText: s, fontSize: size, fontFamily: 5, textAlign: anchor === "start" ? "left" : "center", verticalAlign: "middle", strokeColor: color });
  }
  arrow(x1, y1, x2, y2, { label, curve = 0, dashed = false, labelDy = -12, labelDx = 0, color = INK } = {}) {
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
    const nx = -dy / L, ny = dx / L, cx = mx + nx * curve, cy = my + ny * curve;
    const o = opts({ seed: this.seed++, stroke: color, strokeLineDash: dashed ? [8, 8] : undefined });
    this.parts.push(ops(curve ? gen.curve([[x1, y1], [cx, cy], [x2, y2]], o) : gen.line(x1, y1, x2, y2, o)));
    const tx = curve ? x2 - cx : dx, ty = curve ? y2 - cy : dy, T = Math.hypot(tx, ty) || 1, ux = tx / T, uy = ty / T, s = 13;
    this.parts.push(ops(gen.linearPath([[x2 - ux * s - uy * s * 0.55, y2 - uy * s + ux * s * 0.55], [x2, y2], [x2 - ux * s + uy * s * 0.55, y2 - uy * s - ux * s * 0.55]], opts({ seed: this.seed++, stroke: color }))));
    this.elements.push({ type: "arrow", x: x1, y: y1, width: dx, height: dy, points: curve ? [[0, 0], [cx - x1, cy - y1], [dx, dy]] : [[0, 0], [dx, dy]], endArrowhead: "arrow", strokeStyle: dashed ? "dashed" : "solid", strokeColor: color });
    if (label) this.text(cx + labelDx, cy + labelDy, label, { size: 13, color: MUTED });
  }
  icon(key, x, y, size) {
    const i = ICON[key];
    this.parts.push(`<g transform="translate(${x} ${y}) scale(${size / 24})"><path d="${i.path}" fill="#${i.hex}"/></g>`);
  }
  note(x, y, s, { size = 13, anchor = "start" } = {}) { this.text(x, y, s, { size, color: MUTED, anchor }); }
  tinyImage(x, y, size = 36, { ring = true, blob = false } = {}) {
    // a hand-drawn stand-in for a lens image: dark square with an Einstein ring
    this.parts.push(ops(gen.rectangle(x, y, size, size, opts({ fill: "#343a40", fillStyle: "solid", seed: this.seed++, roughness: 0.8 }))));
    if (ring) this.parts.push(ops(gen.circle(x + size / 2, y + size / 2, size * 0.55, opts({ stroke: "#f8f9fa", strokeWidth: 2, seed: this.seed++, roughness: 0.9 }))));
    if (blob) this.parts.push(ops(gen.circle(x + size * 0.7, y + size * 0.32, size * 0.14, opts({ fill: "#f8f9fa", fillStyle: "solid", stroke: "#f8f9fa", seed: this.seed++ }))));
    this.elements.push({ type: "rectangle", x, y, width: size, height: size, backgroundColor: "#343a40", fillStyle: "solid" });
  }
  svg() {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.w} ${this.h}" width="${this.w}" height="${this.h}" font-family="Excalifont, 'Segoe Print', 'Comic Sans MS', cursive">
<defs><style>@font-face{font-family:Excalifont;src:url(data:font/woff2;base64,${FONT}) format("woff2")}</style></defs>
<rect width="100%" height="100%" fill="#ffffff"/>
${this.parts.join("\n")}
</svg>`;
  }
  excalidraw() {
    let id = 0;
    const elements = this.elements.map((e) => ({ id: `el${id++}`, version: 1, versionNonce: id, isDeleted: false, groupIds: [], boundElements: [], updated: 1, link: null, locked: false, strokeColor: INK, backgroundColor: "transparent", fillStyle: "hachure", strokeWidth: 1, strokeStyle: "solid", roughness: 1, opacity: 100, angle: 0, seed: id * 7919, ...e }));
    return JSON.stringify({ type: "excalidraw", version: 2, source: "lenscraft", elements, appState: { viewBackgroundColor: "#ffffff", gridSize: null }, files: {} }, null, 1);
  }
}

function write(name, sk) {
  fs.writeFileSync(path.join(OUT, `${name}.svg`), sk.svg());
  fs.writeFileSync(path.join(OUT, "src", `${name}.excalidraw`), sk.excalidraw());
  console.log("wrote", name, `${sk.w}x${sk.h}`, `${sk.parts.length} parts`);
}

/* ============================================================ 1. the loop, with the control */
{
  const sk = new Sketch(1100, 640);
  const sim = sk.rect(60, 70, 230, 100, { fill: FILL.blue, label: "Simulate", sub: "Lenstronomy, on Modal" });
  const train = sk.rect(720, 70, 230, 100, { fill: FILL.orange, label: "Train", sub: "ResNet-18", icon: "pytorch" });
  const evalb = sk.rect(720, 290, 230, 90, { fill: FILL.teal, label: "Evaluate", sub: "3,000 held-out images" });
  const weak = sk.rect(380, 440, 260, 100, { fill: FILL.yellow, label: "Find weak spot", sub: "bin by mass, axion, SNR" });
  const gate = sk.diamond(175, 330, 170, 110, { fill: FILL.green, label: "human", sub: "y / e / n" });

  // little image strip under Simulate
  [0, 1, 2, 3].forEach((i) => sk.tinyImage(330 + i * 44, 92, 34, { blob: i === 1 }));
  sk.note(330, 150, "15,000 images, 3 classes");

  sk.arrow(sim.r + 190, sim.cy - 12, train.x - 8, train.cy - 12, { label: "images + labels", labelDy: -14 });
  sk.arrow(train.cx, train.b + 8, evalb.cx, evalb.y - 8, { label: "model", labelDx: 38, labelDy: 0 });
  sk.arrow(evalb.x - 8, evalb.cy + 20, weak.r + 8, weak.cy - 30, { label: "per-image scores", curve: 14, labelDy: 18 });
  sk.arrow(weak.x - 8, weak.cy, gate.r + 8, gate.cy + 10, { label: "proposed run card", curve: 18, labelDy: 20 });
  sk.arrow(gate.cx, gate.y - 8, sim.cx, sim.b + 8, { label: "approved", labelDx: 46, labelDy: 0 });
  sk.arrow(gate.x - 8, gate.cy, 30, gate.cy, { color: "#c92a2a" });
  sk.note(22, gate.cy + 24, "denied: agent\nexplains, no retry", { size: 12 });

  // the result box
  const res = sk.rect(720, 440, 330, 150, { fill: FILL.grey, label: "", dashed: true });
  sk.text(res.cx, res.y + 22, "one round, same budget", { size: 15, bold: true });
  sk.text(res.x + 20, res.y + 58, "targeted 2,000 images", { size: 13, color: MUTED, anchor: "start" });
  sk.text(res.r - 20, res.y + 58, "29%  ->  62%", { size: 15, color: "#2b8a3e", anchor: "end" });
  sk.text(res.x + 20, res.y + 90, "uniform 2,000 images", { size: 13, color: MUTED, anchor: "start" });
  sk.text(res.r - 20, res.y + 90, "29%  ->  28%", { size: 15, color: "#c92a2a", anchor: "end" });
  sk.text(res.x + 20, res.y + 122, "accuracy in the weak region", { size: 12, color: MUTED, anchor: "start" });
  write("sketch-loop", sk);
}

/* ============================================================ 2. three designs */
{
  const sk = new Sketch(1100, 570);
  sk.text(550, 42, "three designs before any code", { size: 22 });

  const v1 = sk.rect(50, 100, 300, 230, { fill: FILL.red, label: "v1", sub: "write everything", size: 24 });
  const v2 = sk.rect(400, 100, 300, 230, { fill: FILL.yellow, label: "v2", sub: "Orchestral AI", size: 24 });
  const v3 = sk.rect(750, 100, 300, 230, { fill: FILL.green, label: "v3", sub: "Pydantic AI", size: 24, icon: "pydantic" });
  sk.arrow(v1.r + 8, v1.cy, v2.x - 8, v2.cy);
  sk.arrow(v2.r + 8, v2.cy, v3.x - 8, v3.cy);

  // what each one meant, as small stacked notes inside the boxes
  const rows = (b, lines) => lines.forEach((l, i) => sk.text(b.cx, b.y + 165 + i * 20, l, { size: 13, color: MUTED }));
  rows(v1, ["own the agent loop", "own retries, schemas", "own the approval flow"]);
  rows(v2, ["what HEPTAPOD uses", "young, few users", "install risk"]);
  rows(v3, ["run cards already Pydantic", "tool schemas for free", "approval gating built in"]);

  // verdicts under each
  sk.text(v1.cx, 365, "too much to own", { size: 16, color: "#c92a2a" });
  sk.text(v2.cx, 365, "right idea, risky bet", { size: 16, color: "#e67700" });
  sk.text(v3.cx, 365, "shipped", { size: 16, color: "#2b8a3e", bold: true });

  // the one thing that changed the architecture
  const call = sk.rect(300, 420, 500, 110, { fill: FILL.grey, dashed: true });
  sk.text(call.cx, call.y + 24, "what v3 unlocked", { size: 14, bold: true });
  sk.text(call.cx, call.y + 52, "two agents (propose / execute) with a human between", { size: 13, color: MUTED });
  sk.arrow(call.cx, call.y + 64, call.cx, call.y + 80, { color: MUTED });
  sk.text(call.cx, call.y + 92, "one agent, gated tools, approval callback in our code", { size: 13, color: MUTED });
  write("sketch-versions", sk);
}

/* ============================================================ 3. architecture */
{
  const sk = new Sketch(1150, 720);
  const you = sk.ellipse(120, 120, 160, 76, { fill: FILL.green, label: "you", sub: "terminal" });
  const agent = sk.rect(340, 60, 260, 120, { fill: FILL.violet, label: "Agent", sub: "Pydantic AI, any LLM provider", icon: "pydantic" });
  sk.arrow(you.r + 8, you.cy - 10, agent.x - 8, agent.cy - 10, { label: "natural language", labelDy: -14 });
  sk.arrow(agent.x - 8, agent.cy + 14, you.r + 8, you.cy + 14, { label: "answer", labelDy: 16 });

  // gate between agent and the expensive tools
  const gate = sk.diamond(470, 270, 200, 90, { fill: FILL.green, label: "approve?", sub: "y / e / n" });
  sk.arrow(agent.cx, agent.b + 8, gate.cx, gate.y - 8, { label: "wants to run a tool", labelDx: 80, labelDy: 0 });
  sk.arrow(gate.x - 8, gate.cy, you.cx, you.b + 8, { label: "shows the run card", curve: 30, labelDx: -30, labelDy: 18 });

  // tools
  const t1 = sk.rect(340, 370, 260, 56, { fill: FILL.blue, label: "simulate", sub: "LensCard, costs $" });
  const t2 = sk.rect(340, 436, 260, 56, { fill: FILL.orange, label: "train / evaluate", sub: "GPU, costs $" });
  const t3 = sk.rect(340, 502, 260, 56, { fill: FILL.yellow, label: "find weak spot", sub: "free" });
  const t4 = sk.rect(340, 568, 260, 56, { fill: FILL.teal, label: "summarize / report", sub: "free" });
  sk.arrow(gate.cx, gate.b + 8, t1.cx, t1.y - 8, { label: "approved", labelDx: 44, labelDy: 0 });
  sk.text(255, 450, "gated", { size: 13, color: "#c92a2a" });
  sk.parts.push(ops(gen.rectangle(330, 362, 280, 138, opts({ seed: 99, stroke: "#c92a2a", strokeLineDash: [6, 6], roughness: 0.8 }))));

  // backend
  const be = sk.rect(700, 370, 190, 90, { fill: FILL.grey, label: "ComputeBackend", sub: "submit, status, wait", size: 15 });
  sk.arrow(t1.r + 8, t1.cy + 20, be.x - 8, be.cy - 10);
  sk.arrow(t2.r + 8, t2.cy, be.x - 8, be.cy + 10);
  const local = sk.rect(700, 500, 90, 70, { fill: "#ffffff", label: "local", sub: "laptop", size: 15 });
  const modal = sk.rect(810, 500, 300, 190, { fill: FILL.grey, label: "", sub: "" });
  sk.icon("modal", modal.x + 14, modal.y + 12, 26);
  sk.text(modal.x + 50, modal.y + 25, "Modal", { size: 18, bold: true, anchor: "start" });
  sk.arrow(be.cx - 40, be.b + 8, local.cx, local.y - 8, { label: "flag", labelDx: -24, labelDy: 0 });
  sk.arrow(be.cx + 40, be.b + 8, modal.x + 60, modal.y - 8);
  // inside modal
  sk.rect(modal.x + 16, modal.y + 50, 125, 56, { fill: FILL.blue, label: "shards", sub: "500 imgs each", size: 14 });
  sk.rect(modal.x + 158, modal.y + 50, 125, 56, { fill: FILL.orange, label: "L4 GPU", sub: "training", size: 14 });
  const vol = sk.rect(modal.x + 16, modal.y + 122, 267, 52, { fill: FILL.yellow, label: "volume", sub: "images, models, scores", size: 14 });
  sk.icon("python", vol.x + 6, vol.y - 2, 0.01); // no-op placeholder keeps element count stable
  sk.icon("python", modal.x + 200, modal.y + 8, 22); sk.icon("pytorch", modal.x + 230, modal.y + 8, 22); sk.icon("numpy", modal.x + 260, modal.y + 8, 22);

  // flow back
  sk.arrow(t3.r + 8, t3.cy, t3.r + 70, t3.cy, { dashed: true });
  sk.note(t3.x, t4.b + 24, "find weak spot reads the per-image scores off the volume", { size: 12 });
  sk.note(t3.x, t4.b + 42, "and proposes the next LensCard; nothing runs until you approve it", { size: 12 });
  sk.note(40, 600, "the agent never sees pixels:\nit reasons over one JSON record\nper image, plus per-image scores", { size: 13 });
  write("sketch-architecture", sk);
}

/* ============================================================ 4. the uncertainty step */
{
  const sk = new Sketch(1150, 620);
  sk.text(575, 36, "where is the model weak?  bin the held-out images by a physical knob", { size: 19 });

  // step 1: scores table
  const tbl = sk.rect(40, 80, 250, 160, { fill: "#ffffff", label: "" });
  sk.text(tbl.cx, tbl.y + 18, "per-image scores", { size: 14, bold: true });
  ["img 0412  vortex  P=0.31", "img 0413  vortex  P=0.88", "img 0414  vortex  P=0.19", "img 0415  subhalo P=0.97", "..."].forEach((l, i) => sk.text(tbl.x + 14, tbl.y + 48 + i * 22, l, { size: 12, color: MUTED, anchor: "start" }));
  sk.note(40, 262, "uncertainty = 1 - P(true class)", { size: 12 });

  // step 2: bars
  const x0 = 380, y0 = 330, bw = 70, gap = 18;
  const heights = [190, 36, 22, 18], accs = ["29%", "100%", "100%", "100%"];
  heights.forEach((h, i) => {
    const x = x0 + i * (bw + gap);
    sk.rect(x, y0 - h, bw, h, { fill: i === 0 ? FILL.red : FILL.blue });
    sk.text(x + bw / 2, y0 - h - 14, accs[i], { size: 12, color: i === 0 ? "#c92a2a" : MUTED });
  });
  sk.arrow(x0 - 10, y0, x0 + 4 * (bw + gap) + 10, y0, { label: "axion mass  (light  ->  heavy)", labelDy: 22 });
  sk.text(x0 - 40, 230, "mean\nuncertainty", { size: 12, color: MUTED });
  sk.arrow(tbl.r + 8, tbl.cy, x0 - 20, 200, { label: "group + average", curve: -20, labelDy: -16 });
  sk.note(x0, 380, "same for subhalos vs n_subhalos, and every class vs SNR", { size: 12 });

  // physical reason
  const why = sk.rect(380, 420, 340, 110, { fill: FILL.grey, dashed: true });
  sk.text(why.cx, why.y + 22, "why light axions are hard", { size: 14, bold: true });
  sk.text(why.cx, why.y + 50, "lighter axion  ->  longer vortex line", { size: 13, color: MUTED });
  sk.text(why.cx, why.y + 72, "same mass spread thinner  ->  arcs barely move", { size: 13, color: MUTED });
  sk.text(why.cx, why.y + 94, "so it looks like a subhalo", { size: 13, color: MUTED });

  // step 3: card
  const card = sk.rect(820, 120, 290, 170, { fill: FILL.yellow, label: "" });
  sk.text(card.cx, card.y + 22, "next run card", { size: 16, bold: true });
  ["substructure: vortex", "axion_mass: 1.7e-24 eV", "mass_fraction: 0.03", "n_images: 2000", "seed: 1101"].forEach((l, i) => sk.text(card.x + 18, card.y + 54 + i * 22, l, { size: 13, color: MUTED, anchor: "start" }));
  sk.arrow(x0 + bw + 10, y0 - 150, card.x - 8, card.cy, { label: "weakest bin  ->  its midpoint", curve: -50, labelDy: -34 });

  // step 4: after
  const after = sk.rect(820, 380, 290, 150, { fill: "#ffffff", label: "" });
  sk.text(after.cx, after.y + 22, "retrain, measure again", { size: 15, bold: true });
  // mini before/after bars
  sk.rect(after.x + 40, after.y + 110 - 50, 40, 50, { fill: FILL.red });
  sk.text(after.x + 60, after.y + 128, "before\n29%", { size: 11, color: MUTED });
  sk.rect(after.x + 110, after.y + 110 - 24, 40, 24, { fill: FILL.green });
  sk.text(after.x + 130, after.y + 128, "after\n62%", { size: 11, color: MUTED });
  sk.rect(after.x + 190, after.y + 110 - 52, 40, 52, { fill: FILL.grey });
  sk.text(after.x + 210, after.y + 128, "uniform\n28%", { size: 11, color: MUTED });
  sk.note(after.x + 40, after.y + 48, "bars = uncertainty in the weak bin", { size: 11 });
  sk.arrow(card.cx, card.b + 8, after.cx, after.y - 8, { label: "human approves, 2k images", labelDx: 100, labelDy: 0 });
  write("sketch-uncertainty", sk);
}

/* ============================================================ 5. what flows through it */
{
  const sk = new Sketch(1250, 440);
  const steps = [
    ["LensCard", "class, mass, seed", FILL.blue],
    ["lensjsonl", "one record per image:\nlabel, SNR, residual", FILL.teal],
    ["checkpoint", "best-val epoch", FILL.orange],
    ["scores", "P(class) per image", FILL.violet],
    ["uncertainty", "weakest cell\n-> next card", FILL.yellow],
  ];
  const boxes = steps.map(([label, sub, fill], i) => {
    const x = 40 + i * 250;
    const b = sk.rect(x, 90, 170, 110, { fill, label, size: 19 });
    sub.split("\n").forEach((l, j) => sk.text(b.cx, b.cy + 16 + j * 17, l, { size: 12, color: MUTED }));
    return b;
  });
  const verbs = ["simulate", "train", "evaluate", "bin + rank"];
  boxes.slice(0, -1).forEach((b, i) => sk.arrow(b.r + 6, b.cy, boxes[i + 1].x - 6, boxes[i + 1].cy, { label: verbs[i], labelDy: -12 }));
  // return path drawn as three straight segments under the row, so it never crosses a box
  const last = boxes[boxes.length - 1], first = boxes[0], yb = 300;
  const o = opts({ seed: 501 });
  sk.parts.push(ops(gen.linearPath([[last.cx, last.b + 6], [last.cx, yb], [first.cx, yb]], o)));
  sk.arrow(first.cx, yb, first.cx, first.b + 8);
  sk.text(625, yb + 20, "closes the loop: the human approves the proposed card, then it is simulated", { size: 13, color: MUTED });
  sk.text(625, 40, "what flows through it", { size: 20 });
  // tiny image strip under lensjsonl to show what the records point at
  [0, 1, 2].forEach((i) => sk.tinyImage(boxes[1].x + 20 + i * 46, 216, 32, { blob: i === 2 }));
  sk.note(boxes[1].x + 160, 232, ".npy files, on the volume", { size: 11 });
  sk.note(40, 400, "the agent only ever reads the JSON on this line, never the image pixels", { size: 13 });
  write("sketch-dataflow", sk);
}

/* ============================================================ 6. the approval gate */
{
  const sk = new Sketch(1150, 560);
  const prop = sk.rect(40, 60, 210, 90, { fill: FILL.violet, label: "model proposes", sub: "simulate_lens_batch(card)" });
  const pause = sk.rect(300, 60, 220, 90, { fill: FILL.grey, label: "run pauses", sub: "DeferredToolRequests", dashed: true });
  const gate = sk.diamond(820, 105, 220, 110, { fill: FILL.green, label: "human", sub: "reads the card" });
  sk.arrow(prop.r + 8, prop.cy, pause.x - 8, pause.cy);
  sk.arrow(pause.r + 8, pause.cy, gate.x - 8, gate.cy, { label: "shown in the terminal", labelDy: -14 });

  // three outcomes fan out below the diamond
  const y = 330;
  const yes = sk.rect(40, y, 230, 100, { fill: FILL.blue, label: "y  approve", sub: "tool runs as proposed", size: 18 });
  const edit = sk.rect(330, y, 260, 100, { fill: FILL.yellow, label: "e  edit", sub: "change fields, then run", size: 18 });
  const no = sk.rect(650, y, 260, 100, { fill: FILL.red, label: "n  deny", sub: "model reads the reason, no retry", size: 18 });
  sk.arrow(gate.cx - 40, gate.b + 4, yes.cx, yes.y - 8, { curve: 40, color: "#2b8a3e" });
  sk.arrow(gate.cx - 10, gate.b + 6, edit.cx, edit.y - 8, { color: "#e67700" });
  sk.arrow(gate.cx + 20, gate.b + 6, no.cx, no.y - 8, { color: "#c92a2a" });
  sk.text(edit.cx, edit.b + 26, "e.g.  n_images=40 substructure=subhalo", { size: 12, color: MUTED });

  const resume = sk.rect(960, 330, 170, 100, { fill: FILL.grey, label: "run resumes", sub: "DeferredToolResults", size: 15, dashed: true });
  sk.arrow(no.r + 8, no.cy, resume.x - 8, resume.cy, { dashed: true });
  sk.text(resume.cx, resume.b + 26, "whatever the answer", { size: 12, color: MUTED });
  sk.note(40, 510, "one callback, three callers: the terminal prompt, the benchmark's auto-approver, and the loop driver", { size: 12 });
  write("sketch-approval", sk);
}
