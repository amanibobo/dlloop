/* Generates the four hand-drawn diagrams as Excalidraw-style SVGs (rough.js strokes, Excalifont
   handwriting, pastel fills, tool logos) plus matching .excalidraw scene files you can open and
   edit at excalidraw.com.   Run: node scripts/sketches.cjs   (writes to public/media/) */

const fs = require("node:fs");
const path = require("node:path");
const rough = require("roughjs/bundled/rough.cjs.js");
const si = require("simple-icons");

const OUT = path.join(__dirname, "..", "public", "media");
const FONT = fs.readFileSync(path.join(OUT, "src", "Excalifont.woff2")).toString("base64");

// Excalidraw's default palette
const INK = "#1e1e1e";
const FILL = { blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", orange: "#ffd8a8", violet: "#d0bfff", grey: "#e9ecef", red: "#ffc9c9" };
const ICON = { python: si.siPython, pytorch: si.siPytorch, pydantic: si.siPydantic, modal: si.siModal, numpy: si.siNumpy, hf: si.siHuggingface };

const gen = rough.generator();
const opts = (extra = {}) => ({ stroke: INK, strokeWidth: 1.5, roughness: 1.4, bowing: 1, seed: 7, fillStyle: "hachure", hachureGap: 6, fillWeight: 1, ...extra });

function ops(drawable) {
  const paths = gen.toPaths(drawable);
  return paths.map((p) => `<path d="${p.d}" stroke="${p.stroke}" stroke-width="${p.strokeWidth}" fill="${p.fill || "none"}" stroke-linecap="round" stroke-linejoin="round"/>`).join("");
}

class Sketch {
  constructor(w, h) {
    this.w = w; this.h = h; this.parts = []; this.elements = []; this.seed = 1;
  }
  rect(x, y, w, h, { fill, label, sub, icon, radius = 12 } = {}) {
    const d = gen.rectangle(x, y, w, h, opts({ fill, seed: this.seed++ }));
    this.parts.push(ops(d));
    this.elements.push({ type: "rectangle", x, y, width: w, height: h, backgroundColor: fill || "transparent", roundness: { type: 3 } });
    let ty = y + h / 2;
    if (icon) { this.icon(icon, x + w / 2 - 14, y + 10, 28); ty = y + h / 2 + 14; }
    if (label) this.text(x + w / 2, sub ? ty - 8 : ty, label, { size: 20, bold: true });
    if (sub) this.text(x + w / 2, ty + 14, sub, { size: 13, color: "#495057" });
    return { x, y, w, h, cx: x + w / 2, cy: y + h / 2, r: x + w, b: y + h };
  }
  ellipse(cx, cy, w, h, { fill, label, sub } = {}) {
    const d = gen.ellipse(cx, cy, w, h, opts({ fill, seed: this.seed++ }));
    this.parts.push(ops(d));
    this.elements.push({ type: "ellipse", x: cx - w / 2, y: cy - h / 2, width: w, height: h, backgroundColor: fill || "transparent" });
    if (label) this.text(cx, sub ? cy - 8 : cy, label, { size: 20, bold: true });
    if (sub) this.text(cx, cy + 14, sub, { size: 13, color: "#495057" });
    return { cx, cy };
  }
  text(x, y, s, { size = 16, color = INK, bold = false, anchor = "middle" } = {}) {
    const lines = String(s).split("\n");
    const lh = size * 1.25;
    const y0 = y - ((lines.length - 1) * lh) / 2;
    lines.forEach((ln, i) => {
      this.parts.push(`<text x="${x}" y="${y0 + i * lh}" font-size="${size}" fill="${color}" text-anchor="${anchor}" dominant-baseline="central" font-weight="${bold ? 600 : 400}">${esc(ln)}</text>`);
    });
    this.elements.push({ type: "text", x: anchor === "start" ? x : x - (s.length * size * 0.28), y: y - size * 0.7, width: s.length * size * 0.56, height: lh * lines.length, text: s, originalText: s, fontSize: size, fontFamily: 5, textAlign: anchor === "start" ? "left" : "center", verticalAlign: "middle", strokeColor: color });
  }
  arrow(x1, y1, x2, y2, { label, curve = 0, dashed = false, labelDy = -10 } = {}) {
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
    const nx = -dy / L, ny = dx / L; // normal
    const cx = mx + nx * curve, cy = my + ny * curve;
    const o = opts({ seed: this.seed++, strokeLineDash: dashed ? [8, 8] : undefined });
    const d = curve ? gen.curve([[x1, y1], [cx, cy], [x2, y2]], o) : gen.line(x1, y1, x2, y2, o);
    this.parts.push(ops(d));
    // arrowhead, oriented along the final tangent
    const tx = curve ? x2 - cx : dx, ty = curve ? y2 - cy : dy, T = Math.hypot(tx, ty) || 1;
    const ux = tx / T, uy = ty / T, s = 14;
    const h1 = [x2 - ux * s - uy * s * 0.55, y2 - uy * s + ux * s * 0.55];
    const h2 = [x2 - ux * s + uy * s * 0.55, y2 - uy * s - ux * s * 0.55];
    this.parts.push(ops(gen.linearPath([h1, [x2, y2], h2], opts({ seed: this.seed++ }))));
    this.elements.push({ type: "arrow", x: x1, y: y1, width: dx, height: dy, points: curve ? [[0, 0], [cx - x1, cy - y1], [dx, dy]] : [[0, 0], [dx, dy]], endArrowhead: "arrow", strokeStyle: dashed ? "dashed" : "solid" });
    if (label) this.text(cx, cy + labelDy, label, { size: 14, color: "#495057" });
  }
  icon(key, x, y, size) {
    const i = ICON[key];
    const k = size / 24;
    this.parts.push(`<g transform="translate(${x} ${y}) scale(${k})"><path d="${i.path}" fill="#${i.hex}"/></g>`);
  }
  note(x, y, s, { size = 14 } = {}) {
    this.text(x, y, s, { size, color: "#495057", anchor: "start" });
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
    const elements = this.elements.map((e) => ({
      id: `el${id++}`, version: 1, versionNonce: id, isDeleted: false, groupIds: [], boundElements: [], updated: 1, link: null, locked: false,
      strokeColor: INK, backgroundColor: "transparent", fillStyle: "hachure", strokeWidth: 1, strokeStyle: "solid", roughness: 1, opacity: 100, angle: 0, seed: id * 7919,
      ...e,
    }));
    return JSON.stringify({ type: "excalidraw", version: 2, source: "lenscraft", elements, appState: { viewBackgroundColor: "#ffffff", gridSize: null }, files: {} }, null, 1);
  }
}

function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

function write(name, sk) {
  fs.writeFileSync(path.join(OUT, `${name}.svg`), sk.svg());
  fs.writeFileSync(path.join(OUT, "src", `${name}.excalidraw`), sk.excalidraw());
  console.log("wrote", name, `${sk.w}x${sk.h}`, `${sk.parts.length} parts`);
}

/* ------------------------------------------------------------------ 1. the loop */
{
  const sk = new Sketch(900, 520);
  const sim = sk.rect(70, 60, 220, 90, { fill: FILL.blue, label: "Simulate", sub: "lens images" });
  const train = sk.rect(600, 60, 220, 90, { fill: FILL.orange, label: "Train", sub: "classifier", icon: "pytorch" });
  const weak = sk.rect(335, 330, 230, 90, { fill: FILL.yellow, label: "Find weak spot", sub: "where it's unsure" });
  sk.arrow(sim.r + 8, sim.cy, train.x - 8, train.cy, { label: "images", curve: -28 });
  sk.arrow(train.cx, train.b + 8, weak.r - 20, weak.y - 8, { label: "scores", curve: -26 });
  sk.arrow(weak.x + 20, weak.y - 8, sim.cx, sim.b + 8, { curve: -26 });
  sk.text(150, 300, "simulate more\nof that", { size: 14, color: "#495057" });
  sk.ellipse(450, 200, 150, 54, { fill: FILL.green, label: "human", sub: "approves" });
  sk.note(600, 460, "repeat until the model stops", { size: 15 });
  sk.note(600, 482, "being surprised", { size: 15 });
  write("sketch-loop", sk);
}

/* ------------------------------------------------------------------ 2. versions */
{
  const sk = new Sketch(960, 420);
  const v1 = sk.rect(50, 110, 240, 150, { fill: FILL.red, label: "v1", sub: "build everything by hand" });
  const v2 = sk.rect(360, 110, 240, 150, { fill: FILL.yellow, label: "v2", sub: "Orchestral AI" });
  const v3 = sk.rect(670, 110, 240, 150, { fill: FILL.green, label: "v3", sub: "Pydantic AI", icon: "pydantic" });
  sk.arrow(v1.r + 8, v1.cy, v2.x - 8, v2.cy);
  sk.arrow(v2.r + 8, v2.cy, v3.x - 8, v3.cy);
  sk.text(v1.cx, 300, "too much to own", { size: 15, color: "#495057" });
  sk.text(v2.cx, 300, "young framework,\ninstall risk", { size: 15, color: "#495057" });
  sk.text(v3.cx, 300, "cards already Pydantic,\napproval built in", { size: 15, color: "#495057" });
  sk.text(480, 55, "three designs before any code", { size: 20 });
  write("sketch-versions", sk);
}

/* ------------------------------------------------------------------ 3. architecture */
{
  const sk = new Sketch(1000, 560);
  const you = sk.ellipse(120, 120, 150, 70, { fill: FILL.green, label: "you", sub: "y / e / n" });
  const agent = sk.rect(330, 70, 230, 100, { fill: FILL.violet, label: "Agent", sub: "Pydantic AI", icon: "pydantic" });
  sk.arrow(you.cx + 80, you.cy, agent.x - 8, agent.y + 50, { label: "ask / approve", labelDy: -14 });
  // tools
  const t1 = sk.rect(330, 240, 230, 60, { fill: FILL.blue, label: "simulate" });
  const t2 = sk.rect(330, 310, 230, 60, { fill: FILL.orange, label: "train / evaluate" });
  const t3 = sk.rect(330, 380, 230, 60, { fill: FILL.yellow, label: "find weak spot" });
  sk.arrow(agent.cx, agent.b + 8, t1.cx, t1.y - 8, { label: "tools", labelDy: 0 });
  // modal
  const modal = sk.rect(680, 240, 260, 200, { fill: FILL.grey, label: "Modal", sub: "containers + GPU", icon: "modal" });
  sk.arrow(t2.r + 8, t2.cy, modal.x - 8, t2.cy);
  sk.arrow(t1.r + 8, t1.cy, modal.x - 8, t1.cy);
  sk.icon("python", 700, 390, 30); sk.icon("pytorch", 745, 390, 30); sk.icon("numpy", 790, 390, 30);
  sk.text(810, 500, "images + models live here,\nnever on the laptop", { size: 14, color: "#495057" });
  sk.text(120, 470, "approval sits between\nthe agent and anything\nthat costs money", { size: 14, color: "#495057" });
  write("sketch-architecture", sk);
}

/* ------------------------------------------------------------------ 4. uncertainty */
{
  const sk = new Sketch(960, 480);
  // a bar chart of bins along axion mass, one bar tall
  sk.text(480, 40, "bin the held-out images by a physical knob", { size: 20 });
  const x0 = 120, y0 = 330, bw = 110, gap = 30;
  const heights = [200, 40, 25, 20];
  const labels = ["light", "", "", "heavy"];
  heights.forEach((h, i) => {
    const x = x0 + i * (bw + gap);
    sk.rect(x, y0 - h, bw, h, { fill: i === 0 ? FILL.red : FILL.blue });
    if (labels[i]) sk.text(x + bw / 2, y0 + 22, labels[i], { size: 15, color: "#495057" });
  });
  sk.arrow(x0 - 10, y0, x0 + 4 * (bw + gap), y0, { label: "axion mass", labelDy: 42 });
  sk.text(x0 + bw / 2, y0 - 225, "29% right", { size: 15, color: "#c92a2a" });
  sk.text(60, 200, "how\nunsure", { size: 15, color: "#495057" });
  // the arrow out to the next card
  const card = sk.rect(690, 150, 230, 130, { fill: FILL.yellow, label: "next batch" });
  sk.text(card.cx, card.cy + 26, "2,000 light-axion vortices", { size: 13, color: "#495057" });
  sk.arrow(x0 + bw + 8, y0 - 150, card.x - 8, card.cy, { label: "weakest bin becomes\nthe next run card", curve: -40, labelDy: -30 });
  sk.text(805, 320, "then retrain and\nmeasure again", { size: 14, color: "#495057" });
  write("sketch-uncertainty", sk);
}
