/* Render each public/media/sketch-*.svg to a 2x PNG with the embedded font. Run after sketches.cjs. */
const { chromium } = require("playwright");
const fs = require("node:fs"); const path = require("node:path");
(async () => {
  const dir = path.join(__dirname, "..", "public", "media");
  const browser = await chromium.launch();
  for (const f of fs.readdirSync(dir).filter((n) => n.startsWith("sketch-") && n.endsWith(".svg"))) {
    const svg = fs.readFileSync(path.join(dir, f), "utf8");
    const [, w, h] = svg.match(/viewBox="0 0 (\d+) (\d+)"/);
    const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 2 });
    await page.setContent(`<html><body style="margin:0">${svg}</body></html>`);
    await page.evaluate(() => document.fonts.ready); await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(dir, f.replace(".svg", ".png")), clip: { x: 0, y: 0, width: +w, height: +h } });
    await page.close(); console.log("rasterized", f.replace(".svg", ".png"));
  }
  await browser.close();
})();
