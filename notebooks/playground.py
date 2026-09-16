import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Simulate a lens")


@app.cell
def _():
    import base64
    import io
    import json
    import sys

    import marimo as mo
    import numpy as np

    return base64, io, json, mo, np, sys


@app.cell
def _(json, mo, sys):
    # A precomputed grid of simulations (notebooks/data/playground_grid.json), so the widget runs
    # entirely in the browser: nothing is simulated on a server when someone moves a slider.
    def _load_text(name: str) -> str:
        # In the browser build the data lives at /marimo/data/ on the site (shared by all the
        # notebooks, independent of where the export's assets are); locally it is next to the file.
        if "pyodide" in sys.modules:
            from urllib.parse import urlsplit

            from pyodide.http import open_url

            u = urlsplit(str(mo.notebook_location()))
            return open_url(f"{u.scheme}://{u.netloc}/marimo/data/{name}").read()
        return (mo.notebook_location() / "data" / name).read_text()

    data = json.loads(_load_text("playground_grid.json"))
    MF, LA, SEEDS, GRID = data["mass_fractions"], data["log_axion"], data["seeds"], data["grid"]
    return GRID, LA, MF, SEEDS


@app.cell
def _(LA, MF, SEEDS, mo):
    substructure = mo.ui.dropdown(options=["vortex", "subhalo", "none"], value="vortex", label="substructure")
    mass_fraction = mo.ui.slider(steps=MF, value=0.03, label="mass fraction")
    log_axion = mo.ui.slider(steps=LA, value=-23.0, label="log10 axion mass (eV)")
    seed = mo.ui.slider(steps=SEEDS, value=3, label="seed")
    return log_axion, mass_fraction, seed, substructure


@app.cell
def _(log_axion, mass_fraction, mo, seed, substructure):
    controls = [substructure, seed]
    if substructure.value != "none":
        controls.append(mass_fraction)
    if substructure.value == "vortex":
        controls.append(log_axion)
    mo.md(
        f"""
The simulator the agent drives, with DeepLenseSim's defaults: a 10<sup>12</sup> M<sub>☉</sub> lens at z = 0.5,
a Sérsic source at z = 1, Euclid-like pixels. Left: the observed image. Right: what the substructure adds to the arcs.

{mo.hstack(controls, gap=2, justify="start", wrap=True)}
"""
    )
    return


@app.cell
def _(GRID, base64, io, log_axion, mass_fraction, mo, np, seed, substructure):
    from PIL import Image

    cls = substructure.value
    key = f"{cls}|{0 if cls == 'none' else mass_fraction.value}|{log_axion.value if cls == 'vortex' else 0}|{seed.value}"
    entry = GRID[key]

    def _u8(b64: str) -> "np.ndarray":
        return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(64, 64)

    def _png(rgb: "np.ndarray") -> bytes:
        buf = io.BytesIO()
        Image.fromarray(rgb).resize((320, 320), Image.NEAREST).save(buf, format="PNG")
        return buf.getvalue()

    gray = _u8(entry["img"])
    observed = np.stack([gray] * 3, axis=-1)

    # residual: 128 = zero; blue below, red above, white at zero (a small diverging map, no matplotlib)
    r = (_u8(entry["res"]).astype(float) - 128.0) / 127.0
    t = np.abs(r)[..., None]
    red = np.array([178, 24, 43], dtype=float)
    blue = np.array([33, 102, 172], dtype=float)
    white = np.array([255, 255, 255], dtype=float)
    col = np.where(r[..., None] >= 0, red, blue)
    residual = (white * (1 - t) + col * t).clip(0, 255).astype(np.uint8)

    f = entry["facts"]
    facts = [f"Einstein radius {f['theta_E']:.2f}″", f"field of view {f['fov']:.1f}″"]
    if cls == "vortex":
        facts.append(f"vortex half-length {f['vortex_half_length']:.2f}″")
    if cls == "subhalo":
        facts.append(f"{f['n_subhalos']} subhalos")
    if cls != "none":
        facts.append(f"peak perturbation {f['peak_fraction']:.1%} of peak brightness")
    else:
        facts.append("no substructure, so the residual is zero")

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([mo.md("**observed**"), mo.image(src=_png(observed), width=320)], gap=0.3),
                    mo.vstack([mo.md("**perturbation from substructure**"), mo.image(src=_png(residual), width=320)], gap=0.3),
                ],
                gap=1.5,
                justify="start",
                wrap=True,
            ),
            mo.md(" · ".join(facts)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
