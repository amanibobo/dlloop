import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Simulate a lens")


@app.cell
def _():
    from functools import lru_cache

    import marimo as mo
    import numpy as np

    return lru_cache, mo, np


@app.cell
def _(mo):
    substructure = mo.ui.dropdown(options=["vortex", "subhalo", "none"], value="vortex", label="substructure")
    mass_fraction = mo.ui.slider(0.005, 0.1, step=0.005, value=0.03, label="mass fraction")
    log_axion = mo.ui.slider(-24.0, -22.0, step=0.1, value=-23.0, label="log10 axion mass (eV)")
    seed = mo.ui.number(0, 9999, value=3, label="seed")
    return log_axion, mass_fraction, seed, substructure


@app.cell
def _(log_axion, mass_fraction, mo, seed, substructure):
    controls = [substructure, mass_fraction, seed] + ([log_axion] if substructure.value == "vortex" else [])
    mo.md(
        f"""
The same simulator the agent drives, with DeepLenseSim's defaults: a 10<sup>12</sup> M<sub>☉</sub> lens at z = 0.5,
a Sérsic source at z = 1, Euclid-like pixels. Left: the observed image. Right: what the substructure adds to the arcs.

{mo.hstack(controls, gap=2, justify="start", wrap=True)}
"""
    )
    return


@app.cell
def _(log_axion, lru_cache, mass_fraction, np, seed, substructure):
    from lenscraft.schema import LensCard
    from lenscraft.sim.batch import image_rng
    from lenscraft.sim.lenstronomy_backend import simulate_one

    @lru_cache(maxsize=256)
    def render(cls: str, mf: float, la: float, sd: int):
        kw = {"n_images": 1, "substructure": cls, "image_size": 64, "seed": sd}
        if cls != "none":
            kw["substructure_mass_fraction"] = mf
        if cls == "vortex":
            kw["axion_mass"] = 10**la
        card = LensCard(**kw)
        r = simulate_one(card, image_rng(card, 0))
        return r.image, r.model - r.smooth_model, r.realised

    image, residual, realised = render(substructure.value, float(mass_fraction.value), float(log_axion.value), int(seed.value))
    return image, realised, residual


@app.cell
def _(image, mo, np, realised, residual, substructure):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 4), facecolor="white")
    a.imshow(np.sqrt(np.clip(image, 0, None)), cmap="gray")
    a.set_title("observed", loc="left", fontsize=10)
    v = max(float(np.abs(residual).max()), 1e-9)
    b.imshow(residual, cmap="RdBu_r", vmin=-v, vmax=v)
    b.set_title("perturbation from substructure", loc="left", fontsize=10)
    for ax in (a, b):
        ax.axis("off")
    fig.tight_layout()

    facts = [f"Einstein radius {realised['theta_E']:.2f}″", f"field of view {realised['field_of_view']:.1f}″"]
    if substructure.value == "vortex":
        facts.append(f"vortex half-length {realised['vortex_half_length']:.2f}″")
    if substructure.value == "subhalo":
        facts.append(f"{int(realised['n_subhalos'])} subhalos")
    peak = float(np.abs(residual).max() / max(float(image.max()), 1e-9))
    facts.append(f"peak perturbation {peak:.1%} of peak brightness")
    mo.vstack([fig, mo.md(" · ".join(facts))])
    return


if __name__ == "__main__":
    app.run()
