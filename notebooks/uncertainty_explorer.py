import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Where is the classifier weak?")


@app.cell
def _():
    import io
    import json
    import sys

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return alt, io, json, mo, np, pd, sys


@app.cell
def _(json, mo, np, sys):
    # Data sits next to the notebook (data/...). mo.notebook_location() is a directory locally and
    # a URL in the browser build, where the files are fetched synchronously.
    def _load_text(name: str) -> str:
        # In the browser build the data lives at /marimo/data/ on the site (shared by all the
        # notebooks, independent of where the export's assets are); locally it is next to the file.
        if "pyodide" in sys.modules:
            from urllib.parse import urlsplit

            from pyodide.http import open_url

            u = urlsplit(str(mo.notebook_location()))
            return open_url(f"{u.scheme}://{u.netloc}/marimo/data/{name}").read()
        return (mo.notebook_location() / "data" / name).read_text()

    scores = json.loads(_load_text("scores_clf_base_b.json"))
    _imgs = json.loads(_load_text("images_te1k.json"))
    import base64

    images = {k: np.frombuffer(base64.b64decode(v), dtype=np.uint8).reshape(64, 64) for k, v in _imgs.items()}
    return images, scores


@app.cell
def _(np, pd, scores):
    CLASSES = ["none", "subhalo", "vortex"]
    rows = []
    for im in scores["images"]:
        li = CLASSES.index(im["label"])
        rows.append(
            {
                "image_id": im["image_id"],
                "label": im["label"],
                "pred": im["pred"],
                "p_none": im["probs"][0],
                "p_subhalo": im["probs"][1],
                "p_vortex": im["probs"][2],
                "uncertainty": 1.0 - im["probs"][li],
                "correct": im["pred"] == im["label"],
                "snr": im["snr"],
                "mass_fraction": im["mass_fraction"],
                "axion_mass": im["axion_mass"] if im["axion_mass"] is not None else np.nan,
                "n_subhalos": im["n_subhalos"] if im["n_subhalos"] is not None else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    AXES = {"none": ["snr"], "subhalo": ["n_subhalos", "snr"], "vortex": ["axion_mass", "snr"]}
    return AXES, CLASSES, df


@app.cell
def _(AXES, mo):
    cls = mo.ui.dropdown(options=["vortex", "subhalo", "none"], value="vortex", label="class")
    n_bins = mo.ui.slider(2, 8, value=4, label="bins")
    return cls, n_bins


@app.cell
def _(AXES, cls, mo):
    axis = mo.ui.dropdown(options=AXES[cls.value], value=AXES[cls.value][0], label="axis")
    return (axis,)


@app.cell
def _(axis, cls, mo, n_bins, scores):
    mo.md(
        f"""
**{scores['model_id']}** on {sum(1 for _ in scores['images']):,} held-out images. Per-image uncertainty is
1 − P(true class). Pick a class and an axis; bins are quantiles. Click a bar to see images from that cell.

{mo.hstack([cls, axis, n_bins], gap=2, justify="start")}
"""
    )
    return


@app.cell
def _(alt, axis, cls, df, mo, n_bins, np, pd):
    sub = df[(df["label"] == cls.value) & df[axis.value].notna()].copy()
    log = axis.value == "axion_mass"
    v = np.log10(sub[axis.value]) if log else sub[axis.value]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins.value + 1)))
    which = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    sub["bin"] = which
    cells = (
        sub.groupby("bin")
        .agg(n=("uncertainty", "size"), uncertainty=("uncertainty", "mean"), accuracy=("correct", "mean"))
        .reset_index()
    )
    lo = edges[cells["bin"]]
    hi = edges[cells["bin"] + 1]
    if log:
        lo, hi = 10**lo, 10**hi
    cells["range"] = [f"{a:.3g} to {b:.3g}" for a, b in zip(lo, hi)]
    cells["cell"] = [f"{axis.value} {r}" for r in cells["range"]]
    color = {"none": "#2a78d6", "subhalo": "#eb6834", "vortex": "#1baf7a"}[cls.value]
    cells["label"] = [f"{r}  |  accuracy {a:.0%}  |  n={int(k)}" for r, a, k in zip(cells["range"], cells["accuracy"], cells["n"])]
    # no cornerRadiusEnd: inside mo.ui.altair_chart it makes the bars disappear (marimo 0.24)
    bars = (
        alt.Chart(cells)
        .mark_bar(color=color)
        .encode(
            y=alt.Y("label:N", sort=alt.EncodingSortField("bin"), title=None, axis=alt.Axis(labelLimit=320, labelFontSize=11)),
            x=alt.X("uncertainty:Q", scale=alt.Scale(domain=[0, 1]), title="mean uncertainty (1 - P(true class))"),
            tooltip=["range:N", "n:Q", alt.Tooltip("uncertainty:Q", format=".3f"), alt.Tooltip("accuracy:Q", format=".3f")],
        )
        .properties(height=32 * len(cells) + 20, width="container")
    )
    chart = mo.ui.altair_chart(bars, chart_selection="point", legend_selection=False)
    weakest = cells.sort_values("uncertainty", ascending=False).iloc[0]
    return cells, chart, sub, weakest


@app.cell
def _(chart, mo, weakest):
    mo.vstack(
        [
            chart,
            mo.md(
                f"Weakest: **{weakest['cell']}** (n = {int(weakest['n'])}, mean uncertainty {weakest['uncertainty']:.2f}, "
                f"accuracy {weakest['accuracy']:.0%}). Labels on the bars are accuracy per bin."
            ),
        ]
    )
    return


@app.cell
def _(chart, images, io, mo, sub):
    from PIL import Image

    def _png(arr) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).resize((128, 128), Image.NEAREST).save(buf, format="PNG")
        return buf.getvalue()

    picked = chart.value
    if picked is None or len(picked) == 0:
        _out = mo.md("*Click a bar to see sample images from that cell.*")
    else:
        bins = set(int(b) for b in picked["bin"].tolist())
        _rows = sub[sub["bin"].isin(bins) & sub["image_id"].isin(images.keys())].sort_values("uncertainty", ascending=False).head(12)
        if len(_rows) == 0:
            _out = mo.md("*No bundled images in this cell (the page ships 200 per class).*")
        else:
            tiles = [
                mo.vstack(
                    [
                        mo.image(src=_png(images[r["image_id"]]), width=110),
                        mo.md(f"<span style='font-size:11px;white-space:nowrap;color:{'#111' if r['correct'] else '#d03b3b'}'>{r['pred']} {r['uncertainty']:.2f}</span>"),
                    ],
                    gap=0.2,
                )
                for _, r in _rows.iterrows()
            ]
            _out = mo.vstack([mo.md("Most uncertain images in the selected cell (red = misclassified):"), mo.hstack(tiles, wrap=True, gap=0.8, justify="start")])
    _out
    return


if __name__ == "__main__":
    app.run()
