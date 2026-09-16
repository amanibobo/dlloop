import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="pass@k")


@app.cell
def _():
    from math import comb

    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, comb, mo, pd


@app.cell
def _(mo):
    n = mo.ui.slider(1, 30, value=3, label="trials n")
    return (n,)


@app.cell
def _(mo, n):
    c = mo.ui.slider(0, n.value, value=min(3, n.value), label="successes c")
    return (c,)


@app.cell
def _(c, mo, n):
    mo.md(
        f"""
The unbiased estimators from the benchmark: **pass@k** is the chance that at least one of k sampled attempts
succeeds, **pass^k** the chance that all k do, given n trials with c successes.

{mo.hstack([n, c], gap=2, justify="start")}
"""
    )
    return


@app.cell
def _(alt, c, comb, mo, n, pd):
    def pass_at_k(n_, c_, k):
        return 1.0 if n_ - c_ < k else 1.0 - comb(n_ - c_, k) / comb(n_, k)

    def pass_pow_k(n_, c_, k):
        return 0.0 if c_ < k else comb(c_, k) / comb(n_, k)

    ks = list(range(1, n.value + 1))
    df = pd.DataFrame(
        [{"k": k, "estimator": "pass@k", "value": pass_at_k(n.value, c.value, k)} for k in ks]
        + [{"k": k, "estimator": "pass^k", "value": pass_pow_k(n.value, c.value, k)} for k in ks]
    )
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2)
        .encode(
            x=alt.X("k:Q", axis=alt.Axis(tickMinStep=1), title="k"),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            color=alt.Color("estimator:N", scale=alt.Scale(range=["#2a78d6", "#eb6834"]), legend=alt.Legend(title=None, orient="top-left")),
            tooltip=["estimator:N", "k:Q", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=220, width="container")
    )
    p1 = pass_at_k(n.value, c.value, 1)
    note = (
        f"With n = {n.value} and c = {c.value}: pass@1 = {p1:.2f}. "
        + ("Both benchmark arms scored 3 of 3, which pins pass@1 at 1.00 but says little about how often a fourth trial would pass." if n.value == 3 and c.value == 3 else "")
    )
    mo.vstack([chart, mo.md(note)])
    return


if __name__ == "__main__":
    app.run()
