"""Interactive simulator playground: a marimo app served from Modal and embedded in the case-study
site. Its own Modal app (no GPU) so it deploys independently of the training functions::

    modal deploy -m lenscraft.compute.modal_playground
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "lenscraft-playground"
_NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "playground.py"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("pydantic>=2.7", "numpy>=1.26", "scipy>=1.12", "astropy>=6.0", "lenstronomy>=1.12", "marimo>=0.24", "matplotlib>=3.8")
    .add_local_python_source("lenscraft")
)
if modal.is_local() and _NOTEBOOK.exists():
    image = image.add_local_file(str(_NOTEBOOK), "/root/playground.py")


@app.function(image=image, scaledown_window=300, timeout=3600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def playground():
    """marimo in run mode (no code shown); scales to zero when idle."""
    import marimo

    server = marimo.create_asgi_app(quiet=True, include_code=False).with_app(path="", root="/root/playground.py")
    return server.build()
