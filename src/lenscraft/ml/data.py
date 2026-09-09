"""lensjsonl -> torch Dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lenscraft.schema.lensjsonl import LensRecord, read_records
from lenscraft.schema.ml import CLASS_INDEX


def run_id_of(record: LensRecord) -> str:
    return record.image_id.rsplit("_", 1)[0]


def load_run_records(data_root: str | Path, run_ids: list[str]) -> list[LensRecord]:
    records: list[LensRecord] = []
    for run_id in run_ids:
        path = Path(data_root) / run_id / "records.lensjsonl"
        if not path.exists():
            raise FileNotFoundError(f"no run {run_id!r} under {data_root}")
        records.extend(read_records(path))
    return records


def record_image_path(data_root: str | Path, record: LensRecord) -> Path:
    return Path(data_root) / run_id_of(record) / record.image_path


def load_image(path: str | Path, input_size: int) -> torch.Tensor:
    """Load a .npy image as a (1, S, S) float tensor scaled per-image to [0, 1]."""
    arr = np.load(path).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    x = torch.from_numpy(arr)[None, None]  # (1, 1, H, W)
    if x.shape[-1] != input_size or x.shape[-2] != input_size:
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return x[0]


PRELOAD_MAX_BYTES = 4 * 1024**3  # keep datasets up to ~4 GB of float32 pixels in RAM


class LensImageDataset(Dataset):
    """Images are decoded and resized once and kept in memory when they fit (``preload``);
    otherwise they are loaded per access. On a network volume preloading is the difference
    between GPU-bound and IO-bound epochs."""

    def __init__(self, records: list[LensRecord], data_root: str | Path, input_size: int, preload: bool | None = None) -> None:
        self.records = records
        self.data_root = Path(data_root)
        self.input_size = input_size
        self.labels = torch.tensor([CLASS_INDEX[r.substructure_type] for r in records], dtype=torch.long)
        if preload is None:
            preload = len(records) * input_size * input_size * 4 <= PRELOAD_MAX_BYTES
        self.images: torch.Tensor | None = None
        if preload and records:
            self.images = torch.stack([load_image(record_image_path(self.data_root, r), input_size) for r in records])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        if self.images is not None:
            return self.images[i], int(self.labels[i])
        rec = self.records[i]
        return load_image(record_image_path(self.data_root, rec), self.input_size), int(self.labels[i])
