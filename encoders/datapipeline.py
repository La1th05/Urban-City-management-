from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.eurosat import EuroSAT
from torchvision.transforms import transforms
from common import Settings, get_settings
from .city_tiles import make_synthetic_city


settings= get_settings()

def stratified_split(labels: np.ndarray, val_frac: float = 0.1, seed: int = 0):
    """Seeded split that keeps the class balance. Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val.extend(idx[:n_val])
        train.extend(idx[n_val:])
    return np.sort(np.array(train)), np.sort(np.array(val))


def compute_mean_std(images: np.ndarray):
    """Per-channel mean/std over the TRAIN images only ([N, 3, H, W] in [0, 1])."""
    mean = images.mean(axis=(0, 2, 3)).astype(np.float32)
    std = images.std(axis=(0, 2, 3)).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


class NormalizedTiles(Dataset):
    """Wraps an in-memory image array [N, 3, T, T] (values in [0, 1])."""

    def __init__(self, images, labels, mean, std, augment: bool = False):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.mean = torch.from_numpy(mean).view(3, 1, 1)
        self.std = torch.from_numpy(std).view(3, 1, 1)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, k):
        x = self.images[k]
        if self.augment:  # satellite imagery has no canonical orientation
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=[1, 2])
        return (x - self.mean) / self.std, self.labels[k]


# ----------------------------------------------------------------------------
# Loading raw arrays
# ----------------------------------------------------------------------------
def load_eurosat_arrays(root: str = "data/eurosat", download: bool = True):
    from torchvision import datasets, transforms
    
    ds = datasets.EuroSAT(root=root, download=download, transform=transforms.ToTensor())
    assert list(ds.classes) == settings.EUROSAT_CLASSES, "class order changed; update city_tiles.py"
    images = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)
    labels = np.array(ds.targets, dtype=np.int64)
    return images, labels


def load_synthetic_arrays(n_cities: int = 8, seed: int = 0):
    """Many small synthetic cities flattened into one tile pool. Labels = land-use id."""
    imgs, labs = [], []
    for s in range(n_cities):
        city = make_synthetic_city(seed=seed + s)
        h, w = city["land_use"].shape
        imgs.append(city["tiles"].reshape(h * w, 3, settings.T, settings.T))
        labs.append(np.where(city["water"],5,city["land_use"]).reshape(-1))
    return np.concatenate(imgs).astype(np.float32), np.concatenate(labs).astype(np.int64)


def get_dataloaders(source: str = "eurosat", root: str = "data/eurosat",
                    batch_size: int = 128, val_frac: float = 0.1, seed: int = 0,
                    num_workers: int = 2, stats_path: str =None):
    
    stats_path = stats_path or f"data/norm_stats_{source}.json"
    torch.manual_seed(seed=0)
    if source == "eurosat":
        images, labels = load_eurosat_arrays(root)
        assert images.min()>=0 and images.max() <= 1.0, "images must be in [0, 1] before normalization"
        class_names = settings.EUROSAT_CLASSES
    elif source == "synthetic":
        images, labels = load_synthetic_arrays(seed=seed)
        class_names = list(settings.LAND_USE)+["water"]
    else:
        raise ValueError(f"unknown source {source!r}")

    tr, va = stratified_split(labels, val_frac, seed)
    mean, std = compute_mean_std(images[tr])      # train statistics only: no leakage
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(
        {"source": source, "mean": mean.tolist(), "std": std.tolist()}))

    train_ds = NormalizedTiles(images[tr], labels[tr], mean, std, augment=True)
    val_ds = NormalizedTiles(images[va], labels[va], mean, std, augment=False)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)
    info = {"mean": mean, "std": std, "class_names": class_names,
            "n_train": len(tr), "n_val": len(va)}
    return train_dl, val_dl, info


if __name__ == "__main__":
    train_dl, val_dl, info = get_dataloaders(source="synthetic", num_workers=0)
    x, y = next(iter(train_dl))
    print("train batch", tuple(x.shape), x.dtype, "| labels", y[:8].tolist())
    print("n_train", info["n_train"], "n_val", info["n_val"])
    print("mean", info["mean"], "std", info["std"])
    n = len(info["class_names"])
    for name, dl in (("train", train_dl), ("val", val_dl)):
        print(name, "class counts", torch.bincount(dl.dataset.labels, minlength=n).tolist())