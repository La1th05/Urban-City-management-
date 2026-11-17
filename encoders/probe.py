"""
probe.py  --  linear probe on frozen embeddings, with two baselines.

    python -m encoders.probe --ckpt outputs/mae/encoder.pt

Compares three 'encoders' on the same EuroSAT train/val split:
    ssl      the pretrained MAE encoder (frozen)
    random   the same architecture, randomly initialised, never trained
    color    per-channel mean and std of the tile (6 numbers, no learning)
For each: a linear classifier is trained on train embeddings only and scored on val.
Reports overall accuracy, accuracy per EuroSAT class, accuracy after grouping the
10 classes into the 5 zoning-style groups, and the biggest confusions (SSL only).
Labels are used only here, never during MAE pretraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from encoders.datapipeline import get_dataloaders
from encoders.mae import Encoder

# EuroSAT class -> zoning-style group (same simplification as the city mapping)
GROUP_OF = {
    "AnnualCrop": "green", "Forest": "green", "HerbaceousVegetation": "green",
    "Pasture": "green", "PermanentCrop": "green", "Industrial": "industrial",
    "Residential": "residential", "Highway": "mixed", "River": "water", "SeaLake": "water",
}


@torch.no_grad()
def extract(fn, dl, device):
    feats, labels = [], []
    for x, y in dl:
        feats.append(fn(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def fit_predict(ftr, ytr, fva, n_classes, steps=500, lr=1e-2, wd=1e-4):
    """Train a linear layer on standardised train features; return val predictions."""
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fva = (ftr - mu) / sd, (fva - mu) / sd
    torch.manual_seed(0)
    lin = nn.Linear(ftr.shape[1], n_classes)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(ftr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return lin(fva).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/mae/encoder.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["config"]

    train_dl, val_dl, info = get_dataloaders(source="eurosat", batch_size=512,
                                             seed=args.seed, num_workers=0)
    if not (np.allclose(info["mean"], ck["norm"]["mean"], atol=1e-5)
            and np.allclose(info["std"], ck["norm"]["std"], atol=1e-5)):
        print("WARNING: normalisation stats differ from the ones used in pretraining")

    train_dl.dataset.augment = False  # no random flips when extracting features
    tr_dl = DataLoader(train_dl.dataset, batch_size=512, shuffle=False)
    names = list(info["class_names"])
    n_classes = len(names)

    def build(trained: bool):
        torch.manual_seed(args.seed)
        enc = Encoder(img_size=cfg["img_size"], patch=cfg["patch"], in_ch=cfg["in_ch"],
                      dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"])
        if trained:
            enc.load_state_dict(ck["encoder"])
        return enc.to(device).eval()

    ssl_enc, rand_enc = build(True), build(False)
    extractors = {
        "ssl": ssl_enc.embed,
        "random": rand_enc.embed,
        "color": lambda x: torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1),
    }

    group_names = sorted(set(GROUP_OF.values()))
    group_idx = np.array([group_names.index(GROUP_OF[n]) for n in names])

    results, preds = {}, {}
    for key, fn in extractors.items():
        ftr, ytr = extract(fn, tr_dl, device)
        fva, yva = extract(fn, val_dl, device)
        pred = fit_predict(ftr, ytr, fva, n_classes)
        y, p = yva.numpy(), pred.numpy()
        per_class = {names[c]: float((p[y == c] == c).mean()) for c in range(n_classes)}
        results[key] = {
            "overall_acc": float((p == y).mean()),
            "group_acc": float((group_idx[p] == group_idx[y]).mean()),
            "per_class_acc": per_class,
            "feature_dim": int(ftr.shape[1]),
        }
        preds[key] = (y, p)

    print(f"\nval images: {len(preds['ssl'][0])} | classes: {n_classes}\n")
    print(f"{'features':<10}{'dim':>5}{'overall':>10}{'5 groups':>10}")
    for key, r in results.items():
        print(f"{key:<10}{r['feature_dim']:>5}{r['overall_acc']:>10.3f}{r['group_acc']:>10.3f}")

    print("\nper-class accuracy")
    print(f"{'class':<22}{'ssl':>8}{'random':>8}{'color':>8}")
    for n in names:
        print(f"{n:<22}" + "".join(f"{results[k]['per_class_acc'][n]:>8.3f}" for k in extractors))

    y, p = preds["ssl"]
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y, p), 1)
    off = [(cm[i, j], names[i], names[j]) for i in range(n_classes) for j in range(n_classes) if i != j]
    print("\nlargest SSL confusions (true -> predicted, count)")
    for cnt, t, q in sorted(off, reverse=True)[:6]:
        print(f"  {t} -> {q}: {cnt}")

    out = Path(args.ckpt).parent / "probe_results.json"
    out.write_text(json.dumps(results, indent=2))
    np.savetxt(Path(args.ckpt).parent / "probe_confusion_ssl.csv", cm, fmt="%d", delimiter=",",
               header=",".join(names), comments="")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()