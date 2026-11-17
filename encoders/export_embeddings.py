"""
export_embeddings.py  --  write the real [H, W, D] float32 embeddings (shared contract, section 3).

    python -m encoders.export_embeddings --ckpt outputs/mae/encoder.pt

Default: builds a city from real EuroSAT patches (known land use, commercial uses the
Industrial proxy), encodes every cell with the frozen MAE encoder, and writes
    data/embeddings.npy        [20, 20, 64] float32, cell (i, j) = row i, column j
    data/city_seed0.npz        land_use [20, 20] int64 and water [20, 20] bool for that city

Other options:
    --variant random   untrained encoder (the random-init ablation baseline) -> data/embeddings_random.npy
    --city synthetic   synthetic tiles instead of EuroSAT patches -> *_synthetic.npy
                       (synthetic tiles are outside the EuroSAT training distribution; plumbing only)

Load it elsewhere with:  get_embeddings("data/embeddings.npy")
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import get_settings
from encoders.city_tiles import compose_city_from_eurosat, make_land_use_map, make_synthetic_city
from encoders.datapipeline import load_eurosat_arrays
from encoders.mae import Encoder

s = get_settings()


def cluster_check(emb: np.ndarray, land: np.ndarray, water: np.ndarray):
    """Mean cosine similarity of centred embeddings: same land use versus different land use."""
    cls = np.where(water, 5, land).reshape(-1)
    e = emb.reshape(-1, emb.shape[-1]).astype(np.float64)
    e = e - e.mean(axis=0, keepdims=True)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    sim = e @ e.T
    same = cls[:, None] == cls[None, :]
    np.fill_diagonal(same, False)
    diff = cls[:, None] != cls[None, :]
    return float(sim[same].mean()), float(sim[diff].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/mae/encoder.pt")
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--city", choices=["composed", "synthetic"], default="composed")
    ap.add_argument("--city-seed", type=int, default=0)
    ap.add_argument("--variant", choices=["ssl", "random"], default="ssl")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random-init variant")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["config"]
    assert cfg["dim"] == s.D, f"encoder width {cfg['dim']} does not match contract D = {s.D}"

    # ---- the city: land use, water mask, tiles [H, W, 3, 64, 64] in [0, 1] ----
    if args.city == "composed":
        land, water = make_land_use_map(s.H, s.W, args.city_seed)
        images, labels = load_eurosat_arrays()
        tiles = compose_city_from_eurosat(images, labels, land, water, seed=args.city_seed)
    else:
        print("note: synthetic tiles are outside the EuroSAT training distribution; use for plumbing only")
        city = make_synthetic_city(seed=args.city_seed)
        land, water, tiles = city["land_use"], city["water"], city["tiles"]

    # ---- normalise exactly like pretraining (EuroSAT train-split statistics) ----
    mean = torch.tensor(ck["norm"]["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(ck["norm"]["std"], dtype=torch.float32).view(1, 3, 1, 1)
    x = torch.from_numpy(tiles.reshape(s.H * s.W, s.C, s.T, s.T)).float()  # row-major: k = i * W + j
    x = (x - mean) / std

    # ---- encoder ----
    torch.manual_seed(args.seed)
    enc = Encoder(img_size=cfg["img_size"], patch=cfg["patch"], in_ch=cfg["in_ch"],
                  dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"])
    if args.variant == "ssl":
        enc.load_state_dict(ck["encoder"])
    enc = enc.to(device).eval()

    def encode() -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 200):
                out.append(enc.embed(x[i:i + 200].to(device)).float().cpu())
        return torch.cat(out).numpy().reshape(s.H, s.W, s.D).astype(np.float32)

    emb = encode()
    assert emb.shape == (s.H, s.W, s.D) and emb.dtype == np.float32
    assert np.isfinite(emb).all(), "non-finite values in the embeddings"
    assert np.allclose(emb, encode(), atol=1e-4), "encoding is not deterministic"

    # ---- save ----
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "embeddings.npy" if args.variant == "ssl" else f"embeddings_random_s{args.seed}.npy"
    if args.city == "synthetic":
        name = name.replace(".npy", "_synthetic.npy")
    np.save(out / name, emb)
    np.savez(out / f"city_seed{args.city_seed}.npz", land_use=land, water=water)

    within, between = cluster_check(emb, land, water)
    print(f"saved {out / name}  shape {emb.shape}  dtype {emb.dtype}  ({emb.nbytes / 1024:.0f} KB)")
    print(f"embedding mean {emb.mean():.3f}  std {emb.std():.3f}")
    print(f"mean cosine similarity (centred): same land use {within:.3f} | different {between:.3f}")
    print(f"city map saved to {out / f'city_seed{args.city_seed}.npz'}")


if __name__ == "__main__":
    main()