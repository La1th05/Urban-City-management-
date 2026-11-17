"""
map_adapter.py  --  turn the map app's fine label grid into the agents' coarse city grid.

Input  : the app's exported grid (see export snippet in the notes), either
           map_grid.json   {"meters_per_cell": 2, "cells": [["Street", "Green", null, ...], ...]}
         or
           map_labels.npy  integer ids [h, w]  +  --legend legend.json  {"Street": 1, ...}
Output : data/map_city.npz with
         fractions  [H, W, K] float32  share of each observed class in every coarse cell
         land_use   [H, W]    int64    existing land use: 0 res, 1 commercial, 3 green, 4 mixed, -1 open
         water      [H, W]    bool     cells that are mostly water
         outside    [H, W]    bool     cells that are mostly out of bound
         street     [H, W]    bool     cells with enough street to carry a road link
         edge_index [2, E]    int64    4-neighbour links between street cells (road graph, proposed)
         cell_m     float             metres per coarse cell (row direction)

Run from the project root:
    python -m encoders.map_adapter --labels map_grid.json --grid 0 0 --out data/map_city.npz --plot data/map_city.png
    (--grid 0 0 picks H, W automatically: about 400 cells with the map's own aspect ratio;
     --grid 20 20 forces 20 x 20, which makes cells non-square on a wide map)

Imported maps have no satellite tiles, so they carry NO MAE embeddings: agents use the
class-fraction channels (embedding_source = none). Say so in the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# observed channels (order is part of the contract once agreed)
CHANNELS = ["street", "green", "house", "building", "commercial", "water", "terrain", "outside"]
STREET, GREEN, HOUSE, BUILDING, COMMERCIAL, WATER, TERRAIN, OUTSIDE = range(8)

# app label names (lower-cased) -> channel. Add aliases here when the app gets new labels.
NAME_TO_CHANNEL = {
    "street": STREET, "road": STREET, "streets": STREET,
    "green": GREEN, "park": GREEN, "grass": GREEN, "forest": GREEN,
    "house": HOUSE, "houses": HOUSE, "residential": HOUSE,
    "building": BUILDING, "buildings": BUILDING, "built": BUILDING,
    "commercial": COMMERCIAL, "shop": COMMERCIAL, "retail": COMMERCIAL,
    "water": WATER, "river": WATER, "lake": WATER,
    "terrain": TERRAIN, "land": TERRAIN, "empty": TERRAIN,
    "out of bound": OUTSIDE, "out_of_bound": OUTSIDE, "outside": OUTSIDE, "out": OUTSIDE,
}

# zoning ids shared with the contract: 0 residential, 1 commercial, 2 industrial, 3 green, 4 mixed.
# ASSUMPTION to confirm with the app owner: 'Building' (generic) is treated as mixed.
ZONE_OF_CHANNEL = {HOUSE: 0, COMMERCIAL: 1, GREEN: 3, BUILDING: 4}


def names_to_channels(cells) -> np.ndarray:
    """Rows of label names (None = unpainted = terrain) -> [h, w] channel ids; fail loudly on unknown names."""
    h, w = len(cells), len(cells[0])
    out = np.zeros((h, w), dtype=np.int64)
    for r, row in enumerate(cells):
        assert len(row) == w, "ragged grid"
        for c, name in enumerate(row):
            key = "terrain" if name is None else str(name).strip().lower()
            if key not in NAME_TO_CHANNEL:
                raise ValueError(f"label {name!r} is not in NAME_TO_CHANNEL; add an alias")
            out[r, c] = NAME_TO_CHANNEL[key]
    return out


def ids_to_channels(ids: np.ndarray, legend: dict) -> np.ndarray:
    lut = {}
    for name, cid in legend.items():
        key = name.strip().lower()
        if key not in NAME_TO_CHANNEL:
            raise ValueError(f"legend class {name!r} is not in NAME_TO_CHANNEL; add an alias")
        lut[int(cid)] = NAME_TO_CHANNEL[key]
    unknown = set(np.unique(ids).tolist()) - set(lut)
    if unknown:
        raise ValueError(f"ids {sorted(unknown)} appear in the grid but not in the legend")
    out = np.zeros(ids.shape, dtype=np.int64)
    for cid, ch in lut.items():
        out[ids == cid] = ch
    return out


def coarsen(ch: np.ndarray, H: int, W: int) -> np.ndarray:
    """[h, w] channel ids -> [H, W, K] fractions (every fine cell counted once)."""
    h, w = ch.shape
    assert H <= h and W <= w, "coarse grid cannot be finer than the app grid"
    rb = (np.arange(h) * H) // h
    cb = (np.arange(w) * W) // w
    onehot = np.eye(len(CHANNELS), dtype=np.float32)[ch]            # [h, w, K]
    sums = np.zeros((H, W, len(CHANNELS)), dtype=np.float32)
    np.add.at(sums, (rb[:, None], cb[None, :]), onehot)
    counts = sums.sum(axis=-1, keepdims=True)
    return (sums / np.maximum(counts, 1.0)).astype(np.float32)


def derive(frac: np.ndarray, street_thr: float = 0.15, zone_thr: float = 0.4):
    """Masks, existing land use and street cells. All thresholds are proposals: tune and document them."""
    outside = frac[..., OUTSIDE] > 0.5
    water = (frac[..., WATER] > 0.5) & ~outside
    chans = list(ZONE_OF_CHANNEL)
    cand = np.stack([frac[..., c] for c in chans], axis=-1)
    zone_ids = np.array([ZONE_OF_CHANNEL[c] for c in chans], dtype=np.int64)
    land_use = np.where(cand.max(-1) >= zone_thr, zone_ids[cand.argmax(-1)], -1).astype(np.int64)
    land_use[outside | water] = -1
    street = (frac[..., STREET] >= street_thr) & ~outside & ~water
    return outside, water, land_use, street


def street_graph(street: np.ndarray) -> np.ndarray:
    """edge_index [2, E] between 4-neighbouring street cells, flat index k = i * W + j, both directions."""
    H, W = street.shape
    idx = np.arange(H * W).reshape(H, W)
    src, dst = [], []
    for a, b, m in ((idx[:, :-1], idx[:, 1:], street[:, :-1] & street[:, 1:]),
                    (idx[:-1, :], idx[1:, :], street[:-1, :] & street[1:, :])):
        src.append(a[m]); dst.append(b[m])
    s, d = np.concatenate(src), np.concatenate(dst)
    return np.stack([np.concatenate([s, d]), np.concatenate([d, s])]).astype(np.int64)


def adapt(ch: np.ndarray, fine_cell_m: float, H: int, W: int) -> dict:
    frac = coarsen(ch, H, W)
    outside, water, land_use, street = derive(frac)
    return {
        "fractions": frac, "land_use": land_use, "water": water, "outside": outside,
        "street": street, "edge_index": street_graph(street),
        "cell_m": np.float32(fine_cell_m * ch.shape[0] / H),
    }


def load_input(labels_path: str, legend_path: str = ""):
    """Returns (channel grid [h, w], metres per fine cell or None)."""
    p = Path(labels_path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        cells = data["cells"] if isinstance(data, dict) else data
        mpp = data.get("meters_per_cell") if isinstance(data, dict) else None
        return names_to_channels(cells), mpp
    if not legend_path:
        raise ValueError("an .npy label file needs --legend legend.json")
    return ids_to_channels(np.load(p), json.loads(Path(legend_path).read_text())), None


def save_plot(city: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    lu = city["land_use"]
    shown = np.zeros(lu.shape, dtype=float)          # 0 open / unassigned
    shown[lu == 0] = 1                               # residential
    shown[lu == 1] = 2                               # commercial
    shown[lu == 4] = 3                               # mixed (generic building)
    shown[lu == 3] = 4                               # green
    shown[city["water"]] = 5                         # water
    shown[city["outside"]] = np.nan                  # out of bound
    cmap = ListedColormap(["#bbbbbb", "#e8b23a", "#7d5bbe", "#d3543c", "#2e9e4f", "#3a7bd5"]).with_extremes(bad="#111111")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].imshow(shown, cmap=cmap, vmin=0, vmax=5)
    ax[0].set_title("existing land use (open, res, commercial, mixed, green, water)", fontsize=8)
    ax[1].imshow(city["fractions"][..., STREET], cmap="gray_r"); ax[1].set_title("street fraction", fontsize=8)
    built = city["fractions"][..., [HOUSE, BUILDING, COMMERCIAL]].sum(-1)
    ax[2].imshow(built, cmap="gray_r"); ax[2].set_title("built-up fraction", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="map_grid.json from the app, or a .npy of class ids")
    ap.add_argument("--legend", default="", help=".json {class name: id}, only for .npy input")
    ap.add_argument("--fine-cell-m", type=float, default=0.0, help="metres per app cell (overrides the JSON value)")
    ap.add_argument("--grid", type=int, nargs=2, default=[20, 20], metavar=("H", "W"),
                    help="coarse grid; 0 0 = automatic (about 400 cells, same aspect ratio)")
    ap.add_argument("--out", default="data/map_city.npz")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()

    ch, mpp = load_input(args.labels, args.legend)
    fine_m = args.fine_cell_m or mpp
    if not fine_m:
        raise ValueError("metres per app cell unknown: put meters_per_cell in the JSON or pass --fine-cell-m")
    h, w = ch.shape
    H, W = args.grid
    if H == 0 or W == 0:
        W = max(1, int(round((400 * w / h) ** 0.5)))
        H = max(1, int(round(400 / W)))
    city = adapt(ch, float(fine_m), H, W)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **city)
    f = city["fractions"]
    print(f"app grid {h} x {w} at {fine_m} m  ->  coarse grid {H} x {W}  ({city['cell_m']:.0f} m per coarse cell, row direction)")
    print("mean share per channel:", {c: round(float(f[..., i].mean()), 3) for i, c in enumerate(CHANNELS)})
    print(f"outside cells {int(city['outside'].sum())} | water cells {int(city['water'].sum())} | "
          f"street cells {int(city['street'].sum())} | street links {city['edge_index'].shape[1]}")
    names = {0: "residential", 1: "commercial", 3: "green", 4: "mixed"}
    print("existing land use:", {n: int((city['land_use'] == i).sum()) for i, n in names.items()},
          "| open", int((city['land_use'] == -1).sum()))
    print(f"saved {args.out}")
    if args.plot:
        save_plot(city, args.plot)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()