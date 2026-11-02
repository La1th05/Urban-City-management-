"""
city_tiles.py  --  grid cell <-> image tile mapping (NumPy only, no torch).

THE CONTRACT
------------
City grid          : H x W cells (default 20 x 20).
One cell           : exactly one image tile of T x T pixels (default 64 x 64).
Tile size on ground: Sentinel-2 RGB is 10 m/pixel, so 64 px = 640 m per cell
                     (close to the ~500 m cell size in the project plan).
City raster        : shape [C, H*T, W*T]  (e.g. [3, 1280, 1280]).
Cell (i, j)        : i = row (north -> south), j = column (west -> east).
Cell (i, j) pixels : rows [i*T, (i+1)*T), cols [j*T, (j+1)*T).
Tile array         : [H, W, C, T, T], so tiles[i, j] is the image of cell (i, j).
Embeddings         : [H, W, D] float32, same (i, j) order as the tiles.

Land use ids (the 5 zoning classes used by the simulator):
    0 residential, 1 commercial, 2 industrial, 3 green, 4 mixed
Water is NOT a land-use class; it is a blocked mask (cells where nothing can be built).
"""
from __future__ import annotations

import numpy as np
from common import Settings, get_settings

settings = get_settings()

RES, COM, IND, GRN, MIX = range(5)

# EuroSAT class names, in torchvision's alphabetical order (index = label id).

# EuroSAT label -> (land-use id or None, is_water).
# EuroSAT has no "commercial" or "mixed" class, and Highway is a road, not a zone.
# Highway is treated as "mixed" (urban fabric) as a documented simplification.
EUROSAT_TO_LAND_USE = {
    "AnnualCrop": (GRN, False), "Forest": (GRN, False),
    "HerbaceousVegetation": (GRN, False), "Pasture": (GRN, False),
    "PermanentCrop": (GRN, False), "Industrial": (IND, False),
    "Residential": (RES, False), "Highway": (MIX, False),
    "River": (None, True), "SeaLake": (None, True),
}

def cell_to_pixel_box(i: int, j: int, tile: int = settings.T):
    """Return (row_start, row_end, col_start, col_end) of cell (i, j)."""
    return i * tile, (i + 1) * tile, j * tile, (j + 1) * tile


def pixel_to_cell(row: int, col: int, tile: int = settings.T):
    """Return the (i, j) cell containing pixel (row, col)."""
    return row // tile, col // tile


def cut_into_tiles(raster: np.ndarray, h: int = settings.H, w: int = settings.W, tile: int = settings.T):
    """[C, h*tile, w*tile] -> [h, w, C, tile, tile]."""
    c, rh, rw = raster.shape
    assert rh == h * tile and rw == w * tile, (
        f"raster {raster.shape} does not match grid {h}x{w} with tile {tile}")
    return raster.reshape(c, h, tile, w, tile).transpose(1, 3, 0, 2, 4).copy()


def stitch_tiles(tiles: np.ndarray):
    """Inverse of cut_into_tiles: [h, w, C, t, t] -> [C, h*t, w*t]."""
    h, w, c, t, _ = tiles.shape
    return tiles.transpose(2, 0, 3, 1, 4).reshape(c, h * t, w * t)


# ----------------------------------------------------------------------------
# Synthetic city (fallback + pipeline testing; no downloads needed)
# ----------------------------------------------------------------------------
def make_land_use_map(h: int = settings.H, w: int =settings.W, seed: int = 0):
    """Smooth random land-use map [h, w] (ints 0..4) plus a water mask [h, w] (bool)."""
    rng = np.random.default_rng(seed)
    # Coarse random field, upsampled by repetition so neighbouring cells correlate.
    coarse = rng.integers(0, 5, size=(h // 4 + 1, w // 4 + 1))
    land = np.kron(coarse, np.ones((4, 4), dtype=int))[:h, :w]
    # Commercial core near the centre.
    yy, xx = np.mgrid[:h, :w]
    core = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (min(h, w) / 6) ** 2
    land[core] = COM
    # A diagonal river as the water mask.
    water = np.abs((yy - xx * h / w) - h / 4) < 0.8
    return land.astype(np.int64), water


# Base RGB colour and texture strength per land-use class (rough satellite look).
_BASE_RGB = {
    RES: (0.55, 0.50, 0.45), COM: (0.65, 0.65, 0.68), IND: (0.50, 0.50, 0.55),
    GRN: (0.20, 0.45, 0.20), MIX: (0.58, 0.55, 0.50),
}
_TEXTURE = {RES: 0.12, COM: 0.18, IND: 0.15, GRN: 0.07, MIX: 0.14}
_WATER_RGB = (0.12, 0.25, 0.45)


def render_synthetic_tile(land_use: int, is_water: bool, rng, tile: int = settings.T):
    """One synthetic [3, tile, tile] float32 tile in [0, 1]."""
    base = np.array(_WATER_RGB if is_water else _BASE_RGB[land_use], dtype=np.float32)
    amp = 0.03 if is_water else _TEXTURE[land_use]
    noise = rng.standard_normal((1, tile, tile)).astype(np.float32) * amp
    img = base[:, None, None] + noise
    if not is_water and land_use in (RES, COM, IND, MIX):  # faint grid pattern = streets
        period = {RES: 8, COM: 16, IND: 32, MIX: 12}[land_use]
        img[:, ::period, :] *= 0.8
        img[:, :, ::period] *= 0.8
    return np.clip(img, 0.0, 1.0)


def make_synthetic_city(h: int = settings.H, w: int = settings.W, tile: int = settings.T, seed: int = 0):
    """Return dict(land_use [h,w], water [h,w], tiles [h,w,3,tile,tile])."""
    rng = np.random.default_rng(seed)
    land, water = make_land_use_map(h, w, seed)
    tiles = np.zeros((h, w, settings.C, tile, tile), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            tiles[i, j] = render_synthetic_tile(land[i, j], bool(water[i, j]), rng, tile)
    return {"land_use": land, "water": water, "tiles": tiles}


def compose_city_from_eurosat(images: np.ndarray, labels: np.ndarray,
                              land_use: np.ndarray, water: np.ndarray, seed: int = 0):
    """Fill every cell with a random real EuroSAT patch whose class matches the cell.

    images : [N, 3, 64, 64] float32 in [0, 1]    labels : [N] EuroSAT label ids
    land_use, water : the [h, w] maps from make_land_use_map (or your own).
    Returns tiles [h, w, 3, 64, 64]. Real imagery, known ground-truth land use.
    """
    rng = np.random.default_rng(seed)
    by_target = {}  # (land_use or "water") -> indices of matching EuroSAT patches
    for idx, name in enumerate(settings.EUROSAT_CLASSES):
        lu, is_water = EUROSAT_TO_LAND_USE[name]
        key = "water" if is_water else lu
        by_target.setdefault(key, []).append(idx)
    pools = {k: np.flatnonzero(np.isin(labels, v)) for k, v in by_target.items()}

    h, w = land_use.shape
    tiles = np.zeros((h, w) + images.shape[1:], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            key = "water" if water[i, j] else int(land_use[i, j])
            pool = pools.get(key)
            if pool is None or len(pool) == 0:  # e.g. commercial: no EuroSAT equivalent
                pool = pools[MIX]               # fall back to urban-fabric patches
            tiles[i, j] = images[rng.choice(pool)]
    return tiles