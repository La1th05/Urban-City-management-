import numpy as np
from common import get_settings
from encoders.city_tiles import (cut_into_tiles, stitch_tiles, cell_to_pixel_box,
                                 pixel_to_cell, make_synthetic_city, GRN)
from encoders.datapipeline import stratified_split

s = get_settings()

def main():
    H, W, T, C = s.H, s.W, s.T, s.C
    rng = np.random.default_rng(0)

    # 1. cut/stitch round trip and cell-to-pixel mapping
    raster = rng.random((C, H * T, W * T), dtype=np.float32)
    tiles = cut_into_tiles(raster)
    assert tiles.shape == (H, W, C, T, T)
    assert np.array_equal(stitch_tiles(tiles), raster)
    for i, j in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1), (3, 17)]:
        r0, r1, c0, c1 = cell_to_pixel_box(i, j)
        assert np.array_equal(tiles[i, j], raster[:, r0:r1, c0:c1])
        assert pixel_to_cell(r0, c0) == (i, j)
        assert pixel_to_cell(r1 - 1, c1 - 1) == (i, j)

    # 2. split: reproducible, disjoint, complete
    labels = rng.integers(0, 6, 3200)
    a = stratified_split(labels, 0.1, 0)
    b = stratified_split(labels, 0.1, 0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(np.intersect1d(*a)) == 0
    assert len(a[0]) + len(a[1]) == len(labels)

    # 3. synthetic city: tile content matches its labels
    city = make_synthetic_city(seed=0)
    assert city["tiles"].shape == (H, W, C, T, T)
    assert city["water"].sum() > 0
    w = city["tiles"][city["water"]].mean(axis=(0, 2, 3))
    assert w[2] > w[0], "water tiles should be blue-dominant"
    g = city["tiles"][(city["land_use"] == GRN) & ~city["water"]].mean(axis=(0, 2, 3))
    assert g[1] > g[0], "green tiles should be green-dominant"
    print("all checks passed")

if __name__ == "__main__":
    main()