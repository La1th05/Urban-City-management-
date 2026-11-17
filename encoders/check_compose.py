import hashlib
from collections import Counter, defaultdict
import numpy as np
from common import get_settings
from encoders.datapipeline import load_eurosat_arrays
from encoders.city_tiles import make_land_use_map, compose_city_from_eurosat

s = get_settings()
KINDS = ["residential", "commercial", "industrial", "green", "mixed"]

def digest(a):
    return hashlib.md5(a.tobytes()).digest()

def main():
    images, labels = load_eurosat_arrays()
    land, water = make_land_use_map(seed=0)
    tiles = compose_city_from_eurosat(images, labels, land, water, seed=0)
    source = {digest(images[k]): int(labels[k]) for k in range(len(images))}

    counts = defaultdict(Counter)
    for i in range(s.H):
        for j in range(s.W):
            t = tiles[i, j]
            if not t.any():
                src = "ALL ZEROS"
            else:
                d = digest(t)
                src = s.EUROSAT_CLASSES[source[d]] if d in source else "NOT FOUND"
            kind = "water" if water[i, j] else KINDS[int(land[i, j])]
            counts[kind][src] += 1

    for kind, c in counts.items():
        print(kind, dict(c))

if __name__ == "__main__":
    main()