import numpy as np
import matplotlib.pyplot as plt
from encoders.datapipeline import load_eurosat_arrays
from common import get_settings

if __name__ == "__main__":
    images, labels = load_eurosat_arrays()
    names = get_settings().EUROSAT_CLASSES
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for c, ax in enumerate(axes.ravel()):
        k = np.flatnonzero(labels == c)[0]
        ax.imshow(images[k].transpose(1, 2, 0))
        ax.set_title(names[c], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("data/eurosat_samples.png", dpi=120)