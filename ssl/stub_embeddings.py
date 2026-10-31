import numpy as np
from pathlib import Path

#Height, Weight , Depth(Dimension)
H,W,D=20,20,64
SEED=0


def make_stub(path="data/embeddings_stub.npy"):
    rng = np.random.default_rng(seed=SEED)
    #Generate random values and cast them to float
    emb= rng.standard_normal((H,W,D)).astype(np.float32)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.save(path,emb)
    return emb

def get_embeddings(path="data/embeddings_stub.npy"):
    emb= np.load(path)
    #vertify data shape and type
    assert emb.shape == (H,W,D) and emb.dtype== np.float32
    return emb

if __name__ == "__main__":
    print(make_stub().shape)