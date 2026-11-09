"""Basic deterministic interface checks for the Day 1 ZoningEnv scaffold.

From project root:
    python -m encoders.stub_embeddings
    python -m agents.check_zoning
"""
import numpy as np

from common import get_settings
from agents.zoning_env import ZoningEnv


def main():
    s = get_settings()
    env = ZoningEnv()
    first, first_info = env.reset(seed=42)
    second, second_info = env.reset(seed=42)

    assert np.array_equal(first["water"], second["water"])
    assert first["cursor"] == second["cursor"]
    assert first_info == second_info
    assert first["embeddings"].shape == (s.H, s.W, s.D)
    assert first["embeddings"].dtype == np.float32
    assert first["land_use"].shape == (s.H, s.W)
    assert first["land_use"].dtype == np.int64
    assert first["water"].dtype == np.bool_
    assert np.all(first["land_use"] == -1)
    assert env.observation_space.contains(first)

    rng = np.random.default_rng(42)
    expected_cells = int(np.count_nonzero(~first["water"]))
    assigned_cells = []
    done = False
    for _ in range(expected_cells):
        mask = env.action_masks()
        assert mask.dtype == np.bool_ and mask.shape == (5,) and mask.any()
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float) and np.isfinite(reward)
        assert isinstance(info["reward_terms"], dict)
        assert not truncated
        assigned_cells.append(info["assigned_cell"])
        done = terminated

    assert done
    assert len(assigned_cells) == expected_cells
    assert len(set(assigned_cells)) == expected_cells
    assert np.array_equal(obs["water"], first["water"])
    assert np.all(obs["land_use"][~obs["water"]] >= 0)
    assert np.all(obs["land_use"][obs["water"]] == -1)
    assert not env.action_masks().any()
    print("PASS: seeded reset, shapes/dtypes, masks, full random episode")
    print(f"Assigned {expected_cells} non-water cells (synthetic, stub embeddings)")
    print("NOTE: reward is placeholder 0.0; no RL training conclusions")


if __name__ == "__main__":
    main()
