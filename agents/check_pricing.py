"""Interface test for Stage 3 PricingEnv (mirrors check_zoning.py)."""

from __future__ import annotations

import numpy as np

from agents.pricing_env import PricingEnv


def check_pricing_interface(seed: int = 7) -> None:
    env = PricingEnv()
    obs1, info1 = env.reset(seed=seed)
    assert obs1.shape == env.observation_space.shape
    assert obs1.dtype == np.float32
    assert isinstance(info1, dict)

    rng = np.random.default_rng(seed)
    nvec = env.action_space.nvec
    offsets = np.cumsum(np.concatenate(([0], nvec[:-1])))

    actions, rewards1, obs_seq1 = [], [], [obs1]
    done = False
    while not done:
        mask = env.action_masks()
        assert mask.dtype == np.bool_
        action = np.array(
            [
                int(rng.choice(np.flatnonzero(mask[offsets[i] : offsets[i] + nvec[i]])))
                for i in range(len(nvec))
            ],
            dtype=np.int64,
        )
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
        assert isinstance(reward, float) and np.isfinite(reward)
        assert "reward_terms" in info and isinstance(info["reward_terms"], dict)
        obs_seq1.append(obs)
        rewards1.append(reward)
        done = bool(terminated or truncated)

    obs2, _ = env.reset(seed=seed)
    np.testing.assert_allclose(obs1, obs2, rtol=0.0, atol=1e-6)
    for idx, action in enumerate(actions):
        obs_next, rew_next, _, _, _ = env.step(action)
        np.testing.assert_allclose(obs_seq1[idx + 1], obs_next, rtol=0.0, atol=1e-6)
        assert abs(rewards1[idx] - rew_next) < 1e-6


if __name__ == "__main__":
    check_pricing_interface()
    print("[PASS] PricingEnv interface test passed.")
