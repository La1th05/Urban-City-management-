"""Classical baselines for Stage 3 PricingEnv (mirrors zoning_baselines.py)."""

from __future__ import annotations

import numpy as np

from agents.pricing_env import PricingEnv
from agents.stage34_sim import Stage34Config


class FixedPricePolicy:
    """Selects fixed (fare_idx, congestion_charge_idx) levels every step."""

    def __init__(self, fare_idx: int = 2, charge_idx: int = 2) -> None:
        self.action = np.array([fare_idx, charge_idx], dtype=np.int64)

    def act(self, env: PricingEnv) -> np.ndarray:
        return self.action.copy()


class GridSearchPricingPolicy:
    """Evaluates all discrete (fare, charge) pairs on the current state and picks the best."""

    def act(self, env: PricingEnv) -> np.ndarray:
        assert env.state is not None
        best_score = -np.inf
        best_pair = np.array([0, 0], dtype=np.int64)
        n_fare, n_charge = env.action_space.nvec

        for f_idx in range(int(n_fare)):
            for c_idx in range(int(n_charge)):
                probe = PricingEnv(
                    simulator=env.sim, config=env.config, initial_state=env.state.copy()
                )
                probe.reset(seed=0)
                _, reward, _, _, _ = probe.step([f_idx, c_idx])
                if reward > best_score:
                    best_score = reward
                    best_pair = np.array([f_idx, c_idx], dtype=np.int64)
        return best_pair


def run_pricing_baselines(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, tuple[float, float]]:
    cfg = Stage34Config()
    policies = {
        "Pricing: Fixed": FixedPricePolicy(fare_idx=2, charge_idx=2),
        "Pricing: GridSearch": GridSearchPricingPolicy(),
    }
    results: dict[str, tuple[float, float]] = {}
    for name, pol in policies.items():
        returns = []
        for s in seeds:
            env = PricingEnv(config=cfg)
            env.reset(seed=s)
            total, done = 0.0, False
            while not done:
                _, r, term, trunc, _ = env.step(pol.act(env))
                total += float(r)
                done = bool(term or trunc)
            returns.append(total)
        results[name] = (float(np.mean(returns)), float(np.std(returns)))
    return results


if __name__ == "__main__":
    for name, (mean_r, std_r) in run_pricing_baselines().items():
        print(f"{name:22s} | label=synthetic | return={mean_r:+.4f} +/- {std_r:.4f}")
