"""Classical baselines for Stage 4 MaintenanceEnv (mirrors transit_baselines.py)."""

from __future__ import annotations

import numpy as np

from agents.maintenance_env import MaintenanceEnv
from agents.stage34_sim import Stage34Config


class DoNothingMaintenancePolicy:
    """Baseline that never repairs any asset."""

    def act(self, env: MaintenanceEnv) -> int:
        return 0


class FixedScheduleMaintenancePolicy:
    """Repairs edges on a deterministic round-robin schedule every interval steps."""

    def __init__(self, interval: int = 2) -> None:
        self.interval = max(1, interval)

    def act(self, env: MaintenanceEnv) -> int:
        assert env.state is not None
        if env.step_count % self.interval != 0:
            return 0
        mask = env.action_masks()
        num_edges = min(env.config.max_edges, env.state.edge_index.shape[1])
        if num_edges == 0:
            return 0
        start = (env.step_count // self.interval) % num_edges
        for offset in range(num_edges):
            edge_idx = (start + offset) % num_edges
            if bool(mask[edge_idx + 1]):
                return int(edge_idx + 1)
        return 0


class WorstFirstMaintenancePolicy:
    """Always repairs the lowest-condition affordable edge (0 if none eligible)."""

    def act(self, env: MaintenanceEnv) -> int:
        assert env.state is not None
        mask = env.action_masks()
        num_edges = min(env.config.max_edges, env.state.edge_index.shape[1])
        best_action = 0
        lowest_cond = 1.0
        for edge_idx in range(num_edges):
            if bool(mask[edge_idx + 1]):
                cond = float(env.state.asset_condition[edge_idx])
                if cond < lowest_cond:
                    lowest_cond = cond
                    best_action = edge_idx + 1
        return best_action


def run_maintenance_baselines(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, tuple[float, float]]:
    cfg = Stage34Config()
    policies = {
        "Maint: DoNothing": DoNothingMaintenancePolicy(),
        "Maint: FixedSched": FixedScheduleMaintenancePolicy(interval=2),
        "Maint: WorstFirst": WorstFirstMaintenancePolicy(),
    }
    results: dict[str, tuple[float, float]] = {}
    for name, pol in policies.items():
        returns = []
        for s in seeds:
            env = MaintenanceEnv(config=cfg)
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
    for name, (mean_r, std_r) in run_maintenance_baselines().items():
        print(f"{name:22s} | label=synthetic | return={mean_r:+.4f} +/- {std_r:.4f}")
