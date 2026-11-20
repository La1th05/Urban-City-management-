"""Stage 4 MaintenanceEnv: budget-constrained asset inspection and repair."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from agents.stage34_sim import Stage34Config, StandInCityState, StandInSimulator


class MaintenanceEnv(gym.Env):
    """Gymnasium environment for Stage 4 infrastructure maintenance scheduling."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: StandInSimulator | None = None,
        config: Stage34Config | None = None,
        initial_state: StandInCityState | None = None,
    ) -> None:
        super().__init__()
        self.config = config or Stage34Config()
        self.sim = simulator or StandInSimulator(self.config)
        self._base_state = initial_state
        self.state: StandInCityState | None = None
        self.step_count = 0

        e_max = self.config.max_edges
        self.action_space = gym.spaces.Discrete(e_max + 1)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(2 * e_max + 2,), dtype=np.float32
        )

    def action_masks(self) -> np.ndarray:
        assert self.state is not None, "Call reset() before action_masks()."
        e_max = self.config.max_edges
        mask = np.zeros(e_max + 1, dtype=bool)
        mask[0] = True

        num_edges = min(e_max, self.state.edge_index.shape[1])
        repair_costs = self.sim.repair_cost_vector(self.state)
        for idx in range(num_edges):
            needs_repair = bool(self.state.asset_condition[idx] < 0.99)
            affordable = bool(self.state.budget >= float(repair_costs[idx]) - 1e-8)
            mask[idx + 1] = needs_repair and affordable
        return mask

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        e_max = self.config.max_edges
        num_edges = min(e_max, self.state.edge_index.shape[1])

        cond_pad = np.zeros(e_max, dtype=np.float32)
        load_pad = np.zeros(e_max, dtype=np.float32)
        cond_pad[:num_edges] = self.state.asset_condition[:num_edges]
        load_pad[:num_edges] = self.sim.compute_usage_load(self.state)[:num_edges]

        rem_steps = float(max(0, self.config.maintenance_steps - self.step_count)) / float(
            max(1, self.config.maintenance_steps)
        )
        tail = np.array([float(self.state.budget), rem_steps], dtype=np.float32)
        return np.concatenate([cond_pad, load_pad, tail], axis=0).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        city_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.state = (
            self._base_state.copy()
            if self._base_state is not None
            else self.sim.make_dummy_city(city_seed)
        )
        self.step_count = 0
        return self._get_obs(), {"data_label": self.config.data_label}

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.state is not None, "Call reset() before step()."
        act = int(action)
        mask = self.action_masks()
        if act < 0 or act >= mask.shape[0] or not bool(mask[act]):
            raise ValueError(f"Action {act} is invalid or masked out.")

        self.state, stats = self.sim.step_maintenance(self.state, act)
        self.step_count += 1

        r_health = float(np.clip(stats["weighted_health"], 0.0, 1.0))
        r_failure = float(-np.clip(stats["critical_frac"], 0.0, 1.0))
        r_cost = float(-np.clip(stats["step_cost"] / 0.15, 0.0, 1.0))

        reward = float(0.60 * r_health + 0.25 * r_failure + 0.15 * r_cost)
        terminated = bool(
            self.step_count >= self.config.maintenance_steps or self.state.budget <= 1e-6
        )
        info: dict[str, Any] = {
            "reward_terms": {
                "health": r_health,
                "failure_avoidance": r_failure,
                "cost": r_cost,
            },
            "step_cost": float(stats["step_cost"]),
            "budget": float(self.state.budget),
            "data_label": self.config.data_label,
        }
        return self._get_obs(), reward, terminated, False, info
