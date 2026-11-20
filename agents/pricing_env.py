"""Stage 3 PricingEnv: discrete transit fare and congestion charge control."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from agents.stage34_sim import Stage34Config, StandInCityState, StandInSimulator


class PricingEnv(gym.Env):
    """Gymnasium environment for Stage 3 transit fare and congestion pricing."""

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
        self._last_stats: dict[str, float] = {}

        n_fare = len(self.config.fare_levels)
        n_charge = len(self.config.congestion_levels)
        self.action_space = gym.spaces.MultiDiscrete(np.array([n_fare, n_charge], dtype=np.int64))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

    def action_masks(self) -> np.ndarray:
        n_fare = len(self.config.fare_levels)
        n_charge = len(self.config.congestion_levels)
        return np.ones(n_fare + n_charge, dtype=bool)

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        s = self.state
        num_edges = float(s.edge_index.shape[1])
        max_fare = max(1e-6, max(self.config.fare_levels))
        max_charge = max(1e-6, max(self.config.congestion_levels))
        total_demand = float(s.demand.sum())
        pop_sum = max(1.0, float(s.population.sum()))

        return np.array(
            [
                num_edges / float(max(1, self.config.max_edges)),
                float(np.mean(s.edge_time_min)) / 15.0,
                float(np.mean(s.edge_cost)) / 2.0,
                float(np.mean(s.asset_condition)),
                total_demand / pop_sum,
                self._last_stats.get("transit_share", 0.5),
                self._last_stats.get("avg_travel_time", 10.0) / 15.0,
                self._last_stats.get("emissions_proxy", 0.5),
                s.fare / max_fare,
                s.congestion_charge / max_charge,
                s.budget,
                float(self.step_count) / float(max(1, self.config.pricing_steps)),
            ],
            dtype=np.float32,
        )

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
        self.state, self._last_stats = self.sim.evaluate_pricing(
            self.state, self.state.fare, self.state.congestion_charge
        )
        return self._get_obs(), {"data_label": self.config.data_label}

    def step(
        self, action: np.ndarray | list[int] | tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.state is not None, "Call reset() before step()."
        act = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(act):
            raise ValueError(f"Invalid PricingEnv action: {action}")

        fare = float(self.config.fare_levels[int(act[0])])
        charge = float(self.config.congestion_levels[int(act[1])])
        self.state, stats = self.sim.evaluate_pricing(self.state, fare, charge)
        self._last_stats = stats
        self.step_count += 1

        r_ridership = float(np.clip(stats["transit_share"], 0.0, 1.0))
        r_emissions = float(np.clip(1.0 - stats["emissions_proxy"], 0.0, 1.0))
        r_revenue = float(np.clip(stats["step_revenue"] / 0.10, -1.0, 1.0))
        r_equity = float(-np.clip(stats["burden_gap"], 0.0, 1.0))

        reward = float(
            0.35 * r_ridership + 0.25 * r_emissions + 0.20 * r_revenue + 0.20 * r_equity
        )
        terminated = bool(self.step_count >= self.config.pricing_steps)
        info: dict[str, Any] = {
            "reward_terms": {
                "ridership": r_ridership,
                "emissions": r_emissions,
                "revenue": r_revenue,
                "equity": r_equity,
            },
            "fare": fare,
            "congestion_charge": charge,
            "budget": float(self.state.budget),
            "data_label": self.config.data_label,
        }
        return self._get_obs(), reward, terminated, False, info
