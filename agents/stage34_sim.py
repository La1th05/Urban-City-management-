"""Contract-compliant stand-in CityState and vectorized simulator for Stages 3-4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import get_settings


def load_core_settings() -> dict[str, int]:
    """Read H, W, T, C, D from common.get_settings() without hardcoding."""
    raw = get_settings()
    keys = ("H", "W", "T", "C", "D")
    return {k: int(raw[k]) if isinstance(raw, dict) else int(getattr(raw, k)) for k in keys}


@dataclass(frozen=True)
class Stage34Config:
    """Configuration for PricingEnv, MaintenanceEnv, and stand-in dynamics."""

    fare_levels: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
    congestion_levels: tuple[float, ...] = (0.0, 2.0, 5.0, 8.0, 12.0)
    max_edges: int = 32
    pricing_steps: int = 5
    maintenance_steps: int = 10
    base_repair_cost: float = 0.08
    wear_rate: float = 0.04
    data_label: str = "synthetic"


@dataclass
class StandInCityState:
    """Stand-in matching Shared Contract Section 6 field names, shapes, and dtypes."""

    land_use: np.ndarray          # [H, W] int64 in 0..C-1
    water: np.ndarray             # [H, W] bool
    population: np.ndarray        # [H, W] float32
    jobs: np.ndarray              # [H, W] float32
    demand: np.ndarray            # [H*W, H*W] float32
    edge_index: np.ndarray        # [2, E] int64
    edge_time_min: np.ndarray     # [E] float32
    edge_cost: np.ndarray         # [E] float32
    fare: float
    congestion_charge: float
    asset_condition: np.ndarray   # [E] float32 in [0, 1]
    budget: float

    def copy(self) -> StandInCityState:
        return StandInCityState(
            land_use=self.land_use.copy(),
            water=self.water.copy(),
            population=self.population.copy(),
            jobs=self.jobs.copy(),
            demand=self.demand.copy(),
            edge_index=self.edge_index.copy(),
            edge_time_min=self.edge_time_min.copy(),
            edge_cost=self.edge_cost.copy(),
            fare=float(self.fare),
            congestion_charge=float(self.congestion_charge),
            asset_condition=self.asset_condition.copy(),
            budget=float(self.budget),
        )

    def validate(self) -> None:
        cfg = load_core_settings()
        h, w, c = cfg["H"], cfg["W"], cfg["C"]
        n_cells = h * w
        num_edges = self.edge_index.shape[1]

        assert self.land_use.shape == (h, w) and self.land_use.dtype == np.int64
        assert np.all((self.land_use >= 0) & (self.land_use < c))
        assert self.water.shape == (h, w) and self.water.dtype == np.bool_
        assert self.population.shape == (h, w) and self.population.dtype == np.float32
        assert self.jobs.shape == (h, w) and self.jobs.dtype == np.float32
        assert self.demand.shape == (n_cells, n_cells) and self.demand.dtype == np.float32
        assert self.edge_index.shape == (2, num_edges) and self.edge_index.dtype == np.int64
        assert self.edge_time_min.shape == (num_edges,) and self.edge_time_min.dtype == np.float32
        assert self.edge_cost.shape == (num_edges,) and self.edge_cost.dtype == np.float32
        assert self.asset_condition.shape == (num_edges,) and self.asset_condition.dtype == np.float32
        assert np.all((self.asset_condition >= 0.0) & (self.asset_condition <= 1.0))
        assert np.isfinite(self.fare) and np.isfinite(self.congestion_charge) and np.isfinite(self.budget)


class StandInSimulator:
    """Lightweight vectorized city dynamics until Member 2's simulator is merged."""

    def __init__(self, config: Stage34Config | None = None) -> None:
        self.config = config or Stage34Config()
        self.settings = load_core_settings()

    def make_dummy_city(self, seed: int) -> StandInCityState:
        rng = np.random.default_rng(seed)
        h, w, c = self.settings["H"], self.settings["W"], self.settings["C"]
        n_cells = h * w

        land_use = rng.integers(0, c, size=(h, w), dtype=np.int64)
        water = rng.random((h, w)) < 0.08
        land_use[water] = 3

        pop_weights = np.array([1.0, 0.1, 0.05, 0.0, 0.7], dtype=np.float32)
        job_weights = np.array([0.1, 1.0, 0.8, 0.05, 0.6], dtype=np.float32)
        population = (pop_weights[land_use] * rng.uniform(0.6, 1.4, size=(h, w))).astype(np.float32)
        jobs = (job_weights[land_use] * rng.uniform(0.6, 1.4, size=(h, w))).astype(np.float32)
        population[water] = 0.0
        jobs[water] = 0.0

        rows, cols = np.divmod(np.arange(n_cells, dtype=np.float32), w)
        dist = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])
        orig = (population.ravel() + 0.2 * jobs.ravel()).astype(np.float32)
        dest = (jobs.ravel() + 0.2 * population.ravel()).astype(np.float32)
        demand = (orig[:, None] * dest[None, :]) / np.power(1.0 + dist, 1.5)
        np.fill_diagonal(demand, 0.0)
        total_d = float(demand.sum())
        if total_d > 0.0:
            demand = (demand / total_d) * float(population.sum())
        demand = demand.astype(np.float32)

        num_edges = min(self.config.max_edges, max(8, n_cells // 16))
        src = rng.integers(0, n_cells, size=(num_edges,), dtype=np.int64)
        dst = (src + rng.integers(1, max(2, w), size=(num_edges,), dtype=np.int64)) % n_cells
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        edge_time_min = rng.uniform(2.0, 12.0, size=(num_edges,)).astype(np.float32)
        edge_cost = rng.uniform(0.5, 1.5, size=(num_edges,)).astype(np.float32)
        asset_condition = rng.uniform(0.35, 0.90, size=(num_edges,)).astype(np.float32)

        state = StandInCityState(
            land_use=land_use,
            water=water,
            population=population,
            jobs=jobs,
            demand=demand,
            edge_index=edge_index,
            edge_time_min=edge_time_min,
            edge_cost=edge_cost,
            fare=float(self.config.fare_levels[1]),
            congestion_charge=float(self.config.congestion_levels[1]),
            asset_condition=asset_condition,
            budget=1.0,
        )
        state.validate()
        return state

    def evaluate_pricing(
        self, state: StandInCityState, fare: float, congestion_charge: float
    ) -> tuple[StandInCityState, dict[str, float]]:
        next_state = state.copy()
        next_state.fare = float(fare)
        next_state.congestion_charge = float(congestion_charge)

        h, w = self.settings["H"], self.settings["W"]
        mean_condition = float(np.mean(next_state.asset_condition))
        mean_transit_time = float(np.mean(next_state.edge_time_min)) / max(0.25, mean_condition)
        mean_car_time = 10.0

        u_transit = -0.12 * mean_transit_time - 0.55 * fare
        u_car = -0.10 * mean_car_time - 0.25 * congestion_charge
        transit_share = float(1.0 / (1.0 + np.exp(np.clip(u_car - u_transit, -20.0, 20.0))))

        total_demand = float(next_state.demand.sum())
        trip_retention = float(np.exp(-0.04 * (fare + 0.5 * congestion_charge)))
        realized_trips = total_demand * trip_retention

        transit_trips = realized_trips * transit_share
        car_trips = realized_trips * (1.0 - transit_share)
        step_revenue = (
            0.015 * fare * transit_trips + 0.010 * congestion_charge * car_trips
        ) / max(1.0, total_demand)
        next_state.budget = float(next_state.budget + step_revenue)

        emissions_proxy = float(np.clip(car_trips / max(1.0, total_demand), 0.0, 1.0))

        origin_demand = next_state.demand.sum(axis=1).reshape(h, w)
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            origin_demand[:mid_h, :mid_w].sum(),
            origin_demand[:mid_h, mid_w:].sum(),
            origin_demand[mid_h:, :mid_w].sum(),
            origin_demand[mid_h:, mid_w:].sum(),
        ]
        income_proxy = np.array([1.2, 1.0, 0.85, 0.75], dtype=np.float32)
        q_share = np.array(quadrants, dtype=np.float32) / max(1.0, float(origin_demand.sum()))
        avg_trip_fee = transit_share * fare + (1.0 - transit_share) * congestion_charge
        burdens = (q_share * avg_trip_fee) / (income_proxy * 5.0)
        burden_gap = float(np.clip(np.max(burdens) - np.min(burdens), 0.0, 1.0))

        return next_state, {
            "transit_share": transit_share,
            "emissions_proxy": emissions_proxy,
            "step_revenue": float(step_revenue),
            "burden_gap": burden_gap,
            "avg_travel_time": float(transit_share * mean_transit_time + (1.0 - transit_share) * mean_car_time),
        }

    def compute_usage_load(self, state: StandInCityState) -> np.ndarray:
        _, stats = self.evaluate_pricing(state, state.fare, state.congestion_charge)
        num_edges = state.edge_index.shape[1]
        origin_demand = state.demand.sum(axis=1)
        raw_load = origin_demand[state.edge_index[0]] * stats["transit_share"] + 0.05
        norm_load = raw_load / max(1e-6, float(np.max(raw_load)))
        return norm_load.astype(np.float32).reshape(num_edges)

    def repair_cost_vector(self, state: StandInCityState) -> np.ndarray:
        damage = np.maximum(0.0, 1.0 - state.asset_condition)
        costs = self.config.base_repair_cost * state.edge_cost * np.maximum(0.15, damage)
        return costs.astype(np.float32)

    def step_maintenance(
        self, state: StandInCityState, action: int
    ) -> tuple[StandInCityState, dict[str, float]]:
        next_state = state.copy()
        usage_load = self.compute_usage_load(next_state)
        repair_costs = self.repair_cost_vector(next_state)

        spent = 0.0
        if action > 0:
            edge_idx = action - 1
            cost = float(repair_costs[edge_idx])
            if cost > next_state.budget + 1e-8:
                raise ValueError(f"Action {action} costs {cost:.4f}, exceeding budget {next_state.budget:.4f}.")
            next_state.budget = float(next_state.budget - cost)
            next_state.asset_condition[edge_idx] = 1.0
            spent = cost

        wear = self.config.wear_rate * usage_load
        next_state.asset_condition = np.clip(next_state.asset_condition - wear, 0.0, 1.0).astype(np.float32)

        weighted_health = float(
            np.sum(usage_load * next_state.asset_condition) / max(1e-6, float(np.sum(usage_load)))
        )
        critical_frac = float(np.mean(next_state.asset_condition < 0.25))
        return next_state, {
            "weighted_health": weighted_health,
            "critical_frac": critical_frac,
            "step_cost": spent,
        }
