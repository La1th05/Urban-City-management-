"""Day 1 scaffold: sequential zoning mechanics only, NOT a trained RL environment.

The reward remains 0.0 until Member 2's simulator metrics are integrated.
All observations are synthetic and embeddings are random stubs for wiring tests.
Run from the project root: python -m agents.check_zoning
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common import get_settings
from encoders.city_tiles import make_land_use_map
from encoders.stub_embeddings import get_embeddings


class ZoningEnv(gym.Env):
    """Assign one of five zone classes to each non-water cell, row-major."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self.h, self.w, self.d = settings.H, settings.W, settings.D
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "embeddings": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.h, self.w, self.d), dtype=np.float32,
            ),
            "land_use": spaces.Box(
                low=-1, high=4, shape=(self.h, self.w), dtype=np.int64,
            ),
            "water": spaces.MultiBinary((self.h, self.w)),
            # H*W denotes the cursor after the last assignable cell.
            "cursor": spaces.Discrete(self.h * self.w + 1),
        })
        self._land_use = np.full((self.h, self.w), -1, dtype=np.int64)
        self._water = np.zeros((self.h, self.w), dtype=np.bool_)
        self._embeddings = np.zeros((self.h, self.w, self.d), dtype=np.float32)
        self._active_cells = np.empty((0,), dtype=np.int64)
        self._position = 0
        self._done = True

    def _cursor(self) -> int:
        if self._position >= len(self._active_cells):
            return self.h * self.w
        return int(self._active_cells[self._position])

    def _observe(self) -> dict:
        return {
            "embeddings": self._embeddings.copy(),
            "land_use": self._land_use.copy(),
            "water": self._water.copy(),
            "cursor": self._cursor(),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # With a fixed seed, we get exactly the same water layout.
        city_seed = int(seed) if seed is not None else int(
            self.np_random.integers(0, 2**31 - 1)
        )
        _, water = make_land_use_map(self.h, self.w, seed=city_seed)
        self._water = np.asarray(water, dtype=np.bool_)
        # Every buildable cell starts UNASSIGNED; water is a separate mask.
        self._land_use = np.full((self.h, self.w), -1, dtype=np.int64)
        self._active_cells = np.flatnonzero(~self._water.ravel())
        self._position = 0
        self._done = len(self._active_cells) == 0
        self._embeddings = get_embeddings()  # random stub, for interface tests only
        if self._embeddings.shape != (self.h, self.w, self.d):
            raise ValueError("Embeddings don't match common.get_settings()")
        return self._observe(), {"data_label": "synthetic", "seed": city_seed}

    def action_masks(self) -> np.ndarray:
        # There are no class-specific zoning constraints until agreed with Member 2.
        return np.ones(5, dtype=np.bool_) if not self._done else np.zeros(5, dtype=np.bool_)

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Episode finished. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid zoning action: {action!r}")
        if not self.action_masks()[int(action)]:
            raise ValueError(f"Action is masked: {action!r}")

        flat_index = self._cursor()
        row, col = divmod(flat_index, self.w)
        self._land_use[row, col] = int(action)
        self._position += 1
        self._done = self._position >= len(self._active_cells)

        # TODO: Replace with simulator metrics and documented reward terms.
        reward = 0.0
        info = {
            "reward_terms": {"simulator_pending": 0.0},
            "assigned_cell": (row, col),
            "data_label": "synthetic",
        }
        terminated, truncated = self._done, False
        return self._observe(), float(reward), terminated, truncated, info
