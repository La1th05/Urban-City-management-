
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def route_action_mask(route_costs, built_routes, budget):
    affordable = route_costs.astype(np.float64) <= budget + 1e-7
    available = ~built_routes

    valid_routes = affordable & available

    return np.concatenate(([True], valid_routes))


class TransitEnv(gym.Env):

    def __init__(self, route_costs, budget=1.0):
        super().__init__()

        self.route_costs = np.asarray(
            route_costs, dtype=np.float32
        )

        self.k = len(self.route_costs)
        self.initial_budget = float(budget)

        self.action_space = spaces.Discrete(self.k + 1)

        self.observation_space = spaces.Dict({
            "route_costs": spaces.Box(
                0, np.inf,
                shape=(self.k,),
                dtype=np.float32
            ),
            "built_routes": spaces.MultiBinary(self.k),
            "budget": spaces.Box(
                0, np.inf,
                shape=(1,),
                dtype=np.float32
            )
        })

        self.budget = self.initial_budget
        self.built_routes = np.zeros(self.k, dtype=bool)
        self._steps = 0
        self._done = True

    def _observe(self):
        return {
            "route_costs": self.route_costs.copy(),
            "built_routes": self.built_routes.copy(),
            "budget": np.array(
                [self.budget], dtype=np.float32
            )
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.budget = self.initial_budget
        self.built_routes = np.zeros(self.k, dtype=bool)
        self._steps = 0
        self._done = False

        return self._observe(), {
            "data_label": "synthetic"
        }

    def action_masks(self):
        if self._done:
            return np.zeros(self.k + 1, dtype=bool)

        return route_action_mask(
            self.route_costs,
            self.built_routes,
            self.budget
        )

    def step(self, action):
        if self._done:
            raise RuntimeError(
                "Episode finished. Call reset()."
            )

        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        if not self.action_masks()[action]:
            raise ValueError("Action is masked")

        if action != 0:
            route_index = action - 1

            self.built_routes[route_index] = True

            self.budget = max(
                0.0,
                self.budget - float(
                    self.route_costs[route_index]
                )
            )

        self._steps += 1

        terminated = not bool(
            self.action_masks()[1:].any()
        )

        truncated = (
            self._steps >= self.k and not terminated
        )

        self._done = terminated or truncated

        # TODO: Integrate Member 2's simulator metrics.
        reward = 0.0

        info = {
            "reward_terms": {
                "simulator_pending": 0.0
            },
            "selected_action": int(action),
            "data_label": "synthetic"
        }

        return (
            self._observe(),
            reward,
            terminated,
            truncated,
            info
        )
