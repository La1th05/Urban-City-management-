
import numpy as np

from agents.transit_env import TransitEnv


def random_policy(obs, mask, rng, coverage, demand):
    valid_actions = np.flatnonzero(mask)

    return int(rng.choice(valid_actions))


def greedy_policy(obs, mask, rng, coverage, demand):

    built = obs["built_routes"]

    # Find demand already covered by built routes
    covered = np.any(coverage[built], axis=0)

    # Calculate additional demand coverage
    uncovered_demand = demand * (~covered)

    gains = coverage @ uncovered_demand

    costs = obs["route_costs"]

    # Calculate benefit-to-cost ratio
    scores = gains / np.maximum(costs, 1e-8)

    # Exclude invalid routes
    scores[~mask[1:]] = -np.inf

    best_route = int(np.argmax(scores))

    if scores[best_route] <= 0:
        return 0

    return best_route + 1


def run_transit(
    policy,
    seed=42,
    *,
    costs=None,
    demand=None,
    coverage=None,
    budget=0.7
):

    # Default synthetic test data
    if costs is None and demand is None and coverage is None:

        costs = [0.2, 0.4, 0.6]

        demand = [50, 40, 30, 20]

        coverage = [
            [True, False, False, True],
            [False, True, True, False],
            [True, True, True, False]
        ]

    elif any(x is None for x in (costs, demand, coverage)):
        raise ValueError("Provide all three input arrays")

    # Convert inputs to NumPy arrays
    costs = np.asarray(costs, dtype=np.float32)
    demand = np.asarray(demand, dtype=np.float32)
    coverage = np.asarray(coverage, dtype=bool)

    # Input validation
    if costs.ndim != 1 or costs.size == 0:
        raise ValueError("Invalid route costs shape")

    if demand.ndim != 1:
        raise ValueError("Demand must be a 1D array")

    if coverage.shape != (costs.size, demand.size):
        raise ValueError("Invalid coverage shape")

    if not np.all(np.isfinite(costs)) or np.any(costs <= 0):
        raise ValueError("Invalid route costs")

    if not np.all(np.isfinite(demand)) or np.any(demand < 0):
        raise ValueError("Invalid demand")

    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Invalid budget")

    # Initialize environment
    env = TransitEnv(costs, budget=budget)

    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0

    terminated = False
    truncated = False

    # Run a complete episode
    while not (terminated or truncated):

        mask = env.action_masks()

        action = policy(
            obs, mask, rng, coverage, demand
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    # Calculate final demand coverage
    built = obs["built_routes"]

    covered = np.any(coverage[built], axis=0)

    covered_demand = float(demand[covered].sum())

    # Print results
    print("Steps:", steps)
    print("Built routes:", built)
    print("Remaining budget:", round(float(obs["budget"][0]), 4))
    print("Covered demand:", covered_demand)
    print("Total reward:", total_reward)

    env.close()

    # Return results for visualization and future integration
    return {
        "built_routes": built.copy(),
        "covered_demand": covered_demand,
        "total_demand": float(demand.sum()),
        "budget": float(obs["budget"][0]),
        "total_reward": total_reward,
        "steps": steps
    }


def main():

    print("\nRandom Policy")
    run_transit(random_policy)

    print("\nGreedy Policy")
    run_transit(greedy_policy)

    print("\nPASS: Both transit policies completed")
    print("NOTE: Synthetic demand, placeholder reward")


if __name__ == "__main__":
    main()
