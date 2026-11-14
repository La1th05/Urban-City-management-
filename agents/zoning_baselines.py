
import numpy as np

from agents.zoning_env import ZoningEnv


def random_policy(obs, mask, rng):
    valid_actions = np.flatnonzero(mask)
    return int(rng.choice(valid_actions))


def rule_based_policy(obs, mask, rng):
    h, w = obs["land_use"].shape

    row, col = divmod(obs["cursor"], w)

    center_row = (h - 1) / 2
    center_col = (w - 1) / 2

    distance = max(
        abs(row - center_row) / max(center_row, 1),
        abs(col - center_col) / max(center_col, 1)
    )

    if distance <= 0.3:
        action = 1  # Commercial

    elif distance <= 0.8:
        action = 0  # Residential

    else:
        action = 3  # Green

    if not mask[action]:
        return random_policy(obs, mask, rng)

    return action


def run_zoning(policy, seed=42):
    env = ZoningEnv()
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0

    while obs["cursor"] < obs["land_use"].size:
        mask = env.action_masks()

        action = policy(obs, mask, rng)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    land_use = obs["land_use"]
    water = obs["water"]

    assert np.all(land_use[water] == -1)
    assert np.all(land_use[~water] >= 0)
    assert steps == np.count_nonzero(~water)

    env.close()

    return land_use, water, total_reward, steps


def main():
    policies = {
        "Random": random_policy,
        "Rule-based": rule_based_policy
    }

    for name, policy in policies.items():
        land, water, reward, steps = run_zoning(
            policy, seed=42
        )

        print("\nPolicy:", name)
        print("Steps:", steps)
        print("Residential:", np.sum((land == 0) & ~water))
        print("Commercial:", np.sum((land == 1) & ~water))
        print("Industrial:", np.sum((land == 2) & ~water))
        print("Green:", np.sum((land == 3) & ~water))
        print("Mixed:", np.sum((land == 4) & ~water))
        print("Total reward:", reward)

    print("\nPASS: Both policies completed")
    print("NOTE: Synthetic city, placeholder reward")


if __name__ == "__main__":
    main()
