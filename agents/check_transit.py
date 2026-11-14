
import numpy as np
from agents.transit_env import TransitEnv


def main():
    env = TransitEnv(
        route_costs=[0.2, 0.4, 0.6],
        budget=0.7
    )

    # 1. Test reset and reproducibility
    obs, info = env.reset(seed=42)
    second_obs, _ = env.reset(seed=42)

    for key in obs:
        assert np.array_equal(obs[key], second_obs[key])

    assert env.observation_space.contains(obs)
    assert obs["route_costs"].dtype == np.float32
    assert obs["built_routes"].dtype == np.bool_
    assert obs["budget"].dtype == np.float32

    # 2. Test route construction
    obs, reward, terminated, truncated, info = env.step(1)

    assert obs["built_routes"][0]
    assert np.isclose(obs["budget"][0], 0.5)

    # 3. Test action masking
    expected_mask = [True, False, True, False]

    assert np.array_equal(
        env.action_masks(), expected_mask
    )

    # 4. Test invalid actions
    for action in [1, 3, 4]:
        try:
            env.step(action)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid action {action} was accepted"
            )

    # 5. Run a complete random episode
    obs, info = env.reset(seed=42)
    rng = np.random.default_rng(42)

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        mask = env.action_masks()

        assert mask.dtype == np.bool_
        assert mask.shape == (env.k + 1,)
        assert mask.any()

        valid_actions = np.flatnonzero(mask)
        action = int(rng.choice(valid_actions))

        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert obs["budget"][0] >= 0
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert isinstance(info["reward_terms"], dict)

        steps += 1
        assert steps <= env.k

    assert not env.action_masks().any()

    print("PASS: reset and reproducibility")
    print("PASS: observation shapes and dtypes")
    print("PASS: budget constraints")
    print("PASS: action masking")
    print("PASS: invalid action rejection")
    print("PASS: full random episode")

    print("Steps:", steps)
    print("Built routes:", obs["built_routes"])
    print("Remaining budget:", obs["budget"][0])

    print("NOTE: synthetic routes and placeholder reward")


if __name__ == "__main__":
    main()
