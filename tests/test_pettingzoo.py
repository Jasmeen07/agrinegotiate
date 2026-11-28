import numpy as np

from agrinegotiate import scenarios
from agrinegotiate.env.pettingzoo_env import AgriNegotiateEnv


def test_parallel_env_random_rollout():
    for name in ["SYM", "RURAL-DTN"]:
        env = AgriNegotiateEnv(scenarios.get(name))
        rng = np.random.default_rng(0)
        for ep in range(5):
            obs, _ = env.reset(seed=ep)
            total = dict.fromkeys(env.possible_agents, 0.0)
            while env.agents:
                acts = {}
                for a in env.agents:
                    assert env.observation_space(a)["observation"].shape == obs[a]["observation"].shape
                    m = obs[a]["action_mask"].astype(bool)
                    acts[a] = np.array([rng.choice(np.nonzero(row)[0]) for row in m])
                obs, rew, term, trunc, _ = env.step(acts)
                for a, r in rew.items():
                    total[a] += r
            assert all(np.isfinite(v) for v in total.values())
