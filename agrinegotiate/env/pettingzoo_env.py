"""PettingZoo ParallelEnv wrapper around the market core, so the benchmark can be used
with standard multi-agent tooling. One agent per farmer and per buyer; an agent's action
is one thread-level action for each counterpart."""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .core import Market, OBS_DIM, F, B


class AgriNegotiateEnv(ParallelEnv):
    metadata = {"name": "agrinegotiate_v0"}

    def __init__(self, scenario, mechanism=None, delay_aware: bool = True, reward_scale: float = 1.0):
        self.sc = scenario
        self.core = Market(scenario, mechanism=mechanism, delay_aware=delay_aware)
        self.reward_scale = reward_scale
        self.possible_agents = [f"farmer_{i}" for i in range(scenario.n_farmers)] + \
                               [f"buyer_{j}" for j in range(scenario.n_buyers)]
        self.agents = []

    def _n_threads(self, agent: str) -> int:
        return self.sc.n_buyers if agent.startswith("farmer") else self.sc.n_farmers

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        n = self._n_threads(agent)
        return spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, (n, OBS_DIM), np.float32),
            "action_mask": spaces.MultiBinary([n, self.core.n_actions]),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.MultiDiscrete([self.core.n_actions] * self._n_threads(agent))

    def _views(self):
        obs, mask, _ = self.core.observe()
        self._mask = mask
        out = {}
        for a in self.agents:
            k = int(a.split("_")[1])
            if a.startswith("farmer"):
                out[a] = {"observation": obs[F, k], "action_mask": mask[F, k].astype(np.int8)}
            else:
                out[a] = {"observation": obs[B, :, k], "action_mask": mask[B, :, k].astype(np.int8)}
        return out

    def reset(self, seed=None, options=None):
        self.core.reset(0 if seed is None else int(seed))
        self.agents = list(self.possible_agents)
        return self._views(), {a: {} for a in self.agents}

    def step(self, actions):
        N, M = self.sc.n_farmers, self.sc.n_buyers
        acts = np.full((2, N, M), self.core.A_WAIT, dtype=int)
        for a, act in actions.items():
            k = int(a.split("_")[1])
            if a.startswith("farmer"):
                acts[F, k, :] = act
            else:
                acts[B, :, k] = act
        done = self.core.step(acts, self._mask)
        rewards = {a: 0.0 for a in self.agents}
        if done:
            r_f, r_b, c_b = self.core.rewards()
            lam = getattr(self.core.mech, "lam", 0.0) if self.core.mech is not None else 0.0
            for a in self.agents:
                k = int(a.split("_")[1])
                rewards[a] = self.reward_scale * float(r_f[k] if a.startswith("farmer") else r_b[k] - lam * c_b[k])
        obs = self._views()
        terms = {a: done for a in self.agents}
        truncs = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, infos
