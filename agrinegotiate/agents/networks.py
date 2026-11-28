"""Policy networks: 64 tanh units, masked categorical policy.

ActorCritic           feed-forward, two hidden layers (delay-unaware agents, and the default).
RecurrentActorCritic  one hidden layer followed by a 64-unit GRU (delay-aware agents).

Both have a second value head that predicts the expected fee base.
`walk_bias` lowers the initial probability of WALK so that early random play does not
teach both sides to leave before they have ever seen a deal (DECISIONS.md #18).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _layer(i, o, gain):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


def _policy_head(hidden, n_actions, walk_bias):
    head = _layer(hidden, n_actions, 0.01)
    if walk_bias:
        with torch.no_grad():
            head.bias[n_actions - 2] = walk_bias          # action index K+1 is WALK
    return head


class ActorCritic(nn.Module):
    recurrent = False

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, walk_bias: float = 0.0):
        super().__init__()
        g = 2 ** 0.5
        self.pi = nn.Sequential(_layer(obs_dim, hidden, g), nn.Tanh(), _layer(hidden, hidden, g), nn.Tanh(),
                                _policy_head(hidden, n_actions, walk_bias))
        self.vf = nn.Sequential(_layer(obs_dim, hidden, g), nn.Tanh(), _layer(hidden, hidden, g), nn.Tanh())
        self.v_reward = _layer(hidden, 1, 1.0)
        self.v_cost = _layer(hidden, 1, 1.0)

    def dist(self, obs, mask):
        logits = self.pi(obs).masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def values(self, obs):
        h = self.vf(obs)
        return self.v_reward(h).squeeze(-1), self.v_cost(h).squeeze(-1)

    def step(self, obs, mask, h=None):
        """Same interface as the recurrent net: (dist, reward value, cost value, new hidden state)."""
        return self.dist(obs, mask), *self.values(obs), h


class RecurrentActorCritic(nn.Module):
    recurrent = True

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, walk_bias: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.Sequential(_layer(obs_dim, hidden, 2 ** 0.5), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = _policy_head(hidden, n_actions, walk_bias)
        self.v_reward = _layer(hidden, 1, 1.0)
        self.v_cost = _layer(hidden, 1, 1.0)

    def initial(self, n: int):
        return torch.zeros(n, self.hidden)

    def step(self, obs, mask, h):
        """One decision step. obs [n, D], mask [n, A], h [n, H] -> dist, value, cost value, new h."""
        out, h2 = self.gru(self.enc(obs).unsqueeze(1), h.unsqueeze(0))
        z = out.squeeze(1)
        d = torch.distributions.Categorical(logits=self.pi(z).masked_fill(~mask, -1e9))
        return d, self.v_reward(z).squeeze(-1), self.v_cost(z).squeeze(-1), h2.squeeze(0)

    def seq(self, obs):
        """Whole trajectories from a zero state. obs [n, T, D] -> logits [n, T, A], v [n, T], vc [n, T]."""
        z, _ = self.gru(self.enc(obs))
        return self.pi(z), self.v_reward(z).squeeze(-1), self.v_cost(z).squeeze(-1)


def make_net(obs_dim, n_actions, hidden=64, recurrent=False, walk_bias=0.0):
    cls = RecurrentActorCritic if recurrent else ActorCritic
    return cls(obs_dim, n_actions, hidden, walk_bias)
