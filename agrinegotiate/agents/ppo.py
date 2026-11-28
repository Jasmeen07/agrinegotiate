"""Independent PPO with one shared network per role (IPPO), action masking,
per-thread decisions and the optional adaptive fee.

Each negotiation thread is one trajectory. The agent's terminal reward is credited
to the last step of each of its threads; gamma = 1.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..env.core import Market, OBS_DIM, F, B
from ..eval.evaluate import EVAL_SEED_START, evaluate
from ..eval.stats import KEYS
from . import scripted as S
from .networks import make_net

TRAIN_EPISODE_SEED_BASE = 10_000_000      # training episodes never reuse tuning or evaluation seeds


@dataclass
class PPOConfig:
    iters: int = 200
    episodes_per_iter: int = 256
    lr: float = 3e-4
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_start: float = 0.01
    ent_end: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    hidden: int = 64
    reward_scale: float = 5.0
    mix_scripted: float = 0.3             # share of training episodes with one scripted side
    mix_pool: tuple = ("Boulware", "Linear", "Conceder", "TitForTat")
    eval_every: int = 10
    eval_episodes: int = 300
    final_eval_episodes: int = 1000
    delay_aware: bool = True
    recurrent: bool = False
    walk_bias: float = -3.0               # initial logit offset on WALK (DECISIONS.md #18)
    target_agent_steps: int = 0           # > 0: train until this many agent-steps instead of `iters`


def _step(net, store, keys, obs, mask):
    """One decision for a batch of threads. Recurrent nets keep a hidden state per thread key in `store`."""
    h = torch.stack([store.get(k, torch.zeros(net.hidden)) for k in keys]) if net.recurrent else None
    dist, v, cv, h2 = net.step(obs, mask, h)
    if net.recurrent:
        store.update(zip(keys, h2))
    return dist, v, cv


class PolicyAgent:
    """Wraps a trained network so it can be used anywhere a scripted agent can."""

    def __init__(self, net, name: str = "IPPO"):
        self.net, self.name = net, name

    def reset(self, seed: int) -> None:
        self.gen = torch.Generator().manual_seed(int(seed) + 12345)
        self.h: dict = {}

    @torch.no_grad()
    def act(self, env, obs, mask, active) -> np.ndarray:
        acts = np.full(active.shape, env.A_WAIT, dtype=int)
        idx = np.nonzero(active)
        if idx[0].size:
            keys = list(zip(idx[0].tolist(), idx[1].tolist()))
            d, _, _ = _step(self.net, self.h, keys, torch.as_tensor(obs[idx]), torch.as_tensor(mask[idx]))
            acts[idx] = torch.multinomial(d.probs, 1, generator=self.gen).squeeze(-1).numpy()
        return acts


class Buffer:
    def __init__(self):
        self.obs, self.mask, self.act, self.logp, self.val, self.cval, self.traj = [], [], [], [], [], [], []
        self.traj_ids: dict = {}
        self.final_r: dict = {}
        self.final_c: dict = {}

    def tid(self, key):
        return self.traj_ids.setdefault(key, len(self.traj_ids))

    def __len__(self):
        return sum(len(o) for o in self.obs)


def collect(sc, nets, cfg: PPOConfig, mech, seeds, rng):
    """Play len(seeds) episodes in lock-step. Returns per-role buffers and the finished results."""
    envs = [Market(sc, mechanism=mech, delay_aware=cfg.delay_aware) for _ in seeds]
    scripted = []
    for e, (env, seed) in enumerate(zip(envs, seeds)):
        env.reset(int(seed))
        ag = {}
        if rng.random() < cfg.mix_scripted:
            side = int(rng.integers(2))
            ag[side] = S.make(str(rng.choice(cfg.mix_pool)), side, sc)
            ag[side].reset(int(seed))
        scripted.append(ag)
    bufs = {F: Buffer(), B: Buffer()}
    hid = {F: {}, B: {}}
    live = list(range(len(envs)))
    while live:
        views = {e: envs[e].observe() for e in live}
        acts = {e: np.full((2, sc.n_farmers, sc.n_buyers), envs[e].A_WAIT, dtype=int) for e in live}
        for s in (F, B):
            rows, where = [], []
            for e in live:
                obs, mask, active = views[e]
                if s in scripted[e]:
                    acts[e][s] = scripted[e][s].act(envs[e], obs[s], mask[s], active[s])
                    continue
                ii, jj = np.nonzero(active[s])
                for i, j in zip(ii, jj):
                    rows.append((obs[s, i, j], mask[s, i, j]))
                    where.append((e, i, j))
            if not rows:
                continue
            o = torch.as_tensor(np.stack([r[0] for r in rows]))
            m = torch.as_tensor(np.stack([r[1] for r in rows]))
            with torch.no_grad():
                d, v, cv = _step(nets[s], hid[s], where, o, m)
                a = d.sample()
                lp = d.log_prob(a)
            buf = bufs[s]
            buf.obs.append(o); buf.mask.append(m); buf.act.append(a); buf.logp.append(lp)
            buf.val.append(v); buf.cval.append(cv)
            buf.traj.append(torch.tensor([buf.tid(w) for w in where]))
            for (e, i, j), ai in zip(where, a.tolist()):
                acts[e][s, i, j] = ai
        for e in live:
            envs[e].step(acts[e], views[e][1])
        live = [e for e in live if not envs[e].done]
    # credit each thread with its agent's terminal reward (farmer i, or buyer j and its fee base)
    rewards = [env.rewards() for env in envs]
    for (e, i, j), tid in bufs[F].traj_ids.items():
        bufs[F].final_r[tid], bufs[F].final_c[tid] = rewards[e][0][i], 0.0
    for (e, i, j), tid in bufs[B].traj_ids.items():
        bufs[B].final_r[tid], bufs[B].final_c[tid] = rewards[e][1][j], rewards[e][2][j]
    return bufs, [env.result() for env in envs]


def _gae(vals, final, lam):
    """Advantages and returns for one trajectory with a single terminal reward, gamma = 1."""
    n = len(vals)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for k in reversed(range(n)):
        nxt = vals[k + 1] if k + 1 < n else 0.0
        delta = (final if k == n - 1 else 0.0) + nxt - vals[k]
        last = delta + lam * last
        adv[k] = last
    return adv, adv + vals


def _advantages(buf: Buffer, cfg: PPOConfig, lam_fair: float):
    """Per-trajectory GAE for the reward critic and the cost critic.
    Returns the trajectories (as index arrays), the Lagrangian advantage A_reward - lam * A_cost, and both returns."""
    val, cval, traj = torch.cat(buf.val).numpy(), torch.cat(buf.cval).numpy(), torch.cat(buf.traj).numpy()
    adv_r, ret_r, adv_c, ret_c = (np.zeros_like(val) for _ in range(4))
    order = np.argsort(traj, kind="stable")
    segs = np.split(order, np.flatnonzero(np.diff(traj[order])) + 1)
    for seg in segs:
        tid = int(traj[seg[0]])
        adv_r[seg], ret_r[seg] = _gae(val[seg], cfg.reward_scale * buf.final_r[tid], cfg.gae_lambda)
        adv_c[seg], ret_c[seg] = _gae(cval[seg], cfg.reward_scale * buf.final_c[tid], cfg.gae_lambda)
    return segs, torch.as_tensor(adv_r - lam_fair * adv_c), torch.as_tensor(ret_r), torch.as_tensor(ret_c)


def update(net, opt, buf: Buffer, cfg: PPOConfig, ent_coef: float, lam_fair: float, rng) -> dict:
    if not buf.obs:
        return {}
    obs, mask = torch.cat(buf.obs), torch.cat(buf.mask)
    act, logp_old = torch.cat(buf.act), torch.cat(buf.logp)
    segs, adv, ret_r, ret_c = _advantages(buf, cfg, lam_fair)
    n = obs.shape[0]
    stats = {}
    if net.recurrent:                      # padded whole trajectories, one row per thread
        tmax = max(len(g) for g in segs)
        idx_pad = torch.full((len(segs), tmax), -1, dtype=torch.long)
        for r, g in enumerate(segs):
            idx_pad[r, :len(g)] = torch.as_tensor(g)
        obs_pad = torch.zeros(len(segs), tmax, obs.shape[1])
        valid_all = idx_pad >= 0
        obs_pad[valid_all] = obs[idx_pad[valid_all]]
    for _ in range(cfg.epochs):
        units = len(segs) if net.recurrent else n
        perm = torch.as_tensor(rng.permutation(units))
        for mb in torch.tensor_split(perm, cfg.minibatches):
            if net.recurrent:
                logits, v, cv = net.seq(obs_pad[mb])
                valid = idx_pad[mb] >= 0
                flat = idx_pad[mb][valid]
                logits, v, cv = logits[valid], v[valid], cv[valid]
                d = torch.distributions.Categorical(logits=logits.masked_fill(~mask[flat], -1e9))
            else:
                flat = mb
                d = net.dist(obs[flat], mask[flat])
                v, cv = net.values(obs[flat])
            lp = d.log_prob(act[flat])
            a = adv[flat]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = torch.exp(lp - logp_old[flat])
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            vloss = 0.5 * ((v - ret_r[flat]) ** 2).mean() + 0.5 * ((cv - ret_c[flat]) ** 2).mean()
            ent = d.entropy().mean()
            loss = pg + cfg.vf_coef * vloss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            stats = dict(pg=pg.item(), vloss=vloss.item(), entropy=ent.item())
    return stats


def train(sc, cfg: PPOConfig, seed: int, out_dir, mech=None, adaptive_fee: bool = False, log=print) -> dict:
    """Train farmer and buyer networks by self-play. Writes config, curve, checkpoints, metrics."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    n_actions = sc.n_bins + 3
    nets = {s: make_net(OBS_DIM, n_actions, cfg.hidden, cfg.recurrent, cfg.walk_bias) for s in (F, B)}
    opts = {s: torch.optim.Adam(nets[s].parameters(), lr=cfg.lr, eps=1e-5) for s in (F, B)}
    (out / "config.json").write_text(json.dumps(dict(
        scenario=sc.to_dict(), ppo=asdict(cfg), seed=seed, adaptive_fee=adaptive_fee,
        mechanism=None if mech is None else {k: v for k, v in vars(mech).items()}), indent=1, default=str))
    agents = {s: PolicyAgent(nets[s]) for s in (F, B)}
    curve, j_c_ema, t0, env_steps = [], None, time.time(), 0
    budget, it = cfg.target_agent_steps, 0
    while (env_steps < budget) if budget else (it < cfg.iters):
        frac = min(1.0, env_steps / budget) if budget else it / max(1, cfg.iters - 1)
        for s in (F, B):
            opts[s].param_groups[0]["lr"] = cfg.lr * (1 - frac) + 1e-6
        ent = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
        seeds = TRAIN_EPISODE_SEED_BASE + seed * 1_000_000 + it * cfg.episodes_per_iter + np.arange(cfg.episodes_per_iter)
        bufs, results = collect(sc, nets, cfg, mech, seeds, rng)
        env_steps += len(bufs[F]) + len(bufs[B])
        lam = getattr(mech, "lam", 0.0)            # 0 without a fee mechanism
        stats = {s: update(nets[s], opts[s], bufs[s], cfg, ent, lam if s == B else 0.0, rng) for s in (F, B)}
        if adaptive_fee:
            deals = [d for r in results for d in r.deals if d["g_t"] > 1e-9]
            if deals:
                j_c = float(np.mean([d["shortfall"] / d["g_t"] for d in deals]))
                j_c_ema = j_c if j_c_ema is None else 0.8 * j_c_ema + 0.2 * j_c
                mech.dual_update(j_c_ema)
        last = (env_steps >= budget) if budget else (it == cfg.iters - 1)
        if it % cfg.eval_every == 0 or last:
            m = evaluate(sc, agents[F], agents[B], cfg.eval_episodes, EVAL_SEED_START, mech, cfg.delay_aware)
            row = dict(iter=it, agent_steps=env_steps, seconds=round(time.time() - t0, 1),
                       lam_fair=lam, j_c_ema=j_c_ema,
                       entropy_f=stats[F].get("entropy"), entropy_b=stats[B].get("entropy"),
                       **{k: m.get(k) for k in ("E", "FGC", "BGC", "S_F", "agreement_rate", "phi_mean",
                                                "time_to_agreement", "price_share", "j_c", "fees_over_gstar",
                                                "fallback_rate", "nash_distance")})
            curve.append(row)
            log(f"[{sc.name} seed {seed}] it {it:4d} steps {env_steps:8d} E {row['E']:.3f} FGC {row['FGC']:.3f} "
                f"S_F {row['S_F'] if row['S_F'] is None else round(row['S_F'], 3)} agree {row['agreement_rate']:.3f} "
                f"t_deal {row['time_to_agreement']} lam {lam:.3f}")
            (out / "curve.json").write_text(json.dumps(curve, indent=1))
        it += 1
    torch.save({"farmer": nets[F].state_dict(), "buyer": nets[B].state_dict(), "hidden": cfg.hidden,
                "recurrent": cfg.recurrent, "delay_aware": cfg.delay_aware,
                "lam_fair": getattr(mech, "lam", 0.0)}, out / "checkpoint.pt")
    final = evaluate(sc, agents[F], agents[B], cfg.final_eval_episodes, EVAL_SEED_START, mech, cfg.delay_aware)
    last10 = curve[-10:]
    summary = dict(
        seed=seed, scenario=sc.name, agent_steps=env_steps, seconds=round(time.time() - t0, 1),
        final_checkpoint_1000_episodes=final,
        last10_checkpoint_mean={k: _mean_or_none([r[k] for r in last10]) for k in KEYS},
        last10_checkpoints_used=len(last10))
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    return summary


def _mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def load_agents(path, sc):
    ck = torch.load(path, weights_only=True)
    out = {}
    for s, key in ((F, "farmer"), (B, "buyer")):
        net = make_net(OBS_DIM, sc.n_bins + 3, ck["hidden"], ck.get("recurrent", False))
        net.load_state_dict(ck[key])
        out[s] = PolicyAgent(net)
    return out
