"""PPO with recurrence, which is what the paper used.

Written directly against our batched environment rather than through a library.  The
usual reason to reach for one is its collectors, and ours is already a batch of
thousands of environments stepping together -- so the library would be wrapping a
batch to hand it back as a batch.  Roughly two hundred lines buys transparency, no
version pinning, and a training loop that can be read end to end.

**Recurrent because the problem is.**  A drone that saw somebody two seconds ago and
cannot see them now has to remember; a policy without memory would have to be told
where they were, which is the oracle in a different hat.

**One network flying both drones**, as in the paper.  Each drone's observation is
encoded separately and only then concatenated, so splitting into two actors later is
a change to where the halves go rather than to the contract.

Note what is deliberately absent: no reward shaping beyond the named terms in
`reward/`, no observation normalisation tricks that would have to be reproduced on
real hardware, and no frame stacking, because the recurrence is doing that job.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hadrsim.core import spec
from hadrsim.policy import encoders


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01          # at the start
    entropy_final: float = 0.0     # and by the end
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    horizon: int = 32          # steps collected before each update
    hidden: int = 128


class ActorCritic(nn.Module):
    """One recurrent policy commanding both drones."""

    def __init__(self, encoder: str = "flat", hidden: int = 128, n_drones: int = 2,
                 spec_: spec.ObsSpec = spec.DEFAULT_OBS):
        super().__init__()
        self.n_drones = n_drones
        self.encoders = nn.ModuleList(
            [encoders.make(encoder, spec_=spec_, out=hidden) for _ in range(n_drones)])
        self.lstm = nn.LSTM(hidden * n_drones, hidden, batch_first=True)

        self.actor = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(),
                                   nn.Linear(128, n_drones * spec.ACTION_DIM))
        self.critic = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        # A learned, state-independent spread. Simpler than predicting it per state and
        # much less prone to collapsing early, which is the usual way a continuous
        # policy stops exploring and never starts again.
        self.log_std = nn.Parameter(torch.full((n_drones * spec.ACTION_DIM,), -0.5))

    def features(self, obs: dict) -> torch.Tensor:
        parts = [enc(obs[key]) for enc, key in zip(self.encoders, ("hlr", "llr"))]
        return torch.cat(parts, -1)

    def forward(self, obs: dict, state=None):
        x = self.features(obs)[:, None]                  # (B, 1, F)
        out, state = self.lstm(x, state)
        h = out[:, 0]
        mean = self.actor(h).view(-1, self.n_drones, spec.ACTION_DIM)
        value = self.critic(h)[:, 0]
        return mean, value, state

    def act(self, obs: dict, state=None, deterministic: bool = False):
        mean, value, state = self(obs, state)
        std = self.log_std.exp().view(1, self.n_drones, spec.ACTION_DIM)
        if deterministic:
            return mean.tanh(), None, value, state
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        # tanh keeps the command inside the contract's [-1, 1] without the gradient
        # pathologies of clipping, at the cost of the log-probability correction below.
        action = raw.tanh()
        logp = (dist.log_prob(raw) - torch.log1p(-action.pow(2) + 1e-6)).sum((-1, -2))
        return action, logp, value, state

    def evaluate(self, obs: dict, action: torch.Tensor, state=None):
        """Score an action already taken. Hands the carried state back.

        The state matters far more than it looks.  Evaluating each step from a
        *fresh* recurrent state scores a different policy from the one that acted --
        one with no memory of the episode so far -- so the importance ratio PPO is
        built on compares two unrelated things, and every update is noise.  The
        rollout has to be replayed in order, carrying state, which is why this
        returns it.
        """
        mean, value, state = self(obs, state)
        std = self.log_std.exp().view(1, self.n_drones, spec.ACTION_DIM)
        dist = torch.distributions.Normal(mean, std)
        raw = torch.atanh(action.clamp(-0.999, 0.999))
        logp = (dist.log_prob(raw) - torch.log1p(-action.pow(2) + 1e-6)).sum((-1, -2))
        return logp, value, dist.entropy().sum((-1, -2)), state


class PPO:
    """Collect, estimate advantage, clip, repeat."""

    def __init__(self, env, config: PPOConfig | None = None, encoder: str = "flat",
                 device=None):
        self.env = env
        self.cfg = config or PPOConfig()
        self.device = torch.device(device or env.device)
        self.net = ActorCritic(encoder, self.cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.obs = env.reset()
        self.state = None

    # --------------------------------------------------------------- collect

    @torch.no_grad()
    def collect(self):
        """Run the environment forward, keeping what the update needs."""
        buf = {k: [] for k in ("obs", "action", "logp", "value", "reward", "done")}
        info = {}
        # The recurrent state this rollout began from, so the update can replay it.
        start_state = None if self.state is None else tuple(h.detach() for h in self.state)
        for _ in range(self.cfg.horizon):
            action, logp, value, self.state = self.net.act(self.obs, self.state)
            nxt, reward, done, info = self.env.step(action)

            buf["obs"].append(_detach(self.obs))
            buf["action"].append(action)
            buf["logp"].append(logp)
            buf["value"].append(value)
            buf["reward"].append(reward)
            buf["done"].append(done.float())

            self.obs = nxt
            if bool(done.all()):
                self.obs = self.env.reset()
                self.state = None
        _, last_value, _ = self.net(self.obs, self.state)
        return buf, last_value, info, start_state

    def advantages(self, buf, last_value):
        """Generalised advantage estimate, walked backwards through the rollout."""
        cfg = self.cfg
        rewards = torch.stack(buf["reward"])
        values = torch.stack(buf["value"])
        dones = torch.stack(buf["done"])

        adv = torch.zeros_like(rewards)
        running = torch.zeros_like(last_value)
        nxt = last_value
        for t in reversed(range(len(rewards))):
            alive = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * nxt * alive - values[t]
            running = delta + cfg.gamma * cfg.gae_lambda * alive * running
            adv[t] = running
            nxt = values[t]
        return adv, adv + values

    # ---------------------------------------------------------------- update

    def entropy_now(self, progress: float) -> float:
        """The exploration bonus, annealed from `entropy` to `entropy_final`.

        Held constant, it never stops paying a policy to stay uncertain, and the
        spread it settles on is whatever balances that bonus against the advantage --
        which here meant never settling at all: `log_std` moved by under two percent
        in two million steps, in every run. A policy that cannot become confident is
        a noisy controller that sometimes stumbles into the right answer, which is
        exactly what the numbers showed: respectable with exploration on, near zero
        with it off.
        """
        cfg = self.cfg
        t = min(max(progress, 0.0), 1.0)
        return cfg.entropy + (cfg.entropy_final - cfg.entropy) * t

    def update(self, buf, adv, returns, start_state=None, progress: float = 0.0) -> dict:
        """Replay the rollout in order, carrying recurrent state, and clip.

        One backward pass over the whole sequence rather than one per timestep.  The
        recurrent state ties the steps together, so they are not independent samples
        and cannot be stepped on individually without severing the very connection
        the LSTM exists to make.
        """
        cfg = self.cfg
        actions = torch.stack(buf["action"])
        old_logp = torch.stack(buf["logp"])
        normalised = (adv - adv.mean()) / (adv.std() + 1e-8)
        alive = 1.0 - torch.stack(buf["done"])
        beta = self.entropy_now(progress)

        stats = {}
        for _ in range(cfg.epochs):
            state = None if start_state is None else tuple(h.clone() for h in start_state)
            losses = []
            for t in range(len(actions)):
                logp, value, entropy, state = self.net.evaluate(
                    buf["obs"][t], actions[t], state)
                ratio = (logp - old_logp[t]).exp()
                clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                policy_loss = -torch.min(ratio * normalised[t],
                                         clipped * normalised[t]).mean()
                value_loss = (value - returns[t]).pow(2).mean()
                losses.append(policy_loss + cfg.value_coef * value_loss
                              - beta * entropy.mean())
                # An episode boundary ends the memory. Carrying it across teaches the
                # policy to remember something that never happened.
                state = tuple(h * alive[t].view(1, -1, 1) for h in state)

            loss = torch.stack(losses).mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
            self.opt.step()
            stats = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                     "entropy": entropy.mean().item(), "beta": beta,
                     "std": float(self.net.log_std.exp().mean())}
        return stats

    def train(self, iterations: int, log_every: int = 5, on_log=None) -> list:
        history = []
        for i in range(1, iterations + 1):
            buf, last_value, info, start_state = self.collect()
            adv, returns = self.advantages(buf, last_value)
            stats = self.update(buf, adv, returns, start_state, i / max(iterations, 1))

            row = {
                "iteration": i,
                "steps": i * self.cfg.horizon * self.env.batch,
                "reward": float(torch.stack(buf["reward"]).mean()),
                "captured": float(info.get("captured", torch.zeros(1)).float().mean()),
                "reached": float(info.get("reached", torch.zeros(1)).float().mean()),
                **stats,
            }
            history.append(row)
            if i % log_every == 0 or i == 1:
                print(f"  {i:>4}  steps {row['steps']:>9,}  reward {row['reward']:+8.3f}"
                      f"  captured {row['captured']*100:5.1f}%"
                      f"  reached {row['reached']*100:5.1f}%"
                      f"  std {row.get('std', 0):5.3f}"
                      f"  beta {row.get('beta', 0):6.4f}")
                if on_log:
                    on_log(row)
        return history

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "config": self.cfg.__dict__}, path)


def _detach(obs: dict) -> dict:
    return {k: {kk: vv.detach() for kk, vv in v.items()}
            for k, v in obs.items() if isinstance(v, dict)}
