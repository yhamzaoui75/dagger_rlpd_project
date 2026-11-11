import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_mind.collectors import EpisodeCollector
from rl_mind.core import Action, Actor
from rl_mind.data import minibatches
from rl_mind.env import VecEnv
from rl_mind.nn import build_mlp

import common


class ExpertActor(Actor[Action]):
    """Drives with the expert's action, which SoloKartEnv puts in every observation."""

    def forward(self, obs) -> Action:
        return Action(value=obs.expert_action)


class DAggerActor(Actor[Action]):
    """Deterministic policy: an MLP from the features to an action in [-1, 1]^2."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int):
        super().__init__()
        self.model = build_mlp([obs_dim, *hidden, action_dim], output_activation=nn.Tanh())

    def forward(self, obs) -> Action:
        return Action(value=self.model(obs.features))


@dataclass(frozen=True)
class DAggerConfig:
    """Hyperparameters of a DAgger run (network size, batch size and learning rate of the SAC practical)."""

    env_name: str = common.ENV_ID
    seed: int = 0

    n_iterations: int = 15
    n_epochs: int = 30
    batch_size: int = 256

    hidden: tuple[int, ...] = (256, 256)
    lr: float = 3e-4


def train_from(dataset, cfg: DAggerConfig) -> tuple[DAggerActor, float]:
    """Train a new policy on the whole dataset by mean-squared error to the expert's actions."""
    actor = DAggerActor(common.FEATURE_DIM, cfg.hidden, common.ACTION_DIM)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    for _ in range(cfg.n_epochs):
        for batch in minibatches(dataset, cfg.batch_size):
            loss = F.mse_loss(actor(batch).value, batch.expert_action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return actor, loss.item()


def run_dagger(cfg: DAggerConfig) -> dict:
    """Run DAgger for one seed; the lap driven at iteration i > 1 measures the policy trained at i - 1.

    The environment and the policies are deterministic, so that lap gives the
    exact lap time of the policy. The final policy is evaluated with the
    shared protocol of ``common.evaluate_policy``.
    """
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, 1, seed=cfg.seed)
    out_dir = common.PROJECT_DIR / "results" / "dagger" / f"seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    actor, dataset, log = ExpertActor(), None, []
    for i in range(1, cfg.n_iterations + 1):
        episode = EpisodeCollector(env, actor).collect(1)[0]
        dataset = episode.obs if dataset is None else type(dataset).cat([dataset, episode.obs])
        actor, loss = train_from(dataset, cfg)
        log.append({
            "iteration": i,
            "driver": "expert" if i == 1 else f"policy {i - 1}",
            "lap_steps": len(episode.reward),
            "lap_finished": bool(episode.terminated),
            "dataset_size": len(dataset),
            "train_loss": loss,
        })
        print(log[-1], flush=True)

    with torch.no_grad():
        final = common.summarize_laps(common.evaluate_policy(
            lambda obs: actor.model(torch.as_tensor(obs["features"])).numpy()))
    final.pop("ci", None)
    final = {key: float(value) for key, value in final.items()}
    print(f"Final policy: {final}", flush=True)

    torch.save(actor.state_dict(), out_dir / "policy.pt")
    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"config": asdict(cfg), "final": final}, f, indent=2)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAgger on SuperTuxKart (one run per seed).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    torch.set_num_threads(1)  # A small MLP trains faster on one thread than with thread contention.
    for seed in args.seeds:
        run_dagger(replace(DAggerConfig(), seed=seed))