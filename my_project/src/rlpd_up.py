import copy, csv, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import common
from dataclasses import dataclass
from torch import Tensor
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution
from tqdm.auto import tqdm
from rl_mind.core import Actor, StochasticAction
from rl_mind.nn import build_mlp, soft_update
from rl_mind.env import VecEnv
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.collectors import TransitionCollector
from rl_mind.profiles import Profile

class SquashedGaussianActor(Actor[StochasticAction]):
    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...],
        action_dim: int,
        min_std: float = 1e-4,
    ):
        super().__init__()
        self.backbone = build_mlp([obs_dim, *hidden], output_activation=nn.ReLU())
        self.mean_layer = nn.Linear(hidden[-1], action_dim)
        self.std_layer = nn.Linear(hidden[-1], action_dim)
        self.softplus = nn.Softplus()
        self.min_std = min_std

        self.tanh = TanhTransform(cache_size=1)

    def dist(self, obs: Tensor) -> TransformedDistribution:
        features = self.backbone(obs)
        mean = self.mean_layer(features)
        std = self.softplus(self.std_layer(features)) + self.min_std
        normal = Independent(Normal(mean, std), 1)
        return TransformedDistribution(normal, [self.tanh])


    def forward(self, obs: Tensor) -> StochasticAction:
        dist = self.dist(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        self.tanh._cached_x_y = [None, None]
        return StochasticAction(value=action, log_prob=log_prob)

    def act(self, obs: Tensor) -> Tensor:
        features = self.backbone(obs)
        return torch.tanh(self.mean_layer(features))

def lap_policy(actor: SquashedGaussianActor):
    def policy(obs):
        with torch.no_grad():
            x = torch.as_tensor(obs["features"]).unsqueeze(0)
            return actor.act(x).squeeze(0).numpy()
    return policy

class EnsembleLinear(nn.Module):
    """`num` independent linear layers, evaluated in a single batched call"""
    
    def __init__(self, num: int, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num, n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(num, 1, n_out))
        bound = math.sqrt(6.0 / (n_in + n_out))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: Tensor, idx: Tensor | None = None) -> Tensor:
        weight, bias = self.weight, self.bias
        if idx is not None:
            weight, bias = weight[idx], bias[idx]
        return torch.matmul(x, weight) + bias

class EnsembleLayerNorm(nn.Module):
    """One LayerNorm per member of an ensemble, evaluated in a single call"""

    def __init__(self, num: int, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num, 1, size))
        self.bias = nn.Parameter(torch.zeros(num, 1, size))
        self.eps = eps

    def forward(self, x: Tensor, idx: Tensor | None = None) -> Tensor:
        weight, bias = self.weight, self.bias
        if idx is not None:
            weight, bias = weight[idx], bias[idx]
        x = F.layer_norm(x, x.shape[-1:], eps=self.eps)
        return x * weight + bias

class EnsembleQNetwork(nn.Module):
    """An ensemble of `num_qs` Q-networks $Q_i(s, a)$, computed in a single call"""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int, num_qs: int, layer_norm: bool = True):
        super().__init__()
        self.num_qs = num_qs
        sizes = [obs_dim + action_dim, *hidden]
        self.linears = nn.ModuleList(EnsembleLinear(num_qs, n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:]))
        self.norms = nn.ModuleList(EnsembleLayerNorm(num_qs, n_out) if layer_norm else nn.Identity() for n_out in sizes[1:])
        self.output = EnsembleLinear(num_qs, sizes[-1], 1)

    def forward(self, obs: Tensor, action: Tensor, idx: Tensor | None = None) -> Tensor:
        x = torch.cat([obs, action], dim=1)
        for linear, norm in zip(self.linears, self.norms):
            x = linear(x, idx)
            if not isinstance(norm, nn.Identity):
                x = norm(x, idx)
            x = F.relu(x)
        return self.output(x, idx).squeeze(-1)

def compute_critic_loss(
    gamma: float,
    alpha: float,
    batch: Transitions[StochasticAction],
    actor: SquashedGaussianActor,
    critic: EnsembleQNetwork,
    target_critic: EnsembleQNetwork,
    num_min_qs: int,
    backup_entropy: bool,
) -> Tensor:
    """Compute the RLPD critic loss (mean squared error over the E critics)"""

    with torch.no_grad():
        next_action = actor(batch.next_obs)
        idx = torch.randperm(critic.num_qs)[:num_min_qs]
        next_qs = target_critic(batch.next_obs, next_action.value, idx)
        next_q = next_qs.min(dim=0).values
        masks = 1.0 - batch.terminated.float()
        target_q = batch.reward + gamma * masks * next_q
        if backup_entropy:
            target_q = target_q - gamma * masks * alpha * next_action.log_prob
    qs = critic(batch.obs, batch.action.value)
    critic_loss = ((qs - target_q) ** 2).mean()
    return critic_loss

def compute_actor_loss(
    alpha: float,
    obs: Tensor,
    actor: SquashedGaussianActor,
    critic: EnsembleQNetwork,
) -> tuple[Tensor, Tensor]:
    """Compute the RLPD actor loss; also return the entropy estimate
    (needed to tune the entropy coefficient)"""

    action = actor(obs)
    qs = critic(obs, action.value)
    q = qs.mean(dim=0)
    actor_loss = (alpha * action.log_prob - q).mean()
    entropy = -action.log_prob.mean().detach()
    return actor_loss, entropy

@dataclass(frozen=True)
class RLPDConfig:
    env_name: str = "SoloKart-v0"
    seed: int = 1
    
    max_steps: int = Profile.pick(30_000, fast_test=1_200) # To be modified for real run
    n_envs: int = 1
    steps_per_round: int = 1
    updates_per_round: int = 20
    learning_starts: int = Profile.pick(5_000, fast_test=300)

    buffer_size: int = 1_000_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005

    entropy_mode: str = "auto"
    init_entropy_coef: float = 1.0
    target_entropy_scale: float = 0.5
    backup_entropy: bool = True

    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    lr: float = 3e-4

    dataset_path: str = str(common.DATA_DIR / "expert_Agent1_n30_noise0.1_seed0.npz")
    offline_ratio: float = 0.5

    num_qs: int = 10
    num_min_qs: int = 2
    critic_layer_norm: bool = True

    eval_interval: int = Profile.pick(2_000, fast_test=400)

def load_offline_buffer(
    path: str,
    obs_dim: int,
    action_dim: int,
) -> ReplayBuffer[StochasticAction]:
    
    data = np.load(path)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    action = torch.as_tensor(data["action"], dtype=torch.float32)
    reward = torch.as_tensor(data["reward"], dtype=torch.float32)
    next_obs = torch.as_tensor(data["next_obs"], dtype=torch.float32)
    terminated = torch.as_tensor(data["terminated"], dtype=torch.bool)
    n = reward.shape[0]

    assert obs.shape == (n, obs_dim) and next_obs.shape == (n, obs_dim), "wrong obs shape"
    assert action.shape == (n, action_dim), "wrong action shape"
    assert terminated.shape == (n,), "wrong terminated shape"
    for name, tensor in [("obs", obs), ("action", action), ("reward", reward), ("next_obs", next_obs)]:
        assert torch.isfinite(tensor).all(), f"non-finite values in {name}"
    assert action.abs().max() <= 1.0, "actions outside [-1, 1]"
    
    transitions = Transitions(obs=obs, action=StochasticAction(value=action, log_prob=torch.zeros(n)), reward=reward, next_obs=next_obs, terminated=terminated)
    buffer = ReplayBuffer(capacity=n)
    buffer.add(transitions)
    return buffer

class FeaturesOnly(gym.ObservationWrapper):
    """Keep only the features vector of the SoloKart observation"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["features"]

    def observation(self, obs: dict) -> np.ndarray:
        return obs["features"]

class UniformRandomActor(Actor[StochasticAction]):
    
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: Tensor) -> StochasticAction:
        n = obs.shape[0]
        action = torch.rand(n, self.action_dim) * 2.0 - 1.0
        return StochasticAction(value=action, log_prob=torch.zeros(n))
    
def run_rlpd(cfg: RLPDConfig) -> dict:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed, wrappers=[FeaturesOnly], expert=None)

    actor = SquashedGaussianActor(env.observation_dim, cfg.actor_hidden, env.action_dim)
    critic = EnsembleQNetwork(env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.num_qs, cfg.critic_layer_norm)
    target_critic = copy.deepcopy(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr)

    auto_entropy = cfg.entropy_mode == "auto"
    log_alpha = torch.log(torch.ones(1) * cfg.init_entropy_coef).requires_grad_(auto_entropy)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.lr) if auto_entropy else None
    target_entropy = -cfg.target_entropy_scale * float(env.action_dim)

    collector = TransitionCollector(env, UniformRandomActor(env.action_dim))
    buffer = ReplayBuffer(cfg.buffer_size)

    n_offline = int(cfg.batch_size * cfg.offline_ratio)
    n_online = cfg.batch_size - n_offline
    offline_buffer = (load_offline_buffer(cfg.dataset_path, env.observation_dim, env.action_dim) if n_offline > 0 else None)

    curve, next_eval = [], cfg.learning_starts

    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_round))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue
        collector.actor = actor

        for _ in range(cfg.updates_per_round):
            parts = [buffer.sample(n_online)]
            if offline_buffer is not None:
                parts.append(offline_buffer.sample(n_offline))
            batch = Transitions.cat(parts)
            alpha = log_alpha.exp().item()

            critic_loss = compute_critic_loss(cfg.gamma, alpha, batch, actor, critic, target_critic, cfg.num_min_qs, cfg.backup_entropy)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            soft_update(critic, target_critic, cfg.tau)

        actor_loss, entropy = compute_actor_loss(alpha, batch.obs, actor, critic)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if auto_entropy:
            alpha_loss = log_alpha.exp() * (entropy - target_entropy)
            
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

        if collector.steps >= next_eval:
            lap = common.evaluate_policy(lap_policy(actor), seeds=common.EVAL_SEEDS[:1])[0]
            curve.append({"env_steps": collector.steps, "lap_steps": lap.steps, "lap_finished": lap.finished})
            pbar.set_description(f"lap={lap.steps} finished={lap.finished}")
            next_eval += cfg.eval_interval
    pbar.close()
    laps = common.evaluate_policy(lap_policy(actor))
    curve.append({"env_steps": collector.steps, "lap_steps": laps[0].steps, "lap_finished": laps[0].finished})
    final = common.summarize_laps(laps)
    print(f"Final policy: {final}")

    out_dir = common.PROJECT_DIR / "results" / "rlpd" / f"seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    return final