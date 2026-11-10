"""Shared setup of the DAgger vs RLPD study on SuperTuxKart.

Everything both methods must agree on lives here, so that the two learners
are compared under strictly identical conditions: the race (one kart alone on
one track, one lap), the expert (Agent1 and its solo fix), the learner's
action space (acceleration and steering only), the observation features and
their normalization, the expert dataset, and the evaluation protocol.

Importing this module registers the gymnasium environment ``SoloKart-v0``,
which can be used directly or through ``rl_mind.VecEnv("SoloKart-v0", ...)``.
"""

import argparse
import contextlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from scipy import stats


def find_stk_src():
    """Return the ``src`` folder of the pystk_project fork, searched upwards from this file."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src"
        if (candidate / "agents" / "team1" / "agent1.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"No pystk_project 'src' folder (with agents/team1/agent1.py) found above {Path(__file__).resolve()}"
    )


STK_SRC = find_stk_src()
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
sys.path.insert(0, str(STK_SRC))

with contextlib.chdir(STK_SRC / "main"):  # Agent2 loads its config at import time through a path relative to src/main.
    from agents.team1.agent1 import Agent1
    from agents.team2.agent2 import Agent2
    from agents.team3.agent3 import Agent3
    from agents.team4.agent4 import Agent4
    from agents.team5.agent5 import Agent5
    from pystk2_gymnasium.definitions import CameraMode
    from pystk2_gymnasium.envs import AgentSpec, STKRaceMultiEnv

AgentSpec.__hash__ = lambda self: hash((self.name, self.rank_start, self.use_ai, self.camera_mode))

TRACK = "abyss"
MAX_EPISODE_STEPS = 1000
EXPERT_NAME = "Agent1"
EXPERT_LAP_STEPS = 570

AGENTS = {
    "Agent1": Agent1,
    "Agent2": Agent2,
    "Agent3": Agent3,
    "Agent4": Agent4,
    "Agent5": Agent5,
}
ACTION_OVERRIDES = {
    "full": {},
    "no-fire": {"fire": False},
    "continuous": {"brake": False, "drift": False, "fire": False, "nitro": False, "rescue": False},
}
PHANTOM_KART = (0.0, 0.0, -1000.0)

N_PATHS = 6
N_ITEMS = 5
N_ITEM_TYPES = 7
FEATURE_DIM = 13 + 9 * N_PATHS + (3 + N_ITEM_TYPES) * N_ITEMS
ACTION_DIM = 2
FEATURE_STATS_PATH = DATA_DIR / "feature_stats.npz"

ENV_ID = "SoloKart-v0"



def make_stk_env(track=TRACK):
    """Create the raw single-kart, one-lap pystk2 race (dict observations and actions, no rendering)."""
    spec = AgentSpec(name="solo", rank_start=0, use_ai=False, camera_mode=CameraMode.OFF)
    return STKRaceMultiEnv(agents=[spec], track=track, render_mode=None, num_kart=1, laps=1)


def add_phantom_kart(raw_obs):
    """Return the observation with one opponent placed 1 km behind the kart.

    Agent1 reads ``karts_position[0]`` whenever it holds a cake, bowling ball,
    plunger, rubber ball or parachute, which raises an IndexError in a solo race
    because the list is empty. An opponent that far behind is outside every
    distance window of Agent1, so it never fires at it and never steers around
    it: the agent drives as if it were leading alone. Other agents already
    handle an empty list and receive the observation unchanged, since some of
    them (Agent4's opponent avoidance) keep internal state as soon as an
    opponent is listed.
    """
    if len(raw_obs["karts_position"]) > 0:
        return raw_obs
    return {**raw_obs, "karts_position": (np.array(PHANTOM_KART, dtype=np.float32),)}


OBS_ADAPTERS = {"Agent1": add_phantom_kart}


def agent_action(agent, agent_name, raw_obs):
    """Full dict action chosen by a pystk_project agent, with its solo fix applied."""
    adapt = OBS_ADAPTERS.get(agent_name, lambda obs: obs)
    return agent.choose_action(adapt(raw_obs))


def to_stk_action(action):
    """Convert a learner action in [-1, 1]^2 (acceleration, steer) into the game's dict action.

    The acceleration is mapped from [-1, 1] to the game's [0, 1] range and every
    boolean command is off, which is the action space shared by DAgger and RLPD.
    """
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    return {
        "acceleration": float((a[0] + 1.0) / 2.0),
        "steer": float(a[1]),
        **ACTION_OVERRIDES["continuous"],
    }


def from_stk_action(stk_action):
    """Project a dict action of the game onto the learner's action space [-1, 1]^2.

    This is the inverse of ``to_stk_action`` on the two continuous commands; the
    boolean commands are dropped, which is harmless for Agent1 (same lap time).
    """
    acceleration = float(np.asarray(stk_action["acceleration"]).reshape(-1)[0])
    steer = float(np.asarray(stk_action["steer"]).reshape(-1)[0])
    return np.clip(np.array([2.0 * acceleration - 1.0, steer], dtype=np.float32), -1.0, 1.0)



def featurize(raw_obs):
    """Fixed-size feature vector of a raw observation, in the kart's frame (z front, x left, y up).

    It keeps what a solo driver needs and drops opponent and power-up
    information: kinematics and position on the track (13 values), the next
    ``N_PATHS`` track segments (start, end, width, distances: 9 values each) and
    the ``N_ITEMS`` closest items on the track (position and one-hot type: 10
    values each), so that the learner sees the bananas Agent1 dodges and the
    boxes it steers to. Missing segments or items are zero-padded.
    """
    def vec(key):
        return np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)

    parts = [
        vec("velocity"), vec("front"), vec("center_path"), vec("center_path_distance"),
        vec("max_steer_angle"), vec("distance_down_track"),
        np.array([raw_obs["jumping"]], dtype=np.float32),
    ]

    paths = np.zeros((N_PATHS, 9), dtype=np.float32)
    n = min(N_PATHS, len(raw_obs["paths_start"]))
    for i in range(n):
        paths[i] = np.concatenate([
            np.asarray(raw_obs["paths_start"][i], dtype=np.float32).reshape(-1),
            np.asarray(raw_obs["paths_end"][i], dtype=np.float32).reshape(-1),
            np.asarray(raw_obs["paths_width"][i], dtype=np.float32).reshape(-1),
            np.asarray(raw_obs["paths_distance"][i], dtype=np.float32).reshape(-1),
        ])

    items = np.zeros((N_ITEMS, 3 + N_ITEM_TYPES), dtype=np.float32)
    n = min(N_ITEMS, len(raw_obs["items_position"]))
    for i in range(n):
        items[i, :3] = np.asarray(raw_obs["items_position"][i], dtype=np.float32).reshape(-1)
        items[i, 3 + int(raw_obs["items_type"][i])] = 1.0

    return np.concatenate(parts + [paths.reshape(-1), items.reshape(-1)])


@dataclass
class FeatureStats:
    """Mean and scale used to normalize the features, shared by both learners."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, features):
        """Standardize features; constant features are only centered."""
        return ((features - self.mean) / self.std).astype(np.float32)


def compute_feature_stats(n_episodes=5, noise_std=0.3, seed=0, path=FEATURE_STATS_PATH):
    """Estimate the feature statistics on noisy expert laps and save them.

    Noise is added to the executed actions so that the statistics also cover
    states slightly off the expert's single deterministic line. Features whose
    standard deviation is below 1e-6 (always constant) get a scale of 1.
    """
    data = collect_expert_dataset(n_episodes, noise_std=noise_std, seed=seed, normalize=False, path=None)
    feats = np.concatenate([data["obs"], data["next_obs"]])
    std = feats.std(axis=0)
    fs = FeatureStats(mean=feats.mean(axis=0), std=np.where(std < 1e-6, 1.0, std))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=fs.mean, std=fs.std)
    return fs


def load_feature_stats(path=FEATURE_STATS_PATH):
    """Load the shared feature statistics, computing and saving them on first use."""
    if not Path(path).is_file():
        print(f"{path} not found: computing feature statistics from noisy expert laps...", flush=True)
        return compute_feature_stats(path=path)
    with np.load(path) as f:
        return FeatureStats(mean=f["mean"], std=f["std"])



class SoloKartEnv(gym.Env):
    """One kart alone on one track for one lap, with the learners' observation and action spaces.

    Observations are dictionaries with ``features`` (normalized vector of size
    ``FEATURE_DIM``) and, when an expert is attached, ``expert_action`` (the
    expert's action in this very state, in [-1, 1]^2). The expert is run at
    every step on the true trajectory, so its internal state stays consistent
    even when the executed actions come from the learner: this is what DAgger
    needs for its labels. Actions are (acceleration, steer) in [-1, 1]^2.

    The reward is the pystk2 one: progress / 10 - 0.1 per step + 10 at the
    finish line (the position term is always 0 when alone). ``terminated`` means
    the lap is finished (no bootstrap); ``truncated`` means the step budget is
    exhausted (bootstrap). ``info`` carries the raw observation and the step
    count for code that does not go through rl_mind.
    """

    metadata = {"render_modes": []}

    def __init__(self, track=TRACK, max_steps=MAX_EPISODE_STEPS, expert=EXPERT_NAME, normalize=True):
        super().__init__()
        self.track = track
        self.max_steps = max_steps
        self.expert_name = expert
        self.stats = load_feature_stats() if normalize else None
        spaces = {"features": gym.spaces.Box(-np.inf, np.inf, (FEATURE_DIM,), np.float32)}
        if expert is not None:
            spaces["expert_action"] = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.stk = make_stk_env(track)
        self.expert = None
        self.raw_obs = None
        self.steps = 0

    def _observe(self):
        """Build the observation of the current state and query the expert on it."""
        feats = featurize(self.raw_obs)
        obs = {"features": self.stats.normalize(feats) if self.stats else feats}
        if self.expert is not None:
            obs["expert_action"] = from_stk_action(agent_action(self.expert, self.expert_name, self.raw_obs))
        return obs

    def reset(self, *, seed=None, options=None):
        """Start a new lap with a fresh expert and return its first observation."""
        super().reset(seed=seed)
        race_seed = None if seed is None else int(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        obs, _ = self.stk.reset(seed=race_seed)
        self.raw_obs = obs["0"]
        self.steps = 0
        if self.expert_name is not None:
            self.expert = AGENTS[self.expert_name](self.stk, path_lookahead=3)
        return self._observe(), {"steps": 0, "finished": False}

    def step(self, action):
        """Apply a learner action for one step of 0.1 s."""
        obs, reward, terminated, _, _ = self.stk.step({"0": to_stk_action(action)})
        self.raw_obs = obs["0"]
        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"steps": self.steps, "finished": bool(terminated)}
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        """Shut down the SuperTuxKart process."""
        self.stk.close()


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=SoloKartEnv)



def collect_expert_dataset(n_episodes, noise_std=0.0, seed=0, normalize=True, path=None):
    """Roll out the expert and return its transitions as NumPy arrays, optionally saved to ``.npz``.

    The executed action is the expert's action plus Gaussian noise of standard
    deviation ``noise_std``, clipped to [-1, 1]; without noise, every lap is the
    same deterministic trajectory. Both the executed action (``action``, what an
    off-policy learner like RLPD must use) and the expert's clean label
    (``expert_action``, what behavioural cloning and DAgger use) are stored,
    together with ``obs``, ``next_obs``, ``reward``, ``terminated``,
    ``truncated`` and the ``episode`` index of each transition.
    """
    env = SoloKartEnv(normalize=normalize)
    rng = np.random.default_rng(seed)
    keys = ("obs", "next_obs", "action", "expert_action", "reward", "terminated", "truncated", "episode")
    data = {key: [] for key in keys}
    try:
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            while not done:
                label = obs["expert_action"]
                action = np.clip(label + noise_std * rng.standard_normal(ACTION_DIM), -1.0, 1.0)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                for key, value in zip(keys, (obs["features"], next_obs["features"], action, label,
                                             reward, terminated, truncated, episode)):
                    data[key].append(value)
                obs, done = next_obs, terminated or truncated
    finally:
        env.close()

    arrays = {key: np.asarray(values) for key, values in data.items()}
    for key in ("obs", "next_obs", "action", "expert_action", "reward"):
        arrays[key] = arrays[key].astype(np.float32)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, noise_std=noise_std, seed=seed, **arrays)
    return arrays


def load_expert_dataset(path):
    """Load a dataset saved by ``collect_expert_dataset`` as a dict of NumPy arrays."""
    with np.load(path) as f:
        return {key: f[key] for key in f.files}



EVAL_SEEDS = tuple(range(1000, 1010))


@dataclass
class LapResult:
    """Outcome of one evaluation lap."""

    seed: int
    finished: bool
    steps: int
    ret: float


def evaluate_policy(policy, seeds=EVAL_SEEDS, max_steps=MAX_EPISODE_STEPS):
    """Drive one evaluation lap per seed with a deterministic policy and return the outcomes.

    ``policy`` receives the observation dictionary of ``SoloKartEnv`` and
    returns an action in [-1, 1]^2; the string ``"expert"`` drives with the
    expert's own action, which gives the reference lap time. A lap still
    running after ``max_steps`` counts as unfinished with ``steps = max_steps``.
    """
    env = SoloKartEnv(max_steps=max_steps)
    results = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            ret, done, info = 0.0, False, {"steps": 0, "finished": False}
            while not done:
                action = obs["expert_action"] if policy == "expert" else policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ret += reward
                done = terminated or truncated
            results.append(LapResult(seed, info["finished"], info["steps"], ret))
    finally:
        env.close()
    return results


def summarize_laps(results):
    """Finish rate and lap-time statistics (finished laps only), as recommended in the course."""
    steps = np.array([r.steps for r in results if r.finished], dtype=float)
    summary = {"laps": len(results), "finished": int(steps.size)}
    if steps.size:
        summary.update(mean=steps.mean(), iqm=stats.trim_mean(steps, 0.25),
                       q10=np.percentile(steps, 10), q90=np.percentile(steps, 90))
    if steps.size >= 2:
        summary.update(std=steps.std(ddof=1), ci=bootstrap_ci(steps))
    return summary



N_BOOTSTRAP = 10_000
ALPHA = 0.05


def bootstrap_ci(sample_a, sample_b=None, seed=0):
    """Percentile bootstrap 95% confidence interval of a mean or of a difference of means.

    With one sample the interval is for its mean; with two independent samples
    it is for mean(a) - mean(b), each sample being resampled separately.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(sample_a, dtype=float)
    boot = rng.choice(a, size=(N_BOOTSTRAP, a.size)).mean(axis=1)
    if sample_b is not None:
        b = np.asarray(sample_b, dtype=float)
        boot = boot - rng.choice(b, size=(N_BOOTSTRAP, b.size)).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def welch_test(sample_a, sample_b):
    """Two-sided Welch's t-test returning the statistic, degrees of freedom and p-value.

    Written out explicitly so that constant samples are handled exactly: when
    both samples have zero variance there is no noise left, and the p-value is
    1 if the two constants are equal and 0 otherwise.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    var_a = a.var(ddof=1) / a.size
    var_b = b.var(ddof=1) / b.size
    diff = a.mean() - b.mean()
    se2 = var_a + var_b
    if se2 == 0.0:
        return math.nan, math.nan, 1.0 if diff == 0.0 else 0.0
    t_stat = diff / math.sqrt(se2)
    df = se2**2 / (var_a**2 / (a.size - 1) + var_b**2 / (b.size - 1))
    return t_stat, df, float(2.0 * stats.t.sf(abs(t_stat), df))



def main():
    """Prepare the shared files: feature statistics and the expert datasets."""
    parser = argparse.ArgumentParser(description="Prepare the files shared by DAgger and RLPD.")
    parser.add_argument("--feature-stats", action="store_true", help="(re)compute data/feature_stats.npz")
    parser.add_argument("--dataset", type=int, metavar="N_EPISODES", help="collect an expert dataset")
    parser.add_argument("--noise", type=float, default=0.0, help="std of the noise on executed actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="drive the expert through SoloKartEnv")
    args = parser.parse_args()

    if args.feature_stats:
        compute_feature_stats()
        print(f"Feature statistics written to {FEATURE_STATS_PATH}")
    if args.dataset:
        path = DATA_DIR / f"expert_{EXPERT_NAME}_n{args.dataset}_noise{args.noise:g}_seed{args.seed}.npz"
        data = collect_expert_dataset(args.dataset, noise_std=args.noise, seed=args.seed, path=path)
        finished = int(data["terminated"].sum())
        print(f"{len(data['reward'])} transitions, {finished}/{args.dataset} laps finished, written to {path}")
    if args.check:
        summary = summarize_laps(evaluate_policy("expert"))
        print(f"Expert through SoloKartEnv: {summary['finished']}/{summary['laps']} laps finished, "
              f"mean {summary.get('mean', math.nan):.1f} steps (reference {EXPERT_LAP_STEPS})")


if __name__ == "__main__":
    main()