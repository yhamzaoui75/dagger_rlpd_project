"""Solo lap-time benchmark of the pystk_project expert agents.

Each agent drives alone on one track for the same list of seeded one-lap races,
restricted to the action space the learning agents will use, so that the
expert's lap time is a reference the learners can actually reach. The script
reports reliability and lap-time statistics, runs Welch's t-test on every pair
of eligible agents with a Bonferroni correction, and writes the raw per-race
results to a CSV file.
"""

import argparse, csv, itertools, math, random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from common import (
    ACTION_OVERRIDES,
    AGENTS,
    ALPHA,
    DATA_DIR,
    TRACK,
    agent_action,
    bootstrap_ci,
    make_stk_env,
    welch_test,
)


@dataclass
class RaceResult:
    """Outcome of one solo race."""

    agent: str
    action_space: str
    seed: int
    finished: bool
    steps: int
    race_time: float
    error: str


def run_race(env, agent_name, seed, max_steps, action_space):
    """Drive one seeded solo race with a fresh agent and return its outcome.

    The agent still computes its full action, but the components outside the
    chosen action space are overwritten before reaching the game, exactly as
    they will be for the learners. The global Python and NumPy generators are
    seeded as well as the race, so a race is reproducible. An exception raised
    by the agent ends the race as a failure whose message is kept, and a race
    still running after the step budget is a failure too; neither is ever
    dropped.
    """
    random.seed(seed)
    np.random.seed(seed)
    agent = AGENTS[agent_name](env, path_lookahead=3)
    override = ACTION_OVERRIDES[action_space]
    obs, _ = env.reset(seed=seed)

    def result(finished, steps, race_time, error):
        return RaceResult(agent_name, action_space, seed, finished, steps, race_time, error)

    for step in range(1, max_steps + 1):
        try:
            action = agent_action(agent, agent_name, obs["0"])
        except Exception as exc:
            return result(False, step - 1, math.nan, f"{type(exc).__name__}: {exc}")
        obs, _, terminated, _, _ = env.step({"0": {**action, **override}})
        if terminated:
            return result(True, step, float(env.world.karts[0].finish_time), "")
    return result(False, max_steps, math.nan, "step budget exhausted")


def evaluate_agent(env, agent_name, seeds, max_steps, action_space):
    """Run one race per seed for an agent and print its failures grouped by cause."""
    print(f"Evaluating {agent_name} on {len(seeds)} solo races...", flush=True)
    results = [run_race(env, agent_name, seed, max_steps, action_space) for seed in seeds]
    causes = {}
    for r in results:
        if not r.finished:
            causes.setdefault(r.error, []).append(r.seed)
    for cause, failed_seeds in causes.items():
        print(f"  {len(failed_seeds)} failure(s), {cause}: seeds {failed_seeds}", flush=True)
    return results


def summarize(results):
    """Per-agent finish rate and lap-time statistics computed on finished races only."""
    steps = np.array([r.steps for r in results if r.finished], dtype=float)
    summary = {"runs": len(results), "finished": steps.size}
    if steps.size >= 2:
        summary.update(
            mean=steps.mean(),
            std=steps.std(ddof=1),
            iqm=stats.trim_mean(steps, 0.25),
            q10=np.percentile(steps, 10),
            q90=np.percentile(steps, 90),
            ci=bootstrap_ci(steps),
            values=steps,
        )
    return summary


def print_summary(summaries, action_space):
    """Print the reliability and lap-time table, fastest eligible agent first."""
    print("\n" + "=" * 92)
    print(f" SOLO LAP TIMES, action space '{action_space}' (steps of 0.1 s, finished races only)")
    print("=" * 92)
    print(f"{'Agent':<8} | {'Finished':>8} | {'Mean':>7} | {'95% CI (bootstrap)':>19} | "
          f"{'Std':>5} | {'IQM':>7} | {'[q10, q90]':>15}")
    print("-" * 92)
    ordered = sorted(summaries.items(), key=lambda kv: kv[1].get("mean", math.inf))
    for name, s in ordered:
        finished = f"{s['finished']}/{s['runs']}"
        if "mean" not in s:
            print(f"{name:<8} | {finished:>8} | {'n/a':>7} |")
            continue
        ci = f"[{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]"
        quantiles = f"[{s['q10']:.0f}, {s['q90']:.0f}]"
        print(f"{name:<8} | {finished:>8} | {s['mean']:>7.1f} | {ci:>19} | "
              f"{s['std']:>5.1f} | {s['iqm']:>7.1f} | {quantiles:>15}")


def compare_all_pairs(summaries):
    """Welch's t-test with Bonferroni correction on every pair of eligible agents.

    An agent is eligible only if it finished every race: an expert that crashes
    or gets stuck is unusable for DAgger, whatever its lap time. Testing every
    pair, rather than only the empirical best against the others, keeps the
    family of tests fixed before looking at the data.
    """
    eligible = sorted(
        (name for name, s in summaries.items() if s["finished"] == s["runs"] and "mean" in s),
        key=lambda name: summaries[name]["mean"],
    )
    excluded = sorted(set(summaries) - set(eligible))
    pairs = list(itertools.combinations(eligible, 2))
    n_tests = len(pairs)

    print("\n" + "=" * 92)
    print(f" PAIRWISE WELCH T-TESTS, Bonferroni over {n_tests} tests (alpha = {ALPHA})")
    print("=" * 92)
    if excluded:
        print(f"Excluded (at least one unfinished race): {', '.join(excluded)}")
    print(f"{'Pair (A - B)':<18} | {'Diff':>6} | {'95% CI (bootstrap)':>19} | {'t':>7} | {'df':>5} | "
          f"{'p':>8} | {'p Bonf.':>8}")
    print("-" * 92)
    p_bonf = {}
    for a, b in pairs:
        va, vb = summaries[a]["values"], summaries[b]["values"]
        t_stat, df, p = welch_test(va, vb)
        p_bonf[(a, b)] = min(1.0, p * n_tests)
        low, high = bootstrap_ci(va, vb)
        mark = " *" if p_bonf[(a, b)] < ALPHA else ""
        print(f"{a + ' - ' + b:<18} | {va.mean() - vb.mean():>6.1f} | {f'[{low:.1f}, {high:.1f}]':>19} | "
              f"{t_stat:>7.2f} | {df:>5.1f} | {p:>8.4f} | {p_bonf[(a, b)]:>8.4f}{mark}")
    return eligible, p_bonf


def print_decision(summaries, eligible, p_bonf):
    """Apply the selection rule: the fastest eligible agent, if it beats every other one.

    When every eligible agent produced the same lap time in all its races, the
    environment is deterministic under this action space: the ranking is then
    exact and the tests above carry no additional information.
    """
    print("\n" + "=" * 92)
    if not eligible:
        print("No agent finished every race: no expert can be selected.")
        return
    best = eligible[0]
    if all(summaries[name]["std"] == 0.0 for name in eligible):
        print("Every agent repeated the same lap time in all races: the environment is deterministic\n"
              "under this action space, so the ranking is exact and needs no statistical test.")
        ranking = ", ".join(f"{name} {summaries[name]['mean']:.0f}" for name in eligible)
        print(f"Ranking (steps): {ranking}")
        tied = [name for name in eligible[1:] if summaries[name]["mean"] == summaries[best]["mean"]]
        if tied:
            print(f"No single expert: {best} is tied with {', '.join(tied)}.")
        else:
            print(f"Selected expert: {best}")
        return
    ties = [other for other in eligible[1:] if p_bonf[(best, other)] >= ALPHA]
    if ties:
        print(f"Fastest mean: {best}, but not significantly faster than {', '.join(ties)}.")
    else:
        print(f"Selected expert: {best} (significantly faster than every other eligible agent).")


def save_csv(path, results):
    """Write one row per race so that tables and figures can be rebuilt without rerunning."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RaceResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    print(f"\nRaw results written to {path}")


def parse_args():
    """Command-line options of the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", default=list(AGENTS), choices=list(AGENTS))
    parser.add_argument("--track", default=TRACK)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--action-space", choices=list(ACTION_OVERRIDES), default="continuous",
                        help="continuous: acceleration and steer only (learners' setting); "
                             "no-fire: every action except items; full: all seven actions")
    parser.add_argument("--out", default=None,
                        help="CSV path, defaults to my_project/data/expert_benchmark_<action-space>.csv")
    args = parser.parse_args()
    if args.out is None:
        args.out = str(DATA_DIR / f"expert_benchmark_{args.action_space}.csv")
    return args


def main():
    """Run the benchmark, print the tables and save the raw results."""
    args = parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.runs))
    env = make_stk_env(args.track)
    all_results = []
    summaries = {}
    try:
        for name in args.agents:
            results = evaluate_agent(env, name, seeds, args.max_steps, args.action_space)
            all_results.extend(results)
            summaries[name] = summarize(results)
    finally:
        env.close()

    print_summary(summaries, args.action_space)
    eligible, p_bonf = compare_all_pairs(summaries)
    print_decision(summaries, eligible, p_bonf)
    save_csv(args.out, all_results)


if __name__ == "__main__":
    main()