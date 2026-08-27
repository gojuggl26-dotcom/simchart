"""ゲートの誤検出率 (帰無仮説下で落ちる確率) をシードを変えて実測する。

S0 は正しい実装であることが判っているので、ここで落ちたゲートは全て偽陽性で
ある。閾値が緩すぎて欠陥を見逃すのと同じくらい、閾値が厳しすぎて正しい実装を
落とすのも困る (段階が 14 あるので、1 回 5% でも通しで 50% 以上どこかが赤くなる)。

    uv run python scripts/seed_sweep.py 40
    uv run python scripts/seed_sweep.py 40 --n-days 100 # 速く回したいとき

欠陥を仕込んだときに落ちることの確認は tests/test_gates_detect_defects.py 側。
両方揃って初めてゲートに意味がある。
"""

from __future__ import annotations

import argparse
import collections
import time

from simchart import Config, run
from simchart.validation import evaluate
from simchart.validation.gates import S0_GATES
from simchart.validation.suite import collect_errors, run_all

#: 実行時系 (決定性など) を除いた、統計量に基づくゲート。
STAT_GATES = tuple(g.name for g in S0_GATES if not g.metric_path.startswith("runtime."))

#: 統計ゲートの評価に集中するため、実行時系は合格として与える。
#: 決定性・RNG 安定性は tests/ 側で別途固定してある。
_RUNTIME_STUB = {
    "pipeline": {"completed": True},
    "determinism": {"bitwise_identical": True},
    "rng_stability": {"unchanged": True, "streams_distinct": True},
    "artifacts": {"metrics_json_ok": True},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_seeds", type=int, nargs="?", default=40)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--n-days", type=int, default=None)
    parser.add_argument("--steps-per-day", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.n_days is not None:
        overrides["n_days"] = args.n_days
    if args.steps_per_day is not None:
        overrides["steps_per_day"] = args.steps_per_day

    fails: collections.Counter[str] = collections.Counter()
    values: dict[str, list] = collections.defaultdict(list)
    runs_with_any_failure = 0
    started = time.perf_counter()

    for k in range(args.n_seeds):
        config = Config(seed=args.seed0 + k, **overrides)
        metrics = run_all(run(config), config)
        metrics["runtime"] = dict(_RUNTIME_STUB)
        metrics["runtime"]["validation"] = {"all_callable": not collect_errors(metrics)}

        verdicts = {g.name: g for g in evaluate(S0_GATES, metrics)}
        failed_here = [name for name in STAT_GATES if not verdicts[name].passed]
        for name in failed_here:
            fails[name] += 1
        runs_with_any_failure += bool(failed_here)
        for name in STAT_GATES:
            values[name].append(verdicts[name].value)

        if (k + 1) % 10 == 0:
            note = f"  直近の不合格: {failed_here}" if failed_here else ""
            print(f"  {k + 1}/{args.n_seeds} ({time.perf_counter() - started:.0f} 秒){note}",
                  flush=True)

    config = Config(**overrides)
    print()
    print(f"{args.n_seeds} シード / {config.n_days} 日 x {config.steps_per_day} ステップ")
    print(f"{'ゲート':<24}{'不合格':>7}{'率':>9}   値の範囲")
    print("-" * 78)
    for name in STAT_GATES:
        numeric = [v for v in values[name] if isinstance(v, (int, float)) and not isinstance(v, bool)]
        span = f"[{min(numeric):+.4g}, {max(numeric):+.4g}]" if numeric else "(bool)"
        print(f"{name:<24}{fails[name]:>7}{fails[name] / args.n_seeds:>8.1%}   {span}")
    print("-" * 78)
    print(f"1 つ以上落ちた実行: {runs_with_any_failure}/{args.n_seeds}"
          f" = {runs_with_any_failure / args.n_seeds:.1%}")
    print(f"総経過 {time.perf_counter() - started:.0f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
