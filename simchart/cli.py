"""コマンドラインインターフェース。

    python -m simchart.cli run --config configs/s0.yaml --stage S0
    python -m simchart.cli validate --stage S0
    python -m simchart.cli compare --stages S0 S1

``run`` は「実行 -> 検証 -> ゲート判定 -> 永続化 -> 書き出しの確認 -> ゲート再判定」
という順で動く。最後の 2 段は ``artifacts_written`` ゲートのためで、書いたつもりで
終わらせないために metrics.json を読み直してから最終版を書く。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import Config
from .pipeline import (
    BASELINE_STAGE,
    baseline_invariance_check,
    determinism_check,
    rng_diffusion_check,
    rng_stability_check,
    run as run_pipeline,
    scale_invariance_check,
)
from .report import (
    compare_stages,
    load_metrics,
    make_plots,
    verify_metrics_file,
    write_metrics,
)
from .validation import evaluate, gates_for, summarize
from .validation.base import jsonable
from .validation.suite import collect_errors, run_all

__all__ = ["main"]


# ---------------------------------------------------------------------------
def _build_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config) if args.config else Config()
    overrides: dict[str, Any] = {}
    if args.stage:
        overrides["stage"] = args.stage
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.n_days is not None:
        overrides["n_days"] = args.n_days
    if args.steps_per_day is not None:
        overrides["steps_per_day"] = args.steps_per_day
    return config.replace(**overrides) if overrides else config


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.4e}"
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        # ★jsonable を通す。検証関数が numpy 配列を返す枝があり (S4 のスペクトル
        # 検定の ratios など)、生の dict を json.dumps すると TypeError で
        # **ゲート表示の途中で実行が落ちる**。表示のために解析を落とすのは本末転倒。
        return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))[:60]
    if isinstance(value, np.ndarray):
        return f"<配列 {value.shape}>"
    return str(value)


def _intraday_gph_pair(result, config: Config) -> tuple[float | None, float | None]:
    """日内バー |r| の GPH d を (生, 真値 φ 除去後) の組で返す (S4)。

    ★真値 φ で除く。推定 φ̂ ではなく真値を使うのは、ここで測りたいのが
    「季節性が推定を汚す量」であって「推定器の性能」ではないから。φ̂ の性能は
    別の枝 (seasonality.deseasonalization.recovery) が測る。
    """
    import numpy as np

    from .layers.l0_calendar import build_calendar
    from .rng import RNGRegistry
    from .validation.memory import gph_estimator
    from .validation.seasonality import deseasonalize, true_phi_bars

    obs = result.observation
    bars = obs.to_bars(config.validation.primary_bar_sec)
    r_2d = bars.returns_2d()
    calendar = build_calendar(config, RNGRegistry(config.seed))
    steps_per_day = (
        int(round(obs.session_seconds / obs.step_seconds)) if obs.step_seconds else None
    )
    truth = true_phi_bars(calendar, r_2d.shape[1], steps_per_day=steps_per_day)
    bwe = config.validation.gph_bandwidth_exponent
    raw = gph_estimator(np.abs(r_2d).ravel(), bandwidth_exponent=bwe).get("d")
    if truth["status"] != "ok":
        return raw, None
    dsn = gph_estimator(
        np.abs(deseasonalize(r_2d, np.asarray(truth["value"]))).ravel(),
        bandwidth_exponent=bwe,
    ).get("d")
    return raw, dsn


def _run_multiseed(config: Config, n_seeds: int) -> dict[str, Any]:
    """ノイズの大きい指標をシードを変えて測り、中央値・IQR を返す (S3 指示書 §8)。

    Hill α は 5000 日でも上位 5% が 250 観測しかなく単一シードで ±0.5 ばらつく。
    対象は hill_alpha / leverage_corr / jv_share / skewness の 4 つ (経路統計)。
    分散予算・補償・相関実測・決定性は単一シードで判定できるので対象外。
    追加シードでは対象指標だけを測る (フル検証スイートは回さない)。
    """
    import numpy as np
    from scipy import stats as sp_stats

    from .validation.memory import leverage_function
    from .validation.scaling import realized_variance
    from .validation.tails import bns_jump_test, hill_by_scale, hill_estimator

    per_seed: dict[str, list[float]] = {
        "hill_alpha": [], "leverage_corr": [], "jv_share": [], "skewness_daily": [],
        "hill_scale_slope": [], "gph_d": [],
        # S4: 季節性が**日内**バーの長期記憶推定を汚す量。1 経路では GPH の
        # SE 0.013 に埋もれる (実測バイアス +0.017) ため中央値で判定する。
        # 日次 gph_d は φ_σ の二乗正規化により汚染を受けない (別枝で確認)。
        "gph_d_intraday_raw": [], "gph_d_intraday_deseason": [], "gph_bias_intraday": [],
    }
    seeds = [config.seed + i for i in range(n_seeds)]
    for i, seed in enumerate(seeds):
        seed_config = config.replace(seed=seed)
        result = run_pipeline(seed_config)
        obs = result.observation
        r_daily = obs.to_bars(obs.session_seconds).returns()
        steps_per_day = int(round(obs.session_seconds / obs.step_seconds))
        step_r = np.diff(obs.log_price)
        rv_daily = realized_variance(step_r, steps_per_day)

        hill = hill_estimator(r_daily, 0.05, "both")
        per_seed["hill_alpha"].append(hill.get("alpha"))
        lev = leverage_function(r_daily, rv_daily, horizons=(0, 1))
        per_seed["leverage_corr"].append(lev.get("corr_r_rv_h1"))
        bns = bns_jump_test(step_r, steps_per_day)
        per_seed["jv_share"].append(bns.get("jv_share"))
        per_seed["skewness_daily"].append(float(sp_stats.skew(r_daily, bias=False)))
        hbs = hill_by_scale(r_daily)
        per_seed["hill_scale_slope"].append(hbs.get("slope_vs_log_scale"))
        from .validation.memory import gph_estimator

        per_seed["gph_d"].append(
            gph_estimator(np.abs(r_daily), config.validation.daily_gph_bandwidth_exponent).get("d")
        )
        if config.enable_seasonality:
            raw, dsn = _intraday_gph_pair(result, seed_config)
            per_seed["gph_d_intraday_raw"].append(raw)
            per_seed["gph_d_intraday_deseason"].append(dsn)
            per_seed["gph_bias_intraday"].append(
                raw - dsn if raw is not None and dsn is not None else None
            )
        del result, obs, step_r, rv_daily
        print(f"      シード {seed} ({i + 1}/{n_seeds}) 完了", flush=True)

    out: dict[str, Any] = {"n_seeds": n_seeds, "seeds": seeds}
    for name, values in per_seed.items():
        clean = [v for v in values if v is not None]
        if not clean:
            out[name] = {"median": None, "values": values}
            continue
        arr = np.array(clean, dtype=np.float64)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        out[name] = {
            "median": float(np.median(arr)),
            "iqr": q3 - q1,
            "q1": q1,
            "q3": q3,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": values,
        }
    return out


def _print_gates(gate_results, summary: Mapping[str, Any]) -> None:
    name_width = max(len(g.name) for g in gate_results)
    print()
    print("ゲート判定")
    print("-" * (name_width + 58))
    for g in gate_results:
        mark = "PASS" if g.passed else ("FAIL" if g.critical else "WARN")
        tag = "" if g.critical else " (warning)"
        line = f"  [{mark}] {g.name.ljust(name_width)}  {_fmt(g.value)}"
        print(line)
        if not g.passed:
            print(f"         期待: {g.threshold}{tag}")
            if g.error:
                print(f"         理由: {g.error}")
    print("-" * (name_width + 58))
    print(
        f"  合格 {summary['n_passed']}/{summary['n_gates']}"
        f" / critical {summary['n_critical']} 件中 "
        f"{'全て合格' if summary['all_critical_passed'] else '不合格あり: ' + ', '.join(summary['failed_critical'])}"
    )


def _print_key_metrics(metrics: Mapping[str, Any]) -> None:
    def get(path: str) -> Any:
        node: Any = metrics
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return None
            node = node[part]
        return node

    rows = [
        ("基準粒度 (秒)", get("series.primary_bar_sec")),
        ("リターン本数", get("series.n_primary_returns")),
        ("尖度", get("tails.moments.kurtosis")),
        ("歪度", get("tails.moments.skewness")),
        ("Hill alpha (k=5%)", get("tails.hill.alpha")),
        ("Hill 不安定度", get("tails.hill_profile.instability")),
        ("QQ の R^2", get("tails.qq_normal.r2")),
        ("ACF(r) ラグ1", get("memory.acf_r.lag1")),
        ("ACF(|r|) ラグ1", get("memory.acf_abs_r.lag1")),
        ("Ljung-Box p (ラグ20)", get("memory.ljung_box_r.pvalue_primary")),
        ("GPH d (|r|)", get("memory.gph_abs_r.d")),
        ("local Whittle d (|r|)", get("memory.local_whittle_abs_r.d")),
        ("分散比 max|VR-1|", get("scaling.variance_ratio.max_abs_dev")),
        ("尖度のスケール依存 max|k-3|", get("scaling.kurtosis_by_scale.max_abs_dev_from_3_gated")),
        ("zeta_q 直線性 R^2", get("scaling.zeta_q.r2")),
        ("zeta_q の傾き", get("scaling.zeta_q.slope")),
        ("signature plot 最大乖離", get("scaling.signature_plot.max_rel_dev")),
        ("ADF p (log P)", get("scaling.adf.log_price_pvalue")),
        ("ADF p (r)", get("scaling.adf.returns_pvalue")),
        ("日次 尖度", get("daily.moments.kurtosis")),
        ("日次 ACF(|r|) ラグ1", get("daily.acf_abs_r.lag1")),
        ("日次 GPH d (|r|)", get("daily.gph_abs_r.d")),
        ("日次 |r| ACF べき則 R^2", get("daily.acf_abs_r_powerlaw.r2")),
        ("日次 zeta_q 直線性 R^2", get("daily.zeta_q.r2")),
        ("尖度の減衰傾き (日次→複数日)", get("daily.kurtosis_decay.decay_slope")),
        ("Var(log σ) 断面", get("vol.ensemble.var_log_sigma")),
        ("予算シェア 断面 (分母 0.25)", get("vol.ensemble.shares_of_budget")),
        ("予算使用率 断面", get("vol.ensemble.budget_used_fraction")),
        ("E[σ²]/σ̄² 断面", get("vol.ensemble.e_sigma2_ratio")),
        ("粗さ H (潜在, 5分〜4時間)", get("rough.h_latent.h")),
        ("粗さ ζ_q 線形性 R²", get("rough.h_latent.linearity_r2")),
        ("粗さ H (RV 側, 記録のみ)", get("rough.h_rv.h")),
        ("ボラ増分 ACF(1) (60秒)", get("rough.increment_acf.lag1")),
        ("ラフ予算シェア (経路)", get("rough.share_of_budget_path.value")),
    ]
    width = max(len(label) for label, _ in rows)
    print()
    print("主要指標")
    print("-" * (width + 24))
    for label, value in rows:
        print(f"  {label.ljust(width)}  {_fmt(value)}")


def _print_na(metrics: Mapping[str, Any]) -> None:
    """該当なしの指標を一覧する (「測っていない」ことを可視化する)。"""
    entries: list[str] = []
    for group in ("micro", "cross"):
        for name, node in metrics.get(group, {}).items():
            if isinstance(node, Mapping) and node.get("status") == "not_applicable":
                entries.append(f"  {group}.{name}: {node.get('reason')}")
    if entries:
        print()
        print("該当なし (not_applicable) の指標")
        print("-" * 70)
        for line in entries:
            print(line)


# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = _build_config(args)
    stage = config.stage
    print(f"[1/6] 実行 stage={stage} seed={config.seed} "
          f"n_days={config.n_days} steps_per_day={config.steps_per_day}")

    result = run_pipeline(config)
    print(f"      完了 ({result.runtime_sec:.2f} 秒, グリッド {result.price.n_points:,} 点)")

    print("[2/6] 決定性の確認 (同一シードで再実行)")
    determinism = determinism_check(config, first=result)
    print(f"      ビット単位一致: {determinism['bitwise_identical']}")

    print("[3/6] RNG ストリーム安定性の確認")
    rng_stability = rng_stability_check(config)
    print(f"      既存ストリーム不変: {rng_stability['unchanged']} / "
          f"相互に別系列: {rng_stability['streams_distinct']}")

    rng_diffusion = rng_diffusion_check(config, result)
    print(f"      l2.diffusion 消費列の一致: {rng_diffusion['match']}")

    scale_invariance = None
    if stage != "S0":
        low_steps = config.validation.scale_invariance_steps_per_day
        print(f"[3b/6] 時間スケール不変性 (steps_per_day={low_steps} で対照実行)")
        scale_invariance = scale_invariance_check(config, result)
        print(f"      日次統計の一致: {scale_invariance['passed']}")
        for name, chk in scale_invariance["checks"].items():
            if not chk["passed"]:
                print(f"        不一致: {name}  hi={chk.get('hi')}  lo={chk.get('lo')}")

    print("[4/6] 検証スイート")
    validation_started = time.perf_counter()
    metrics = run_all(result, config)
    errors = collect_errors(metrics)

    baseline_inv = None
    if stage in BASELINE_STAGE:
        base_stage = BASELINE_STAGE[stage]
        print(f"[4b/6] {base_stage} からの不変性照合 (results/{base_stage}/metrics.json)")
        baseline_inv = baseline_invariance_check(
            config, metrics, base_stage, results_root=args.results_dir
        )
        if baseline_inv.get("error"):
            print(f"      基準が読めません: {baseline_inv['error']}")
        else:
            print(f"      不変性: {baseline_inv['passed']}")
            for name, chk in baseline_inv["checks"].items():
                if not chk.get("passed"):
                    print(f"        不一致: {name}  {chk}")

    multiseed = None
    if args.seeds and args.seeds > 1:
        print(f"[4c/6] 多シード判定 ({args.seeds} シード — hill/leverage/JV/skew)")
        multiseed = _run_multiseed(config, args.seeds)
        for name in ("hill_alpha", "leverage_corr", "jv_share", "skewness_daily"):
            info = multiseed.get(name) or {}
            if info.get("median") is not None:
                print(f"      {name}: median={info['median']:+.4f}  IQR={info['iqr']:.4f}")
        metrics["multiseed"] = multiseed

    metrics["runtime"] = {
        "pipeline": {
            "completed": True,
            "runtime_sec": result.runtime_sec,
            "driver": result.meta.get("driver"),
            "layers": result.meta.get("layers"),
            "grid": result.meta.get("grid"),
            "rng_streams_used": result.meta.get("rng_streams_used"),
            "environment": result.meta.get("environment"),
            "result_digest": result.digest(),
        },
        "determinism": determinism,
        "rng_stability": rng_stability,
        "rng_diffusion": rng_diffusion,
        **({"scale_invariance": scale_invariance} if scale_invariance is not None else {}),
        **({"baseline_invariance": baseline_inv} if baseline_inv is not None else {}),
        "validation": {
            "all_callable": not errors,
            "n_errors": len(errors),
            "errors": errors,
            "runtime_sec": time.perf_counter() - validation_started,
        },
        "artifacts": {"metrics_json_ok": False, "reason": "書き出し前"},
        "rng_fingerprint": result.rng_fingerprint,
    }
    print(f"      指標の算出完了 ({metrics['runtime']['validation']['runtime_sec']:.2f} 秒, "
          f"エラー {len(errors)} 件)")

    gates = gates_for(stage)
    gate_results = evaluate(gates, metrics)
    summary = summarize(gate_results)

    print("[5/6] 結果の書き出しと読み直し")
    total_runtime = time.perf_counter() - started
    path = write_metrics(
        stage, config, metrics, gate_results, summary, total_runtime, root=args.results_dir
    )
    verification = verify_metrics_file(path)
    metrics["runtime"]["artifacts"] = verification
    gate_results = evaluate(gates, metrics)
    summary = summarize(gate_results)
    total_runtime = time.perf_counter() - started
    path = write_metrics(
        stage, config, metrics, gate_results, summary, total_runtime, root=args.results_dir
    )
    print(f"      {path} ({verification.get('size_bytes', 0):,} バイト)")

    if not args.no_plots:
        print("[6/6] プロット")
        plots = make_plots(metrics, stage, result=result, root=args.results_dir)
        for plot_path in plots:
            print(f"      {plot_path.name}")
    else:
        print("[6/6] プロットは --no-plots によりスキップ")

    _print_key_metrics(metrics)
    _print_na(metrics)
    _print_gates(gate_results, summary)
    print(f"\n所要時間 {total_runtime:.1f} 秒")
    return 0 if summary["all_critical_passed"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    stage = args.stage
    data = load_metrics(stage, root=args.results_dir)
    metrics = data.get("metrics", {})

    # 保存済みの artifacts 判定を鵜呑みにせず、いま実際にファイルがあるかで上書きする。
    path = Path(args.results_dir or (Path(__file__).resolve().parent.parent / "results"))
    metrics.setdefault("runtime", {})["artifacts"] = verify_metrics_file(
        path / stage / "metrics.json"
    )

    gate_results = evaluate(gates_for(stage), metrics)
    summary = summarize(gate_results)
    print(f"stage={stage}  git_commit={data.get('git_commit')}  "
          f"config_hash={(data.get('config_hash') or '')[:12]}")
    print(f"作成日時 {data.get('created_at')}  実行時間 {data.get('runtime_sec')}")
    _print_key_metrics(metrics)
    _print_gates(gate_results, summary)
    if summary["all_critical_passed"] != data.get("all_critical_passed"):
        print("\n注意: 保存時の判定と再判定の結果が食い違っています "
              f"(保存時 {data.get('all_critical_passed')} / 再判定 {summary['all_critical_passed']})。"
              " ゲート定義が変わった可能性があります。")
    return 0 if summary["all_critical_passed"] else 1


def cmd_compare(args: argparse.Namespace) -> int:
    stages: Sequence[str] = args.stages
    if len(stages) < 2:
        print("compare には 2 つ以上の段階が必要です", file=sys.stderr)
        return 2
    diff = compare_stages(stages, root=args.results_dir, only_changed=args.only_changed)

    print("段階間比較: " + " vs ".join(stages))
    print()
    print("  " + "指標".ljust(52) + "  " + "  ".join(s.rjust(14) for s in stages)
          + ("      差分" if len(stages) == 2 else ""))
    print("-" * (54 + 16 * len(stages) + 12))
    for row in diff["metrics"]:
        cells = "  ".join(_fmt(row["values"][s]).rjust(14) for s in stages)
        delta = f"  {_fmt(row['delta']).rjust(12)}" if len(stages) == 2 else ""
        print(f"  {row['metric'][:52].ljust(52)}  {cells}{delta}")

    print()
    print("  " + "ゲート".ljust(52) + "  " + "  ".join(s.rjust(14) for s in stages))
    print("-" * (54 + 16 * len(stages)))
    for row in diff["gates"]:
        cells = "  ".join(_fmt(row.get(s)).rjust(14) for s in stages)
        print(f"  {row['gate'][:52].ljust(52)}  {cells}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(diff, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"\n{args.json} に書き出しました")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simchart.cli",
        description="段階構築式マイクロ構造シミュレータ (S0 骨格層)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="シミュレーション・検証・ゲート判定・永続化")
    run_parser.add_argument("--config", type=str, default=None, help="設定 YAML / JSON")
    run_parser.add_argument("--stage", type=str, default=None, help="段階名 (既定は設定ファイルの値)")
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--n-days", type=int, default=None)
    run_parser.add_argument("--steps-per-day", type=int, default=None)
    run_parser.add_argument("--results-dir", type=str, default=None, help="results/ の位置")
    run_parser.add_argument("--no-plots", action="store_true")
    run_parser.add_argument(
        "--seeds", type=int, default=None,
        help="ノイズの大きい指標 (hill/leverage/JV/skew) を N シードの中央値で判定する (S3+)",
    )
    run_parser.set_defaults(func=cmd_run)

    validate_parser = sub.add_parser("validate", help="保存済み結果のゲート再判定")
    validate_parser.add_argument("--stage", type=str, required=True)
    validate_parser.add_argument("--results-dir", type=str, default=None)
    validate_parser.set_defaults(func=cmd_validate)

    compare_parser = sub.add_parser("compare", help="段階間の指標差分")
    compare_parser.add_argument("--stages", type=str, nargs="+", required=True)
    compare_parser.add_argument("--results-dir", type=str, default=None)
    compare_parser.add_argument("--only-changed", action="store_true", help="差が 0 の指標を省く")
    compare_parser.add_argument("--json", type=str, default=None, help="比較結果の書き出し先")
    compare_parser.set_defaults(func=cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NotImplementedError as exc:
        print(f"\n未実装のため停止しました:\n  {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
