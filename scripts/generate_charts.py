"""模擬チャートを大量生成する。

    uv run python scripts/generate_charts.py --n-charts 1000

各チャートは**独立した 1 本の価格経路**であり、`Config.seed` を 1 ずつずらして
生成する。RNG は名前ハッシュ方式なので、シードが違えば全ストリームが独立になる
(`sha256(f"{seed}:{stream}")` の出力が別物になるため)。逆に言えば **seed さえ
記録しておけば、どのチャートでも 1 秒刻みの完全な経路をビット単位で再生成できる**
ので、日足 OHLC だけを保存して 1 秒データは捨ててよい。

出力 (`results/<stage>/charts/`)
-------------------------------
``daily_ohlc.parquet``
    全チャートの日足 OHLC。1 行 = 1 チャート 1 日。日中の高値・安値は
    1 秒刻みの経路から取っているので、始値と終値だけから作った偽物ではない。
``intraday_ohlc_sample.parquet``
    先頭数本のチャートの分足 OHLC。全チャート分を書くと数 GB になるので標本のみ。
``charts_index.parquet``
    チャートごとの seed・要約統計・日足終値のダイジェスト。再生成の照合に使う。
``ensemble_metrics.json``
    1000 本まとめての検証。個々のチャートが正しく見えても、**集団として GBM に
    なっていなければ意味がない**ので、こちらを本体と考えること。
``plots/``
    標本経路・ファンチャート・終端分布・ローソク足。

出来高について
--------------
**出来高の列は作らない。** S0 には注文流が無い (L1 は定数強度のスタブ、L3 は
恒等写像) ので、出来高を付けるとしたらそれは捏造になる。S6 で板層、S7 で
Hawkes 注文流を入れた段階で初めて出来高が内生的に決まる。代わりに、日中の
1 秒リターンから計算した**実現ボラティリティ**を列に持たせてある。これは
経路から実際に測れる量である。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from simchart import Config, run
from simchart.config import IMPLEMENTED_FLAGS
from simchart.report import git_info, results_dir
from simchart.validation.base import jsonable, num

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
def ohlc_from_log_price(
    log_price: np.ndarray, n_bars: int, steps_per_bar: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """等間隔格子の対数価格から OHLC を作る。

    ``log_price`` は ``n_bars * steps_per_bar + 1`` 点。バー b は
    ``[b*steps_per_bar, (b+1)*steps_per_bar]`` の**両端を含む**区間に対応する。
    終値は次のバーの始値と同一点であり、S0 にはオーバーナイトが無いのでこれで正しい。
    S4 でギャップを入れたら、ここは日ごとに別の点になる。
    """
    expected = n_bars * steps_per_bar + 1
    if log_price.shape[0] != expected:
        raise ValueError(f"点数が合いません: {log_price.shape[0]} != {expected}")
    panel = log_price[:-1].reshape(n_bars, steps_per_bar)  # view
    open_log = panel[:, 0]
    close_log = log_price[steps_per_bar::steps_per_bar]
    high_log = np.maximum(panel.max(axis=1), close_log)
    low_log = np.minimum(panel.min(axis=1), close_log)
    return open_log, high_log, low_log, close_log


def _pooled_abs_acf1(returns_2d: np.ndarray) -> float:
    """|日次リターン| のラグ1 自己相関を、チャート内で測ってプールする。

    チャートをまたぐラグを作らないこと (別の経路をつなぐと偽の相関が入る)。
    """
    x = np.abs(returns_2d)
    x = x - x.mean()
    num_ = float((x[:, :-1] * x[:, 1:]).sum())
    den = float((x**2).sum())
    return num_ / den if den > 0 else float("nan")


def max_drawdown(close: np.ndarray) -> float:
    """終値ベースの最大ドローダウン (正の比率)。"""
    peak = np.maximum.accumulate(close)
    return float(np.max(1.0 - close / peak))


# ---------------------------------------------------------------------------
def build_base_config(args: argparse.Namespace) -> Config:
    """生成に使う基準 Config を組み立てる。

    **段階の設定ファイルを読むこと。** ``Config(stage="S1")`` を素で組むと
    ``enable_msm`` / ``enable_slow_ou`` が False のままになり、**S0 と同一の経路を
    S1 と称して出力する**という最悪の事故になる (フラグの暗黙 no-op と同じ構造)。
    そのため段階が S0 でないのに実装済みフラグが 1 つも立っていない場合は停止する。
    """
    base = Config.load(args.config) if args.config else Config()
    overrides: dict[str, Any] = {}
    if args.stage is not None:
        overrides["stage"] = args.stage
    if args.base_seed is not None:
        overrides["seed"] = args.base_seed
    if args.n_days is not None:
        overrides["n_days"] = args.n_days
    if args.steps_per_day is not None:
        overrides["steps_per_day"] = args.steps_per_day
    if overrides:
        base = base.replace(**overrides)

    if base.stage != "S0" and not any(getattr(base, f) for f in IMPLEMENTED_FLAGS):
        raise ValueError(
            f"stage={base.stage} なのに実装済みフラグ ({', '.join(IMPLEMENTED_FLAGS)}) が"
            f" 1 つも有効になっていません。S0 と同一の経路を {base.stage} と称して"
            f" 出力してしまうため停止します。--config configs/{base.stage.lower()}.yaml"
            f" を指定してください。"
        )
    return base


def generate(args: argparse.Namespace) -> int:
    if args.recompute:
        # 再計算は保存済みの設定を読むので、段階ガード (フラグの整合検査) は不要。
        # --stage S1 だけ渡しても通るよう、ガードより前に分岐する。
        _cfg = Config.load(args.config) if args.config else None
        stage = args.stage or (_cfg.stage if _cfg else "S0")
        if _cfg is not None and _cfg.market_type != "equity":
            stage = f"perp_{stage}"
        out_dir = results_dir(stage, args.results_dir) / "charts"
        return recompute(out_dir, args)

    base = build_base_config(args)
    # ★時間軸は config の単一情報源から取る (S0-perp §4)。定数直書きだと
    # perp (365 日 / 86,400 秒) で年率換算が √(365/252) = 1.20 倍ずれる。
    # 結果ラベルも market_type で分ける — perp を stage 名のまま書くと
    # 株式のベースライン (results/S0/charts) を上書きしてしまう。
    stage = base.stage if base.market_type == "equity" else f"perp_{base.stage}"
    n_days, steps_per_day = base.n_days, base.steps_per_day
    session_seconds = base.seconds_per_day
    step_seconds = session_seconds / steps_per_day
    seconds_per_year = base.ann_days * session_seconds

    if args.intraday_bar_sec % step_seconds != 0:
        raise ValueError("intraday_bar_sec が刻みの整数倍ではありません")
    steps_per_intraday_bar = int(args.intraday_bar_sec / step_seconds)
    intraday_bars_per_day = steps_per_day // steps_per_intraday_bar

    out_dir = results_dir(stage, args.results_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_charts
    chunk = max(int(args.chunk_size), 1)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    base_seed = base.seed
    flags_on = [f for f in IMPLEMENTED_FLAGS if getattr(base, f)]
    n_chunks = (n + chunk - 1) // chunk
    print(f"{n} 本 x {n_days} 日 x {steps_per_day} ステップ を生成します "
          f"(stage={stage}, seed={base_seed}..{base_seed + n - 1}, "
          f"有効フラグ={', '.join(flags_on) if flags_on else 'なし (S0)'}, "
          f"{chunk} 本ごとに {n_chunks} チャンクへ逐次書き出し)")

    # ★チャンク単位で逐次書き出す。3 時間の生成が中断されても完了分は残り、
    # 同じコマンドで再開すれば未完了のチャンクだけを作る (実際に 2 度中断された)。
    n_done_before = 0
    n_made = 0
    stopped_early = False
    for c_idx in range(n_chunks):
        lo = c_idx * chunk
        hi = min(lo + chunk, n)
        if _chunk_is_complete(parts_dir, c_idx, lo, hi):
            n_done_before += hi - lo
            continue
        # 予算判定は**チャンクを始める前**に予測で行う。終わってから判定すると
        # 最後の 1 チャンク分 (本番で ~260 秒) だけ必ず超過し、呼び出し側の
        # タイムアウトを踏む。平均チャンク時間が分かるまで (最初の 1 本) は実行する。
        if args.time_budget_sec and n_made:
            elapsed = time.perf_counter() - started
            avg = elapsed / n_made * (hi - lo)
            if elapsed + avg > args.time_budget_sec:
                stopped_early = True
                print(f"  次のチャンクは予算 {args.time_budget_sec:.0f} 秒に収まらない見込み"
                      f" (経過 {elapsed:.0f} + 予測 {avg:.0f})。ここで一旦終了します",
                      flush=True)
                break
        _generate_chunk(
            base, lo, hi, n_days, steps_per_day, intraday_bars_per_day,
            steps_per_intraday_bar, args, parts_dir, c_idx,
        )
        n_made += hi - lo
        elapsed = time.perf_counter() - started
        eta = (elapsed / n_made * (n - hi)) if n_made > 0 else 0.0
        print(f"  チャンク {c_idx + 1}/{n_chunks} (〜{hi}/{n} 本)  "
              f"経過 {elapsed:.0f} 秒 / 残り約 {eta:.0f} 秒", flush=True)

    if n_done_before:
        print(f"  (既存のチャンクから {n_done_before} 本を再利用)")

    missing = [
        k for k in range(n_chunks)
        if not _chunk_is_complete(parts_dir, k, k * chunk, min((k + 1) * chunk, n))
    ]
    if missing:
        done = n - sum(
            min((k + 1) * chunk, n) - k * chunk for k in missing
        )
        print(f"\n未完了: {len(missing)} チャンク (完了 {done}/{n} 本)。"
              f" 同じコマンドを再実行すると続きから作ります。")
        return 0 if stopped_early or n_made else 1

    # ------------------------------------------------------------------
    # 全チャンクを結合して最終成果物にする。
    daily = pd.concat(
        [pd.read_parquet(parts_dir / f"daily_{k:04d}.parquet") for k in range(n_chunks)],
        ignore_index=True,
    )
    index_df = pd.concat(
        [pd.read_parquet(parts_dir / f"index_{k:04d}.parquet") for k in range(n_chunks)],
        ignore_index=True,
    )
    intraday_parts = sorted(parts_dir.glob("intraday_*.parquet"))
    intraday_frames = [pd.read_parquet(p) for p in intraday_parts]
    return _finalize(
        base, stage, args, out_dir, daily, index_df, intraday_frames,
        n, n_days, steps_per_day, step_seconds, seconds_per_year,
        base_seed, flags_on, started,
    )


def _chunk_is_complete(parts_dir: Path, c_idx: int, lo: int, hi: int) -> bool:
    """チャンク ``c_idx`` が期待どおりの範囲 ``[lo, hi)`` で完了しているか。

    ★ファイルの存在だけで判定してはならない。``--n-charts`` を変えて再実行すると
    最終チャンクの範囲が変わる (例: n=45 の chunk1 は [25,45)、n=1000 なら [25,50))
    ので、存在だけを見ると**短いチャンクを完了扱いにしてチャートを取りこぼす**。
    index に記録された chart_id が期待範囲と一致することまで確認する。
    """
    daily_part = parts_dir / f"daily_{c_idx:04d}.parquet"
    index_part = parts_dir / f"index_{c_idx:04d}.parquet"
    if not (daily_part.exists() and index_part.exists()):
        return False
    try:
        ids = pd.read_parquet(index_part, columns=["chart_id"])["chart_id"].tolist()
    except Exception:  # noqa: BLE001 - 壊れた part は作り直す
        return False
    return sorted(int(v) for v in ids) == list(range(lo, hi))


def _generate_chunk(
    base: Config,
    lo: int,
    hi: int,
    n_days: int,
    steps_per_day: int,
    intraday_bars_per_day: int,
    steps_per_intraday_bar: int,
    args: argparse.Namespace,
    parts_dir: Path,
    c_idx: int,
) -> None:
    """チャート ``lo..hi-1`` を生成し、チャンクの parquet を書き出す。"""
    base_seed = base.seed
    n_local = hi - lo
    total_days = n_local * n_days
    chart_id = np.repeat(np.arange(lo, hi, dtype=np.int32), n_days)
    day_index = np.tile(np.arange(n_days, dtype=np.int32), n_local)
    open_px = np.empty(total_days, dtype=np.float64)
    high_px = np.empty(total_days, dtype=np.float64)
    low_px = np.empty(total_days, dtype=np.float64)
    close_px = np.empty(total_days, dtype=np.float64)
    log_return = np.empty(total_days, dtype=np.float64)
    realized_vol = np.empty(total_days, dtype=np.float64)
    index_rows: list[dict[str, Any]] = []
    intraday_frames: list[pd.DataFrame] = []

    for i in range(lo, hi):
        seed = base_seed + i
        result = run(base.replace(seed=seed))
        log_price = result.observation.log_price

        o, h, l, c = ohlc_from_log_price(log_price, n_days, steps_per_day)
        # 実現分散。5000 日 x 23400 では 1 本あたり 936MB なので、二乗は in-place で
        # 行って一時配列を増やさない (値は (x**2) と同一)。
        step_returns = np.diff(log_price)
        np.square(step_returns, out=step_returns)
        rv_daily = step_returns.reshape(n_days, steps_per_day).sum(axis=1)
        del step_returns

        sl = slice((i - lo) * n_days, (i - lo + 1) * n_days)
        open_px[sl] = np.exp(o)
        high_px[sl] = np.exp(h)
        low_px[sl] = np.exp(l)
        close_px[sl] = np.exp(c)
        log_return[sl] = c - o
        realized_vol[sl] = np.sqrt(rv_daily * base.ann_days)

        daily_ret = c - o
        index_rows.append(
            {
                "chart_id": i,
                "seed": seed,
                "first_open": float(np.exp(o[0])),
                "last_close": float(np.exp(c[-1])),
                "total_log_return": float(c[-1] - o[0]),
                "realized_vol_annualized": float(
                    np.sqrt(rv_daily.sum() / n_days * base.ann_days)
                ),
                "daily_return_mean": float(daily_ret.mean()),
                "daily_return_std": float(daily_ret.std(ddof=1)),
                "daily_return_kurtosis": float(stats.kurtosis(daily_ret, fisher=False, bias=False)),
                "max_drawdown": max_drawdown(np.exp(c)),
                "close_digest": hashlib.sha256(
                    np.ascontiguousarray(c, dtype=np.float64).tobytes()
                ).hexdigest()[:16],
            }
        )

        if i < args.intraday_samples:
            bo, bh, bl, bc = ohlc_from_log_price(
                log_price, n_days * intraday_bars_per_day, steps_per_intraday_bar
            )
            n_bars = bo.shape[0]
            intraday_frames.append(
                pd.DataFrame(
                    {
                        "chart_id": np.full(n_bars, i, dtype=np.int32),
                        "day": np.repeat(np.arange(n_days, dtype=np.int32), intraday_bars_per_day),
                        "bar": np.tile(
                            np.arange(intraday_bars_per_day, dtype=np.int32), n_days
                        ),
                        "open": np.exp(bo),
                        "high": np.exp(bh),
                        "low": np.exp(bl),
                        "close": np.exp(bc),
                    }
                )
            )
            del bo, bh, bl, bc

        # ★ここで明示的に解放する。del を書かないと、次の反復の run() が走っている
        # あいだ前のチャートの経路 (t / log_p / log_vol = 本番設定で 2.8GB) が
        # まだ束縛されたままになり、常に 2 本分が同時に生きてピークが 2.8GB 増える。
        # 実測でピーク 9.4GB・空き 1.8GB まで落ちた (15.3GB 機で危険水準)。
        del result, log_price, o, h, l, c, rv_daily, daily_ret

        if (i + 1) % args.progress_every == 0:
            print(f"    {i + 1}/{args.n_charts} 本目まで完了", flush=True)

    # チャンクの成果物を書き出す。**先に daily を書き、最後に index を書く** —
    # index の存在が「このチャンクは完了」の印なので、途中で落ちても不完全な
    # チャンクが「完了済み」と誤認されない (再開時に作り直される)。
    pd.DataFrame(
        {
            "chart_id": chart_id,
            "day": day_index,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "log_return": log_return,
            "realized_vol_annualized": realized_vol,
        }
    ).to_parquet(parts_dir / f"daily_{c_idx:04d}.parquet", index=False, compression="zstd")
    if intraday_frames:
        pd.concat(intraday_frames, ignore_index=True).to_parquet(
            parts_dir / f"intraday_{c_idx:04d}.parquet", index=False, compression="zstd"
        )
    pd.DataFrame(index_rows).to_parquet(
        parts_dir / f"index_{c_idx:04d}.parquet", index=False, compression="zstd"
    )


def _finalize(
    base: Config,
    stage: str,
    args: argparse.Namespace,
    out_dir: Path,
    daily: pd.DataFrame,
    index_df: pd.DataFrame,
    intraday_frames: list[pd.DataFrame],
    n: int,
    n_days: int,
    steps_per_day: int,
    step_seconds: float,
    seconds_per_year: float,
    base_seed: int,
    flags_on: list[str],
    started: float,
) -> int:
    """結合済みのデータから最終成果物 (parquet / metrics / plots) を作る。"""
    daily_path = out_dir / "daily_ohlc.parquet"
    daily.to_parquet(daily_path, index=False, compression="zstd")

    index_path = out_dir / "charts_index.parquet"
    index_df.to_parquet(index_path, index=False, compression="zstd")
    index_df.to_csv(out_dir / "charts_index.csv", index=False)

    intraday_path = None
    if intraday_frames:
        intraday = pd.concat(intraday_frames, ignore_index=True)
        intraday_path = out_dir / "intraday_ohlc_sample.parquet"
        intraday.to_parquet(intraday_path, index=False, compression="zstd")

    # ------------------------------------------------------------------
    metrics = ensemble_metrics(
        daily, index_df, base, n, n_days, seconds_per_year, time.perf_counter() - started
    )
    info = git_info()
    payload = {
        "stage": stage,
        "git_commit": info["commit"],
        "git_dirty": info["dirty"],
        "config": base.to_dict(),
        "config_hash": base.config_hash(),
        "generation": {
            "n_charts": n,
            "n_days": n_days,
            "steps_per_day": steps_per_day,
            "step_seconds": step_seconds,
            "base_seed": base_seed,
            "seeds": [base_seed, base_seed + n - 1],
            "enabled_flags": flags_on,
            "intraday_samples": args.intraday_samples,
            "intraday_bar_sec": args.intraday_bar_sec,
            "runtime_sec": time.perf_counter() - started,
        },
        "files": {
            "daily_ohlc": str(daily_path.name),
            "charts_index": str(index_path.name),
            "intraday_ohlc_sample": None if intraday_path is None else str(intraday_path.name),
        },
        "ensemble": jsonable(metrics),
    }
    with open(out_dir / "ensemble_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)

    if not args.no_plots:
        made = make_plots(daily, index_df, base, out_dir / "plots")
        print(f"プロット {len(made)} 枚: {', '.join(p.name for p in made)}")

    report(metrics, daily_path, index_path, intraday_path, out_dir, stage)
    return 0


# ---------------------------------------------------------------------------
def recompute(out_dir: Path, args: argparse.Namespace) -> int:
    """既存の出力から集団検証とプロットだけを作り直す (経路は再生成しない)。

    集計や検定の書き方を直したときに、8 分かけて 1000 本を作り直さずに済ませる
    ための経路。日足 parquet と index は入力として読むだけで書き換えない。
    """
    metrics_path = out_dir / "ensemble_metrics.json"
    with open(metrics_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    base = Config.from_dict(payload["config"])
    daily = pd.read_parquet(out_dir / "daily_ohlc.parquet")
    index_df = pd.read_parquet(out_dir / "charts_index.parquet")
    n_charts = int(index_df.shape[0])
    n_days = int(daily["day"].max()) + 1
    print(f"既存の {n_charts} 本 x {n_days} 日 から集団検証を作り直します")

    seconds_per_year = base.ann_days * base.seconds_per_day
    metrics = ensemble_metrics(
        daily, index_df, base, n_charts, n_days, seconds_per_year,
        float(payload["generation"].get("runtime_sec", 0.0)),
    )
    payload["ensemble"] = jsonable(metrics)
    payload["git_commit"] = git_info()["commit"]
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)

    if not args.no_plots:
        made = make_plots(daily, index_df, base, out_dir / "plots")
        print(f"プロット {len(made)} 枚: {', '.join(p.name for p in made)}")

    intraday = out_dir / "intraday_ohlc_sample.parquet"
    report(metrics, out_dir / "daily_ohlc.parquet", out_dir / "charts_index.parquet",
           intraday if intraday.exists() else None, out_dir, base.stage)
    return 0


# ---------------------------------------------------------------------------
def ensemble_metrics(
    daily: pd.DataFrame,
    index_df: pd.DataFrame,
    config: Config,
    n_charts: int,
    n_days: int,
    seconds_per_year: float,
    runtime_sec: float,
) -> dict[str, Any]:
    """まとめての検証。

    個々のチャートが「それらしく」見えても、集団として意図した過程になっていな
    ければ意味がない。**検査は段階に依らない不変量と、段階ごとに変わる記述統計に
    分けてある。** 混ぜると S1 で「尖度が 3 でないから不合格」のような誤った判定に
    なる (S1 は尖度が 3 でないことこそが目的)。

    段階に依らない不変量 (``universal``)
        1. ``E[Σr²]/T = sigma_bar²`` — 実現**分散**の平均。S1 でボラが変動しても
           E[σ²] = σ̄² の正規化は保たれる (凸性補正の経路レベルでの検証)
        2. ``E[exp(終端対数リターン)] = 1`` — マルチンゲール性 (伊藤補正 −σ²/2)
        3. ``Var(終端対数リターン) = sigma_bar² T`` — 積分分散の期待値
        4. チャート同士が独立 (シードをずらしただけで相関していないか)

    段階で変わる記述統計 (``descriptive``)
        尖度・正規性検定・実現ボラの散らばり・値幅/実体比。S0 では理論値 (尖度 3、
        正規、値幅 σ√(8/π)) が付くが、S1 ではボラ混合により**理論値そのものが
        変わる**ので、理論との比較は S0 のときだけ載せる。
    """
    returns = daily["log_return"].to_numpy().reshape(n_charts, n_days)
    pooled = returns.ravel()
    terminal = returns.sum(axis=1)
    horizon_years = n_days / config.ann_days
    sigma = config.sigma_bar

    # --- 1. 実現ボラ ---
    vols = index_df["realized_vol_annualized"].to_numpy()

    # --- 2. プールした日次リターン ---
    centered = returns - returns.mean()
    acf1 = float(
        (centered[:, :-1] * centered[:, 1:]).sum() / (centered**2).sum()
    )
    n_pooled = pooled.size

    # --- 3. マルチンゲール性 ---
    gross = np.exp(terminal)

    # --- 4. チャート間の独立性 ---
    corr = np.corrcoef(returns)
    iu = np.triu_indices(n_charts, k=1)
    off_diag = corr[iu]
    corr_se = 1.0 / math.sqrt(n_days - 1)
    n_pairs = off_diag.size
    z_pairs = off_diag / corr_se
    max_abs_z = float(np.max(np.abs(z_pairs)))
    # 独立なら |z| の最大値はおよそ Phi^-1(1 - 1/(2*ペア数))。ただし最大値の分布は
    # 右に裾を引くので、目安の値を超えたこと自体は異常を意味しない。**超過確率**で
    # 判断すること: P(max|z| > 観測値) = 1 - (1 - 2*Phi(-観測値))^ペア数。
    expected_max_z = float(stats.norm.ppf(1.0 - 1.0 / (2.0 * n_pairs)))
    tail_prob = 2.0 * stats.norm.sf(max_abs_z)
    max_abs_z_pvalue = float(-np.expm1(n_pairs * np.log1p(-tail_prob)))

    # --- 0. 段階に依らない不変量 ---
    is_s0 = not any(getattr(config, f) for f in IMPLEMENTED_FLAGS)
    horizon_years_u = n_days / config.ann_days
    # 実現分散 (年率) = 各チャートの Σr²/T。実現ボラの二乗ではなく分散で平均する
    # ことが重要 — E[σ] != sqrt(E[σ²]) なので、ボラが変動する段階では実現ボラの
    # 平均は σ̄ より小さくなる (Jensen)。正規化条件は分散の側にある。
    realized_var = (index_df["realized_vol_annualized"].to_numpy()) ** 2
    rv_mean = float(realized_var.mean())
    rv_se = float(realized_var.std(ddof=1) / math.sqrt(n_charts))
    gross_u = np.exp(returns.sum(axis=1))
    terminal_u = returns.sum(axis=1)
    universal = {
        "e_realized_var": {
            "value": num(rv_mean),
            "expected": num(sigma**2),
            "se": num(rv_se),
            "z": num((rv_mean - sigma**2) / rv_se) if rv_se > 0 else None,
            "note": "E[Σr²]/T = sigma_bar^2。ボラが変動しても保たれる正規化条件",
        },
        "martingale": {
            "value": num(float(gross_u.mean())),
            "expected": 1.0,
            "se": num(float(gross_u.std(ddof=1) / math.sqrt(n_charts))),
            "z": num(
                (float(gross_u.mean()) - 1.0)
                / float(gross_u.std(ddof=1) / math.sqrt(n_charts))
            ),
            "note": "E[P_T/P_0] = 1。伊藤補正 -sigma^2/2 が入っているかの検査",
        },
        "terminal_variance": {
            "value": num(float(terminal_u.var(ddof=1))),
            "expected": num(sigma**2 * horizon_years_u),
            "note": "Var(終端対数リターン) = sigma_bar^2 T (積分分散の期待値)",
        },
    }

    # --- 5. OHLC の高値・安値が本当に日中経路から来ているか ---
    # ブラウン運動の 1 期間の値幅は E[max - min] = sigma * sqrt(8/pi)、
    # 始値終値の差は E|close - open| = sigma * sqrt(2/pi)。高値・安値は終値だけからは
    # 復元できない量なので、ここが合うことが「日中を本当に見ている」証拠になる。
    # 1 秒刻みの離散標本なので値幅はわずかに過小になる (刻み間の極値を見逃すため)。
    # Broadie-Glasserman-Kou の補正で相対 -2*0.5826*sqrt(dt/T) 程度。
    sigma_day = sigma / math.sqrt(config.ann_days)
    steps_per_day = config.steps_per_day
    log_range = np.log(daily["high"].to_numpy() / daily["low"].to_numpy())
    abs_body = np.abs(daily["log_return"].to_numpy())
    range_theory = sigma_day * math.sqrt(8.0 / math.pi)
    discrete_factor = 1.0 - 2.0 * 0.5826 / math.sqrt(steps_per_day) / math.sqrt(8.0 / math.pi)

    return {
        "stage": config.stage,
        "is_constant_vol": bool(is_s0),
        "universal": universal,
        "ohlc": {
            "mean_log_range": num(log_range.mean()),
            "mean_abs_body": num(abs_body.mean()),
            "range_to_body_ratio": num(log_range.mean() / abs_body.mean()),
            "steps_per_day": int(steps_per_day),
            # 理論値は定数ボラのときだけ意味を持つ。ボラが変動すると
            # E[値幅] = E[sigma_day] sqrt(8/pi) となり、E[sigma] < sqrt(E[sigma^2])
            # (Jensen) の分だけ σ̄ ベースの理論値より小さくなる。
            **(
                {
                    "expected_log_range_continuous": num(range_theory),
                    "expected_log_range_discrete": num(range_theory * discrete_factor),
                    "range_ratio_vs_discrete": num(
                        log_range.mean() / (range_theory * discrete_factor)
                    ),
                    "expected_abs_body": num(sigma_day * math.sqrt(2.0 / math.pi)),
                    "body_ratio": num(
                        abs_body.mean() / (sigma_day * math.sqrt(2.0 / math.pi))
                    ),
                }
                if is_s0
                else {
                    "theory_not_applicable": (
                        "ボラが変動するため sigma_bar ベースのブラウン運動理論値は"
                        "当てはまらない (E[sigma] < sqrt(E[sigma^2]))"
                    )
                }
            ),
        },
        "realized_vol": {
            # S1 以降はチャートごとに大きく散らばるのが正しい (それがボラ変動の
            # 実体)。「sigma_bar からの最大乖離」は定数ボラ段階でしか意味を持たない。
            "mean": num(vols.mean()),
            "std": num(vols.std(ddof=1)),
            "min": num(vols.min()),
            "max": num(vols.max()),
            "p05": num(float(np.percentile(vols, 5))),
            "p50": num(float(np.percentile(vols, 50))),
            "p95": num(float(np.percentile(vols, 95))),
            "sd_log": num(float(np.log(vols).std(ddof=1))),
            "target_if_constant_vol": num(sigma),
            **(
                {"max_abs_dev_from_target": num(np.max(np.abs(vols - sigma)))}
                if is_s0
                else {}
            ),
        },
        "pooled_daily_returns": {
            "n": int(n_pooled),
            "mean": num(pooled.mean()),
            "expected_mean": num(-0.5 * sigma**2 / config.ann_days),
            "std": num(pooled.std(ddof=1)),
            "expected_std": num(sigma / math.sqrt(config.ann_days)),
            "skewness": num(stats.skew(pooled, bias=False)),
            # 尖度 3 は定数ボラ段階の期待値。S1 以降は 3 より大きいのが正しい。
            "kurtosis": num(stats.kurtosis(pooled, fisher=False, bias=False)),
            "kurtosis_se": num(math.sqrt(24.0 / n_pooled)),
            "kurtosis_expected_if_constant_vol": 3.0,
            # |r| の自己相関はチャート内で測ってプールする (チャートをまたぐ
            # ラグを作らない)。S0 では 0、S1 以降は正になる。
            "abs_acf_lag1": num(_pooled_abs_acf1(returns)),
            "acf_lag1": num(acf1),
            "acf_lag1_z": num(acf1 * math.sqrt(n_pooled)),
        },
        "terminal": {
            "horizon_years": num(horizon_years),
            "log_return_mean": num(terminal.mean()),
            "log_return_mean_expected": num(-0.5 * sigma**2 * horizon_years),
            "log_return_mean_se": num(terminal.std(ddof=1) / math.sqrt(n_charts)),
            "log_return_std": num(terminal.std(ddof=1)),
            "log_return_std_expected": num(sigma * math.sqrt(horizon_years)),
            "gross_return_mean": num(gross.mean()),
            "gross_return_mean_expected": 1.0,
            "gross_return_mean_se": num(gross.std(ddof=1) / math.sqrt(n_charts)),
            # 正規性は定数ボラ段階でのみ期待される。S1 以降の終端は分散混合なので
            # 棄却されるのが正しい (棄却されなければボラが動いていない疑い)。
            "normality_pvalue": num(stats.jarque_bera(terminal).pvalue),
            "normality_expected": "正規" if is_s0 else "非正規 (分散混合)",
            "kurtosis": num(stats.kurtosis(terminal, fisher=False, bias=False)),
        },
        "independence": {
            "n_pairs": int(n_pairs),
            "corr_se_under_independence": num(corr_se),
            "max_abs_corr": num(np.max(np.abs(off_diag))),
            "max_abs_z": num(max_abs_z),
            "expected_max_abs_z": num(expected_max_z),
            "max_abs_z_pvalue": num(max_abs_z_pvalue),
            # 最大値だけ見ると「たまたま大きかった 1 組」に判断を引きずられる。
            # 分布そのものが N(0,1) になっているかを併せて見る。
            "mean_corr": num(off_diag.mean()),
            "z_mean": num(z_pairs.mean()),
            "z_std": num(z_pairs.std(ddof=1)),
            "frac_abs_z_over_1_96": num(np.mean(np.abs(z_pairs) > 1.96)),
            "expected_frac_abs_z_over_1_96": 0.05,
        },
        "drawdown": {
            "median": num(index_df["max_drawdown"].median()),
            "p05": num(index_df["max_drawdown"].quantile(0.05)),
            "p95": num(index_df["max_drawdown"].quantile(0.95)),
            "max": num(index_df["max_drawdown"].max()),
        },
        "runtime_sec": num(runtime_sec),
    }


# ---------------------------------------------------------------------------
def make_plots(
    daily: pd.DataFrame, index_df: pd.DataFrame, config: Config, out_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    n_charts = int(index_df.shape[0])
    n_days = config.n_days
    close = daily["close"].to_numpy().reshape(n_charts, n_days)
    p0 = config.p0

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    # 1. 標本経路
    n_panels = min(12, n_charts)
    n_cols = min(4, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k >= n_panels:
            ax.set_visible(False)
            continue
        ax.plot(close[k], lw=0.8)
        ax.axhline(p0, color="r", ls=":", lw=0.7)
        ax.set_title(f"chart {k}  (seed {int(index_df['seed'].iloc[k])})", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"S0 sample paths ({n_days} days, GBM sigma={config.sigma_bar})")
    save(fig, "sample_paths.png")

    # 2. ファンチャート (理論分位点と重ねる)
    days = np.arange(1, n_days + 1)
    horizon = days / config.ann_days
    sigma, mu = config.sigma_bar, config.mu_drift
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for lo, hi, alpha in ((5, 95, 0.18), (25, 75, 0.28)):
        ax.fill_between(
            days, np.percentile(close, lo, axis=0), np.percentile(close, hi, axis=0),
            alpha=alpha, color="C0", lw=0,
            label=f"empirical {lo}-{hi}%",
        )
    ax.plot(days, np.median(close, axis=0), color="C0", lw=1.2, label="empirical median")
    is_s0 = not any(getattr(config, f) for f in IMPLEMENTED_FLAGS)
    # 定数ボラの GBM 分位点。S1 以降でも E[∫σ²] = σ̄²T なので**総分散は同じ**だが、
    # 分布は分散混合になりテールが厚くなる。したがってこの線は「理論」ではなく
    # 「同じ総分散をもつ定数ボラの参照線」であり、経験帯が外側にはみ出すのが正しい。
    ref_label = "theoretical GBM" if is_s0 else "constant-vol reference (same total variance)"
    for q, style in ((0.05, "--"), (0.5, "-"), (0.95, "--")):
        z = stats.norm.ppf(q)
        theo = p0 * np.exp((mu - 0.5 * sigma**2) * horizon + sigma * np.sqrt(horizon) * z)
        ax.plot(days, theo, color="r", ls=style, lw=1.0,
                label=ref_label if q == 0.5 else None)
    ax.set_xlabel("day")
    ax.set_ylabel("price")
    ax.set_title(
        f"{config.stage} fan chart: {n_charts} charts vs "
        + ("theoretical GBM quantiles" if is_s0 else "constant-vol reference")
    )
    ax.legend(fontsize=8)
    save(fig, "fan_chart.png")

    # 3. 終端分布
    terminal = np.log(close[:, -1] / p0)
    horizon_total = n_days / config.ann_days
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(terminal, bins=50, density=True, alpha=0.65)
    grid = np.linspace(terminal.min(), terminal.max(), 400)
    axes[0].plot(
        grid,
        stats.norm.pdf(grid, (mu - 0.5 * sigma**2) * horizon_total,
                       sigma * math.sqrt(horizon_total)),
        "r-", lw=1.3, label="normal (same variance)" if not is_s0 else "theoretical N",
    )
    axes[0].set_xlabel("terminal log return")
    axes[0].set_title(
        "terminal log return vs theory" if is_s0
        else "terminal log return vs normal (variance mixture -> fat tails)"
    )
    axes[0].legend(fontsize=8)
    stats.probplot(terminal, dist="norm", plot=axes[1])
    axes[1].set_title("normal QQ of terminal log return")
    save(fig, "terminal_distribution.png")

    # 5. チャートごとの実現ボラの分布
    # 定数ボラ段階では σ̄ に一点集中し、確率ボラ段階では大きく散らばる。
    # データセット水準で S0 と S1 の違いが一目で分かる図。
    vols = index_df["realized_vol_annualized"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(vols, bins=min(60, max(10, n_charts // 15)), alpha=0.75)
    ax.axvline(sigma, color="r", ls="--", lw=1.2,
               label=f"sigma_bar = {sigma:.2f}")
    ax.axvline(float(np.median(vols)), color="k", ls=":", lw=1.2,
               label=f"median = {np.median(vols):.4f}")
    ax.set_xlabel("realized volatility per chart (annualized)")
    ax.set_ylabel("charts")
    ax.set_title(
        f"{config.stage}: dispersion of realized volatility across {n_charts} charts"
        + ("" if is_s0 else "  (wide spread is the point of S1)")
    )
    ax.legend(fontsize=8)
    save(fig, "realized_vol_distribution.png")

    # 4. ローソク足 (1 本目の直近 120 日)
    tail = daily[daily["chart_id"] == 0].iloc[-120:]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(tail))
    up = tail["close"].to_numpy() >= tail["open"].to_numpy()
    ax.vlines(x, tail["low"], tail["high"], color="0.35", lw=0.7)
    ax.bar(x[up], (tail["close"] - tail["open"]).to_numpy()[up],
           bottom=tail["open"].to_numpy()[up], width=0.65, color="#2a9d5c")
    ax.bar(x[~up], (tail["open"] - tail["close"]).to_numpy()[~up],
           bottom=tail["close"].to_numpy()[~up], width=0.65, color="#c4453c")
    ax.set_xlabel("day (last 120)")
    ax.set_ylabel("price")
    ax.set_title("S0 chart 0: daily candlesticks (high/low from the 1-second path)")
    save(fig, "candlestick_chart0.png")

    return written


# ---------------------------------------------------------------------------
def report(
    metrics: dict[str, Any], daily_path: Path, index_path: Path,
    intraday_path: Path | None, out_dir: Path, stage: str = "S0",
) -> None:
    def size(path: Path | None) -> str:
        return "-" if path is None else f"{path.stat().st_size / 1e6:.1f} MB"

    print()
    print("出力")
    print("-" * 72)
    print(f"  {daily_path.name:<32} {size(daily_path)}")
    print(f"  {index_path.name:<32} {size(index_path)}")
    if intraday_path:
        print(f"  {intraday_path.name:<32} {size(intraday_path)}")
    print(f"  ensemble_metrics.json            {size(out_dir / 'ensemble_metrics.json')}")

    rv = metrics["realized_vol"]
    ohlc = metrics["ohlc"]
    pooled = metrics["pooled_daily_returns"]
    term = metrics["terminal"]
    indep = metrics["independence"]
    dd = metrics["drawdown"]
    uni = metrics["universal"]
    is_s0 = bool(metrics.get("is_constant_vol", True))

    universal_rows = [
        ("E[Σr²]/T (実現分散の平均)", f"{uni['e_realized_var']['value']:.6f}",
         f"理論 {uni['e_realized_var']['expected']:.6f} (z = {uni['e_realized_var']['z']:+.2f})"),
        ("E[P_T/P_0] (マルチンゲール)", f"{uni['martingale']['value']:.5f}",
         f"理論 1.00000 (z = {uni['martingale']['z']:+.2f})"),
        ("Var(終端対数リターン)", f"{uni['terminal_variance']['value']:.5f}",
         f"理論 {uni['terminal_variance']['expected']:.5f}"),
        ("チャート間 相関 z の標準偏差", f"{indep['z_std']:.4f}",
         f"独立なら 1 / |z|>1.96 が {indep['frac_abs_z_over_1_96'] * 100:.2f}% (理論 5%)"),
        ("チャート間 相関 最大|z|", f"{indep['max_abs_z']:.2f}",
         f"目安 {indep['expected_max_abs_z']:.2f} / 超過確率 p={indep['max_abs_z_pvalue']:.3f}"),
    ]

    descriptive_rows = [
        ("実現ボラ (年率) 中央値", f"{rv['p50']:.5f}",
         f"5-95% [{rv['p05']:.4f}, {rv['p95']:.4f}] / sd(log) {rv['sd_log']:.4f}"),
        ("日次リターン 標準偏差", f"{pooled['std']:.6f}", f"理論 {pooled['expected_std']:.6f}"),
        ("日次リターン 尖度", f"{pooled['kurtosis']:.4f}",
         f"定数ボラなら 3 (s.e. {pooled['kurtosis_se']:.4f})"),
        ("日次 |r| ACF(1)", f"{pooled['abs_acf_lag1']:+.5f}", "定数ボラなら 0"),
        ("日次リターン ACF(1)", f"{pooled['acf_lag1']:+.5f}", f"z = {pooled['acf_lag1_z']:+.2f}"),
        ("日中値幅 log(high/low) 平均", f"{ohlc['mean_log_range']:.6f}",
         (f"理論 {ohlc['expected_log_range_discrete']:.6f}"
          f" (比 {ohlc['range_ratio_vs_discrete']:.4f})") if is_s0
         else f"実体との比 {ohlc['range_to_body_ratio']:.3f}"),
        ("実体 |log(close/open)| 平均", f"{ohlc['mean_abs_body']:.6f}",
         (f"理論 {ohlc['expected_abs_body']:.6f} (比 {ohlc['body_ratio']:.4f})") if is_s0
         else "理論値は定数ボラ限定 (E[σ] < √E[σ²])"),
        ("終端 対数リターン 標準偏差", f"{term['log_return_std']:.5f}",
         f"理論 {term['log_return_std_expected']:.5f}"),
        ("終端 分布の正規性 p (JB)", f"{term['normality_pvalue']:.4f}",
         f"期待: {term['normality_expected']} / 尖度 {term['kurtosis']:.3f}"),
        ("最大ドローダウン 中央値", f"{dd['median']:.4f}", f"95% 点 {dd['p95']:.4f}"),
    ]

    width = max(len(r[0]) for r in universal_rows + descriptive_rows)
    print()
    print(f"段階に依らない不変量 ({stage})")
    print("-" * 78)
    for label, value, note in universal_rows:
        print(f"  {label.ljust(width)}  {value:>18}   {note}")
    print()
    print(f"記述統計 ({stage}: {'定数ボラ' if is_s0 else '確率ボラ — 尖度 3・正規性は成り立たないのが正しい'})")
    print("-" * 78)
    for label, value, note in descriptive_rows:
        print(f"  {label.ljust(width)}  {value:>18}   {note}")
    print("-" * 78)
    print(f"所要 {metrics['runtime_sec']:.0f} 秒")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="模擬チャートを大量生成する")
    parser.add_argument("--n-charts", type=int, default=1000)
    parser.add_argument("--config", type=str, default=None,
                        help="段階の設定ファイル (S0 以外では必須。フラグの取り違えを防ぐ)")
    parser.add_argument("--n-days", type=int, default=None, help="設定ファイルの値を上書き")
    parser.add_argument("--steps-per-day", type=int, default=None, help="同上")
    parser.add_argument("--stage", type=str, default=None, help="同上")
    parser.add_argument("--base-seed", type=int, default=None, help="同上")
    parser.add_argument("--intraday-samples", type=int, default=10,
                        help="分足も保存するチャート本数 (全部書くと数 GB になる)")
    parser.add_argument("--intraday-bar-sec", type=float, default=60.0)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--chunk-size", type=int, default=100,
        help="この本数ごとに parts/ へ逐次書き出す。中断しても完了分は残り、"
             "同じコマンドで再開すると未完了のチャンクだけを作る",
    )
    parser.add_argument(
        "--time-budget-sec", type=float, default=0.0,
        help="この秒数を超えたらチャンクの区切りで正常終了する (0=無制限)。"
             "長時間のバックグラウンド実行が使えない環境で、同じコマンドを"
             "繰り返し呼んで少しずつ進めるために使う",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--recompute", action="store_true",
                        help="経路を作り直さず、既存の出力から集団検証とプロットだけ更新する")
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
