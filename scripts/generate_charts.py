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
from simchart.config import TRADING_DAYS_PER_YEAR
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


def max_drawdown(close: np.ndarray) -> float:
    """終値ベースの最大ドローダウン (正の比率)。"""
    peak = np.maximum.accumulate(close)
    return float(np.max(1.0 - close / peak))


# ---------------------------------------------------------------------------
def generate(args: argparse.Namespace) -> int:
    base = Config(
        seed=args.base_seed, n_days=args.n_days, steps_per_day=args.steps_per_day,
        stage=args.stage,
    )
    n_days, steps_per_day = base.n_days, base.steps_per_day
    session_seconds = 6.5 * 3600.0
    step_seconds = session_seconds / steps_per_day
    seconds_per_year = TRADING_DAYS_PER_YEAR * session_seconds

    if args.intraday_bar_sec % step_seconds != 0:
        raise ValueError("intraday_bar_sec が刻みの整数倍ではありません")
    steps_per_intraday_bar = int(args.intraday_bar_sec / step_seconds)
    intraday_bars_per_day = steps_per_day // steps_per_intraday_bar

    out_dir = results_dir(args.stage, args.results_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recompute:
        return recompute(out_dir, args)

    n = args.n_charts
    total_days = n * n_days
    chart_id = np.repeat(np.arange(n, dtype=np.int32), n_days)
    day_index = np.tile(np.arange(n_days, dtype=np.int32), n)
    open_px = np.empty(total_days, dtype=np.float64)
    high_px = np.empty(total_days, dtype=np.float64)
    low_px = np.empty(total_days, dtype=np.float64)
    close_px = np.empty(total_days, dtype=np.float64)
    log_return = np.empty(total_days, dtype=np.float64)
    realized_vol = np.empty(total_days, dtype=np.float64)

    index_rows: list[dict[str, Any]] = []
    intraday_frames: list[pd.DataFrame] = []
    started = time.perf_counter()

    print(f"{n} 本 x {n_days} 日 x {steps_per_day} ステップ を生成します "
          f"(stage={args.stage}, seed={args.base_seed}..{args.base_seed + n - 1})")

    for i in range(n):
        seed = args.base_seed + i
        result = run(base.replace(seed=seed))
        log_price = result.observation.log_price

        o, h, l, c = ohlc_from_log_price(log_price, n_days, steps_per_day)
        step_returns = np.diff(log_price)
        rv_daily = (step_returns.reshape(n_days, steps_per_day) ** 2).sum(axis=1)

        sl = slice(i * n_days, (i + 1) * n_days)
        open_px[sl] = np.exp(o)
        high_px[sl] = np.exp(h)
        low_px[sl] = np.exp(l)
        close_px[sl] = np.exp(c)
        log_return[sl] = c - o
        realized_vol[sl] = np.sqrt(rv_daily * TRADING_DAYS_PER_YEAR)

        daily_ret = c - o
        index_rows.append(
            {
                "chart_id": i,
                "seed": seed,
                "first_open": float(np.exp(o[0])),
                "last_close": float(np.exp(c[-1])),
                "total_log_return": float(c[-1] - o[0]),
                "realized_vol_annualized": float(
                    np.sqrt(rv_daily.sum() / n_days * TRADING_DAYS_PER_YEAR)
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

        if (i + 1) % args.progress_every == 0 or i + 1 == n:
            elapsed = time.perf_counter() - started
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {i + 1}/{n}  経過 {elapsed:.0f} 秒 / 残り約 {eta:.0f} 秒", flush=True)

    # ------------------------------------------------------------------
    daily = pd.DataFrame(
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
    )
    daily_path = out_dir / "daily_ohlc.parquet"
    daily.to_parquet(daily_path, index=False, compression="zstd")

    index_df = pd.DataFrame(index_rows)
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
        "stage": args.stage,
        "git_commit": info["commit"],
        "git_dirty": info["dirty"],
        "config": base.to_dict(),
        "config_hash": base.config_hash(),
        "generation": {
            "n_charts": n,
            "n_days": n_days,
            "steps_per_day": steps_per_day,
            "step_seconds": step_seconds,
            "base_seed": args.base_seed,
            "seeds": [args.base_seed, args.base_seed + n - 1],
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

    report(metrics, daily_path, index_path, intraday_path, out_dir)
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

    seconds_per_year = TRADING_DAYS_PER_YEAR * 6.5 * 3600.0
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
           intraday if intraday.exists() else None, out_dir)
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
    """1000 本まとめての検証。

    個々のチャートが「それらしく」見えても、集団として幾何ブラウン運動に
    なっていなければ意味がない。特に確認するのは 4 点:

    1. 実現ボラが全チャートで ``sigma_bar`` に一致すること
    2. 日次リターンをプールした尖度が 3、自己相関が 0 であること
    3. 終端の期待値が **1 倍** であること (伊藤補正 −σ²/2 が入っているか)
    4. チャート同士が独立であること (シードをずらしただけで系列が相関していないか)
    """
    returns = daily["log_return"].to_numpy().reshape(n_charts, n_days)
    pooled = returns.ravel()
    terminal = returns.sum(axis=1)
    horizon_years = n_days / TRADING_DAYS_PER_YEAR
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

    # --- 5. OHLC の高値・安値が本当に日中経路から来ているか ---
    # ブラウン運動の 1 期間の値幅は E[max - min] = sigma * sqrt(8/pi)、
    # 始値終値の差は E|close - open| = sigma * sqrt(2/pi)。高値・安値は終値だけからは
    # 復元できない量なので、ここが合うことが「日中を本当に見ている」証拠になる。
    # 1 秒刻みの離散標本なので値幅はわずかに過小になる (刻み間の極値を見逃すため)。
    # Broadie-Glasserman-Kou の補正で相対 -2*0.5826*sqrt(dt/T) 程度。
    sigma_day = sigma / math.sqrt(TRADING_DAYS_PER_YEAR)
    steps_per_day = config.steps_per_day
    log_range = np.log(daily["high"].to_numpy() / daily["low"].to_numpy())
    abs_body = np.abs(daily["log_return"].to_numpy())
    range_theory = sigma_day * math.sqrt(8.0 / math.pi)
    discrete_factor = 1.0 - 2.0 * 0.5826 / math.sqrt(steps_per_day) / math.sqrt(8.0 / math.pi)

    return {
        "ohlc": {
            "mean_log_range": num(log_range.mean()),
            "expected_log_range_continuous": num(range_theory),
            "expected_log_range_discrete": num(range_theory * discrete_factor),
            "range_ratio_vs_discrete": num(log_range.mean() / (range_theory * discrete_factor)),
            "mean_abs_body": num(abs_body.mean()),
            "expected_abs_body": num(sigma_day * math.sqrt(2.0 / math.pi)),
            "body_ratio": num(abs_body.mean() / (sigma_day * math.sqrt(2.0 / math.pi))),
            "steps_per_day": int(steps_per_day),
        },
        "realized_vol": {
            "mean": num(vols.mean()),
            "std": num(vols.std(ddof=1)),
            "min": num(vols.min()),
            "max": num(vols.max()),
            "target": num(sigma),
            "max_abs_dev_from_target": num(np.max(np.abs(vols - sigma))),
        },
        "pooled_daily_returns": {
            "n": int(n_pooled),
            "mean": num(pooled.mean()),
            "expected_mean": num(-0.5 * sigma**2 / TRADING_DAYS_PER_YEAR),
            "std": num(pooled.std(ddof=1)),
            "expected_std": num(sigma / math.sqrt(TRADING_DAYS_PER_YEAR)),
            "skewness": num(stats.skew(pooled, bias=False)),
            "kurtosis": num(stats.kurtosis(pooled, fisher=False, bias=False)),
            "kurtosis_se": num(math.sqrt(24.0 / n_pooled)),
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
            "normality_pvalue": num(stats.jarque_bera(terminal).pvalue),
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
    horizon = days / TRADING_DAYS_PER_YEAR
    sigma, mu = config.sigma_bar, config.mu_drift
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for lo, hi, alpha in ((5, 95, 0.18), (25, 75, 0.28)):
        ax.fill_between(
            days, np.percentile(close, lo, axis=0), np.percentile(close, hi, axis=0),
            alpha=alpha, color="C0", lw=0,
            label=f"empirical {lo}-{hi}%",
        )
    ax.plot(days, np.median(close, axis=0), color="C0", lw=1.2, label="empirical median")
    for q, style in ((0.05, "--"), (0.5, "-"), (0.95, "--")):
        z = stats.norm.ppf(q)
        theo = p0 * np.exp((mu - 0.5 * sigma**2) * horizon + sigma * np.sqrt(horizon) * z)
        ax.plot(days, theo, color="r", ls=style, lw=1.0,
                label="theoretical GBM" if q == 0.5 else None)
    ax.set_xlabel("day")
    ax.set_ylabel("price")
    ax.set_title(f"S0 fan chart: {n_charts} charts vs theoretical GBM quantiles")
    ax.legend(fontsize=8)
    save(fig, "fan_chart.png")

    # 3. 終端分布
    terminal = np.log(close[:, -1] / p0)
    horizon_total = n_days / TRADING_DAYS_PER_YEAR
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(terminal, bins=50, density=True, alpha=0.65)
    grid = np.linspace(terminal.min(), terminal.max(), 400)
    axes[0].plot(
        grid,
        stats.norm.pdf(grid, (mu - 0.5 * sigma**2) * horizon_total,
                       sigma * math.sqrt(horizon_total)),
        "r-", lw=1.3, label="theoretical N",
    )
    axes[0].set_xlabel("terminal log return")
    axes[0].set_title("terminal log return vs theory")
    axes[0].legend(fontsize=8)
    stats.probplot(terminal, dist="norm", plot=axes[1])
    axes[1].set_title("normal QQ of terminal log return")
    save(fig, "terminal_distribution.png")

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
    intraday_path: Path | None, out_dir: Path,
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
    rows = [
        ("実現ボラ (年率) 平均", f"{rv['mean']:.5f}", f"目標 {rv['target']:.3f}"),
        ("実現ボラ 範囲", f"[{rv['min']:.5f}, {rv['max']:.5f}]", ""),
        ("日中値幅 log(high/low) 平均", f"{ohlc['mean_log_range']:.6f}",
         f"理論 {ohlc['expected_log_range_discrete']:.6f} (比 {ohlc['range_ratio_vs_discrete']:.4f})"),
        ("実体 |log(close/open)| 平均", f"{ohlc['mean_abs_body']:.6f}",
         f"理論 {ohlc['expected_abs_body']:.6f} (比 {ohlc['body_ratio']:.4f})"),
        ("日次リターン 標準偏差", f"{pooled['std']:.6f}", f"理論 {pooled['expected_std']:.6f}"),
        ("日次リターン 平均", f"{pooled['mean']:.3e}", f"理論 {pooled['expected_mean']:.3e}"),
        ("日次リターン 尖度", f"{pooled['kurtosis']:.4f}", f"3 (s.e. {pooled['kurtosis_se']:.4f})"),
        ("日次リターン ACF(1)", f"{pooled['acf_lag1']:+.5f}", f"z = {pooled['acf_lag1_z']:+.2f}"),
        ("終端 対数リターン 平均", f"{term['log_return_mean']:+.5f}",
         f"理論 {term['log_return_mean_expected']:+.5f} (s.e. {term['log_return_mean_se']:.5f})"),
        ("終端 対数リターン 標準偏差", f"{term['log_return_std']:.5f}",
         f"理論 {term['log_return_std_expected']:.5f}"),
        ("終端 単純リターン 平均", f"{term['gross_return_mean']:.5f}",
         f"理論 1.00000 (s.e. {term['gross_return_mean_se']:.5f})"),
        ("チャート間 相関 最大|z|", f"{indep['max_abs_z']:.2f}",
         f"目安 {indep['expected_max_abs_z']:.2f} / 超過確率 p={indep['max_abs_z_pvalue']:.3f}"),
        ("チャート間 相関 z の標準偏差", f"{indep['z_std']:.4f}",
         f"独立なら 1 / |z|>1.96 が {indep['frac_abs_z_over_1_96'] * 100:.2f}% (理論 5%)"),
        ("最大ドローダウン 中央値", f"{dd['median']:.4f}", f"95% 点 {dd['p95']:.4f}"),
    ]
    width = max(len(r[0]) for r in rows)
    print()
    print("集団としての検証")
    print("-" * 72)
    for label, value, note in rows:
        print(f"  {label.ljust(width)}  {value:>22}   {note}")
    print("-" * 72)
    print(f"所要 {metrics['runtime_sec']:.0f} 秒")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="模擬チャートを大量生成する")
    parser.add_argument("--n-charts", type=int, default=1000)
    parser.add_argument("--n-days", type=int, default=500)
    parser.add_argument("--steps-per-day", type=int, default=23400)
    parser.add_argument("--stage", type=str, default="S0")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--intraday-samples", type=int, default=10,
                        help="分足も保存するチャート本数 (全部書くと数 GB になる)")
    parser.add_argument("--intraday-bar-sec", type=float, default=60.0)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--recompute", action="store_true",
                        help="経路を作り直さず、既存の出力から集団検証とプロットだけ更新する")
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
