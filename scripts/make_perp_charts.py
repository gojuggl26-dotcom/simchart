"""S0-perp (PerpDEX・24/7) 時点のチャートをランダムなシードで生成する。

    uv run python scripts/make_perp_charts.py --n-charts 10

★このチャートは「それらしく」見えない。それが正解である。
--------------------------------------------------------
S0-perp の価格過程は**幾何ブラウン運動だけ**で、ボラは定数・リターンは正規・
記憶はゼロである (README「S0 で意図的にやっていないこと」の perp 版)。
したがって:

- ローソクの大きさがどこも同じくらい (ボラティリティ・クラスタリングが無い)
- 急落・急騰が無い (テールは S1-perp 以降でボラ過程とジャンプから内生的に出す)
- 出来高の欄が無い

ここに t 分布革新や外生的なテールを足して「それらしく」するのは**禁止事項**で、
時間集計で尖度が下がる性質 (集計正規性) が永久に再現できなくなる。

株式 S0 のチャートとの構造的な違い (perp 固有)
---------------------------------------------
1. **窓が空かない**: 24/7 なのでオーバーナイト・ギャップが存在しない
   (日 d の始値 = 日 d−1 の終値、厳密に一致)。株式チャートの「窓」は
   S4 のギャップ機構によるもので、perp にはその機構自体が無い。
2. **年率換算が 365 日・1 日 86,400 秒** (config の単一情報源 §4)。
3. **板ウォームアップの除外が不要**: S0-perp に板は無い (観測 = p*)。
   S13 のチャートで先頭 5 日を捨てたのは板が定常に達するまでの区間だった。

出来高について
--------------
**出来高の列は作らない。** S0-perp には注文流が無い (L1 はスタブ、L3 板は無効)
ので、出来高を付ければそれは捏造になる (株式 S0 と同じ規約)。代わりに経路から
実際に測れる量として、5 分リターンの**日次実現ボラ**を下段に描く — S0 では
これが平坦になること自体が「ボラが定数である」ことの可視化になる。

出力 (``results/perp_S0/charts/``)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_charts import ohlc_from_log_price  # noqa: E402  (OHLC 定義の単一情報源)

from simchart.config import Config  # noqa: E402
from simchart.pipeline import run  # noqa: E402
from simchart.report import git_info, results_dir  # noqa: E402


# ---------------------------------------------------------------------------
def realized_vol_daily(log_price: np.ndarray, n_days: int, steps_per_day: int,
                       ann_days: int, bar_steps: int) -> np.ndarray:
    """日次実現ボラ (年率)。``bar_steps`` ステップのバーで測る。

    S0-perp の格子は 1 分 (1440 steps/日) なので、既定の 5 ステップ = 5 分足
    288 本/日。GBM では相対 SE = √(2/288) = 8.3% のゆらぎだけが残る。
    """
    sub = log_price[::bar_steps]
    per_day = (sub.shape[0] - 1) // n_days
    r = np.diff(sub[: n_days * per_day + 1])
    rv = (r**2).reshape(n_days, per_day).sum(axis=1)
    return np.sqrt(rv * ann_days)


def max_drawdown(close: np.ndarray) -> float:
    peak = np.maximum.accumulate(close)
    return float(np.max(1.0 - close / peak))


# ---------------------------------------------------------------------------
def generate(cfg: Config, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_days, spd = cfg.n_days, cfg.steps_per_day
    bar_steps = max(int(round(spd / 288)), 1)  # 5 分足 (1440 steps/日 → 5)
    rows, index_rows = [], []
    for cid, seed in enumerate(seeds):
        t0 = time.perf_counter()
        result = run(cfg.replace(seed=seed))
        lp = np.asarray(result.observation.log_price, dtype=np.float64)
        o, h, l, c = ohlc_from_log_price(lp, n_days, spd)
        rv = realized_vol_daily(lp, n_days, spd, cfg.ann_days, bar_steps)

        # ★24/7 の構造検査: 窓が空かないこと (日 d の始値 == 日 d−1 の終値)。
        gap = float(np.max(np.abs(o[1:] - c[:-1]))) if n_days > 1 else 0.0

        po, ph, pl_, pc = (np.exp(v) for v in (o, h, l, c))
        ret = np.empty(n_days)
        ret[0] = c[0] - o[0]
        ret[1:] = np.diff(c)
        rows.append(pd.DataFrame({
            "chart_id": np.full(n_days, cid, dtype=np.int32),
            "day": np.arange(n_days, dtype=np.int32),
            "open": po, "high": ph, "low": pl_, "close": pc,
            "log_return": ret,
            "realized_vol_annualized": rv,
        }))
        index_rows.append({
            "chart_id": cid,
            "seed": int(seed),
            "first_open": float(po[0]),
            "last_close": float(pc[-1]),
            "total_log_return": float(c[-1] - o[0]),
            "annualized_vol": float(ret.std(ddof=1) * math.sqrt(cfg.ann_days)),
            "realized_vol_median": float(np.median(rv)),
            "daily_return_kurtosis": float(stats.kurtosis(ret, fisher=False, bias=False)),
            "daily_return_skew": float(stats.skew(ret, bias=False)),
            "max_drawdown": max_drawdown(pc),
            "max_overnight_gap": gap,
            "close_digest": hashlib.sha256(
                np.ascontiguousarray(c).tobytes()
            ).hexdigest()[:16],
        })
        print(f"  chart {cid} (seed {seed}): {time.perf_counter() - t0:.1f}s"
              f"  年率ボラ {index_rows[-1]['annualized_vol']:.3f}"
              f"  終値 {po[0]:.1f}→{pc[-1]:.1f}", flush=True)
        del result, lp
    return pd.concat(rows, ignore_index=True), pd.DataFrame(index_rows)


# ---------------------------------------------------------------------------
def verify(daily: pd.DataFrame, index_df: pd.DataFrame, cfg: Config,
           n_null: int = 40, null_meta_seed: int = 999) -> dict[str, Any]:
    """「正しく S0 に見えるか」の検証。

    S0 の合格基準は「それらしいこと」ではなく**幾何ブラウン運動であること**:
    尖度 3 / |r| 自己相関ゼロ / 実現ボラが定数 / 年率換算が σ̄ を再現。

    ★日次 ACF には**帰無対照を常時つける** (n_null シードの独立実行)。
    10 本 x 365 日 = 3,650 の日次標本では ACF の抽選ゆらぎが大きく、素の
    |z| > 2 を「実装の欠陥」と読み違える。経緯: 初回実行で z = −3.20 が出て、
    40 シードの帰無分布 (平均 −0.0058、理論バイアス −1/365 に対し z = −0.33)
    と比べて**抽選のゆらぎ**と判明した。事後に検定を足したままにすると
    「結果を見てから検定を変えた」ことになるので、規約として常時実行に固定する。
    ★引いたシードを結果を見てから引き直さないこと (生存バイアス)。
    """
    n = len(index_df)
    n_days = int(daily["day"].max()) + 1
    ret = daily["log_return"].to_numpy().reshape(n, n_days)
    rv = daily["realized_vol_annualized"].to_numpy().reshape(n, n_days)
    pooled = ret.ravel()

    # 1. 年率換算 (§4 の acid test — 252 のままなら √(365/252)=1.20 倍ずれる)
    sd_ann = float(pooled.std(ddof=1) * math.sqrt(cfg.ann_days))
    se_sd = sd_ann / math.sqrt(2.0 * pooled.size)

    # 2. GBM 性
    kurt = float(stats.kurtosis(pooled, fisher=False, bias=False))
    cen = ret - ret.mean(axis=1, keepdims=True)
    acf1 = float((cen[:, :-1] * cen[:, 1:]).sum() / (cen**2).sum())
    a = np.abs(ret) - np.abs(ret).mean(axis=1, keepdims=True)
    absr_acf1 = float((a[:, :-1] * a[:, 1:]).sum() / (a**2).sum())

    # 3. 実現ボラが定数 (S1+ なら Var(log σ)=0.25 → sd(log vol) 0.5)
    bars_per_day = 288
    sd_log_rv = float(np.log(rv).std(ddof=1))
    sd_log_rv_theory = math.sqrt(1.0 / (2.0 * bars_per_day))  # GBM の推定ノイズのみ

    # 4. 24/7 の構造 (窓が空かない)
    max_gap = float(index_df["max_overnight_gap"].max())

    # 5. チャート間の独立性
    corr = np.corrcoef(ret)
    iu = np.triu_indices(n, k=1)
    z = corr[iu] * math.sqrt(n_days - 1)

    # 6. 日次 ACF の帰無対照 (独立な n_null シードで同じ量を測る)
    used = set(int(s) for s in index_df["seed"])
    null_rho: list[float] = []
    null_seeds: list[int] = []
    for s in np.random.default_rng(null_meta_seed).integers(1, 2**31 - 1, size=n_null):
        s = int(s)
        if s in used:
            continue
        r_ = run(cfg.replace(seed=s))
        c_ = np.asarray(r_.observation.log_price)[:: cfg.steps_per_day]
        x_ = np.diff(c_)
        x_ = x_ - x_.mean()
        null_rho.append(float(x_[:-1] @ x_[1:] / (x_ @ x_)))
        null_seeds.append(s)
        del r_
    null = np.asarray(null_rho)
    per_chart_rho = []
    for k in range(n):
        xk = ret[k] - ret[k].mean()
        per_chart_rho.append(float(xk[:-1] @ xk[1:] / (xk @ xk)))
    obs_mean = float(np.mean(per_chart_rho))
    null_mean, null_sd = float(null.mean()), float(null.std(ddof=1))
    bias = -1.0 / n_days  # 標本平均を引くことによる既知バイアス

    return {
        "n_charts": n, "n_days": n_days,
        "annualization": {
            "sd_daily_x_sqrt_ann_days": sd_ann,
            "sigma_bar": cfg.sigma_bar,
            "ann_days": cfg.ann_days,
            "z": (sd_ann - cfg.sigma_bar) / se_sd,
            "note": "§4 の acid test: 252 のままなら 1.20 倍ずれる",
        },
        "gbm_properties": {
            "kurtosis": kurt,
            "kurtosis_se": math.sqrt(24.0 / pooled.size),
            "kurtosis_expected": 3.0,
            "acf_r_lag1": acf1,
            "acf_r_lag1_z": acf1 * math.sqrt(pooled.size),
            "abs_acf_lag1": absr_acf1,
            "abs_acf_lag1_z": absr_acf1 * math.sqrt(pooled.size),
            "skewness": float(stats.skew(pooled, bias=False)),
            "note": "尖度 3・自己相関ゼロが S0 の正解 (テールは S1-perp 以降)",
        },
        "acf_null_control": {
            "per_chart_rho1": per_chart_rho,
            "observed_mean_rho1": obs_mean,
            "null_n_seeds": int(null.size),
            "null_seeds": null_seeds,
            "null_mean_rho1": null_mean,
            "null_sd_rho1": null_sd,
            "expected_bias_minus_1_over_n": bias,
            # 帰無平均が既知バイアスと整合するか = **実装に系統的な自己相関が
            # 無いこと**の判定 (これが本命)
            "systematic_bias_z": (null_mean - bias) / (null_sd / math.sqrt(max(null.size, 1))),
            # 観測 10 本が帰無分布のどこにいるか = 抽選のゆらぎの大きさ
            "observed_vs_null_z": (obs_mean - null_mean) / (null_sd / math.sqrt(n)),
            "note": (
                "10 本 x 365 日では ACF の抽選ゆらぎが大きい。判定は"
                "systematic_bias_z (帰無平均 vs 既知バイアス −1/n) で行う — "
                "素の pooled z は小標本では容易に |2| を超える。"
                "★結果を見てからシードを引き直さないこと (生存バイアス)"
            ),
        },
        "vol_is_constant": {
            "sd_log_realized_vol": sd_log_rv,
            "sd_expected_estimator_noise_only": sd_log_rv_theory,
            "ratio": sd_log_rv / sd_log_rv_theory,
            "note": (
                "5 分足 288 本/日の推定ノイズだけなら比 ≈ 1。確率ボラ (S1-perp) が"
                "入れば sd(log RV) は 0.5 級になる"
            ),
        },
        "structure_24_7": {
            "max_abs_open_minus_prev_close": max_gap,
            "note": "24/7 なので窓は構造的に空かない (株式 S4 の ON ギャップが無い)",
        },
        "independence": {
            "n_pairs": int(z.size),
            "mean_corr": float(corr[iu].mean()),
            "max_abs_z": float(np.max(np.abs(z))),
            "z_std": float(z.std(ddof=1)),
        },
        "descriptive": {
            "annualized_vol_per_chart": [
                float(v) for v in index_df["annualized_vol"]
            ],
            "max_drawdown_median": float(index_df["max_drawdown"].median()),
            "terminal_price_range": [
                float(index_df["last_close"].min()),
                float(index_df["last_close"].max()),
            ],
        },
    }


# ---------------------------------------------------------------------------
def draw_charts(daily: pd.DataFrame, index_df: pd.DataFrame, out_dir: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single = out_dir / "charts"
    single.mkdir(parents=True, exist_ok=True)
    n = len(index_df)
    g_by_id = {cid: g for cid, g in daily.groupby("chart_id")}

    for cid in range(n):
        g = g_by_id[cid]
        row = index_df.iloc[cid]
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(14, 5.6), sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
        )
        x = np.arange(len(g))
        o = g["open"].to_numpy(); h = g["high"].to_numpy()
        lo = g["low"].to_numpy(); c = g["close"].to_numpy()
        up = c >= o
        ax.vlines(x, lo, h, color="0.35", lw=0.5)
        ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.7, color="#2a9d5c", lw=0)
        ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.7, color="#c4453c", lw=0)
        ax.set_ylabel("price")
        ax.grid(alpha=0.15, lw=0.5)
        ax.set_title(
            f"perp S0 — chart {cid:02d}  seed {int(row['seed'])}"
            f"   ann.vol {row['annualized_vol'] * 100:.1f}%"
            f"   maxDD {row['max_drawdown'] * 100:.1f}%"
            f"   (GBM only: flat volatility is the correct S0 answer)",
            fontsize=10, loc="left",
        )
        rv = g["realized_vol_annualized"].to_numpy()
        axv.plot(x, rv * 100.0, lw=0.7, color="#1f4e79")
        axv.axhline(60.0, color="r", ls=":", lw=0.8)
        axv.set_ylabel("real. vol %")
        axv.set_xlabel("day (24/7 — no overnight gaps)")
        axv.grid(alpha=0.15, lw=0.5)
        fig.savefig(single / f"perp_chart_{cid:02d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ギャラリー (終値ライン 10 本)
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.set_visible(False)
            continue
        g = g_by_id[k]
        row = index_df.iloc[k]
        ax.plot(g["close"].to_numpy(), lw=0.9, color="#1f4e79")
        ax.axhline(float(row["first_open"]), color="r", ls=":", lw=0.6)
        ax.set_title(f"{k:02d}  seed {int(row['seed'])}  "
                     f"{row['annualized_vol'] * 100:.0f}%", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"S0-perp simulated charts ({int(daily['day'].max()) + 1} days, 24/7, GBM only, "
        f"sigma=60%/yr) — random seeds",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "gallery.png", dpi=110)
    plt.close(fig)
    return n + 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="S0-perp のチャートを生成する")
    ap.add_argument("--n-charts", type=int, default=10)
    ap.add_argument("--config", type=str, default="configs/s0_perp.yaml")
    ap.add_argument("--n-days", type=int, default=365,
                    help="1 チャートの日数 (既定 365 = perp の 1 年)")
    ap.add_argument("--meta-seed", type=int, default=20260827,
                    help="シードを引くメタシード。★引いたシードは index に記録され、"
                         "seed さえあれば 1 分刻みの全経路を再生成できる")
    ap.add_argument("--results-dir", type=str, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(PROJECT_ROOT / args.config).replace(n_days=args.n_days)
    if cfg.market_type != "perp_clob":
        raise ValueError(f"perp の設定ではありません: market_type={cfg.market_type}")

    # ランダムなシードを引く (メタシードで再現可能)
    rng = np.random.default_rng(args.meta_seed)
    seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=args.n_charts)]

    out_dir = results_dir("perp_S0", args.results_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"S0-perp チャート {args.n_charts} 本 ({cfg.n_days} 日 x "
          f"{cfg.steps_per_day} steps, 24/7, GBM のみ, σ={cfg.sigma_bar}/年, "
          f"ann_days={cfg.ann_days})")
    print(f"  シード (meta {args.meta_seed} から抽選): {seeds}", flush=True)

    daily, index_df = generate(cfg, seeds)
    daily.to_parquet(out_dir / "daily_ohlc.parquet", index=False, compression="zstd")
    index_df.to_parquet(out_dir / "charts_index.parquet", index=False, compression="zstd")
    index_df.to_csv(out_dir / "charts_index.csv", index=False)

    v = verify(daily, index_df, cfg)
    info = git_info()
    payload = {
        "stage": "perp_S0",
        "git_commit": info["commit"],
        "git_dirty": info["dirty"],
        "config_hash": cfg.config_hash(),
        "config": cfg.to_dict(),
        "generation": {
            "n_charts": args.n_charts, "n_days": cfg.n_days,
            "meta_seed": args.meta_seed, "seeds": seeds,
            "market_type": cfg.market_type,
            "ann_days": cfg.ann_days, "seconds_per_day": cfg.seconds_per_day,
            "volume_column": None,
            "volume_note": (
                "S0-perp に注文流は無い (L1 スタブ・L3 板無効) ので出来高は書かない"
                " — 付ければ捏造になる。代わりに 5 分足の日次実現ボラを出す"
            ),
            "runtime_sec": time.perf_counter() - started,
        },
        "verification": v,
    }
    with open(out_dir / "verification.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)

    if not args.no_plots:
        n_img = draw_charts(daily, index_df, out_dir / "images")
        print(f"画像 {n_img} 枚を {out_dir / 'images'} へ")

    ann = v["annualization"]
    gbm = v["gbm_properties"]
    vol = v["vol_is_constant"]
    print()
    print(f"検証 ({v['n_charts']} 本 x {v['n_days']} 日):")
    print(f"  年率換算: 日次 SD x √{ann['ann_days']} = {ann['sd_daily_x_sqrt_ann_days']:.4f}"
          f"  (σ̄ {ann['sigma_bar']}, z={ann['z']:+.2f})")
    print(f"  尖度 {gbm['kurtosis']:.4f} (期待 3, SE {gbm['kurtosis_se']:.3f})"
          f" / ACF(1) z={gbm['acf_r_lag1_z']:+.2f}"
          f" / |r| ACF(1) z={gbm['abs_acf_lag1_z']:+.2f}")
    nc = v["acf_null_control"]
    print(f"  ACF の帰無対照 ({nc['null_n_seeds']} シード): 帰無平均 "
          f"{nc['null_mean_rho1']:+.5f} vs 既知バイアス −1/n "
          f"{nc['expected_bias_minus_1_over_n']:+.5f}"
          f"  → **系統バイアス z={nc['systematic_bias_z']:+.2f}** (判定はこちら)")
    print(f"    今回の 10 本の平均 {nc['observed_mean_rho1']:+.4f} は帰無分布の "
          f"{nc['observed_vs_null_z']:+.2f}σ (抽選のゆらぎ — 引き直さない)")
    print(f"  実現ボラの散らばり sd(log RV) {vol['sd_log_realized_vol']:.4f}"
          f"  (推定ノイズのみなら {vol['sd_expected_estimator_noise_only']:.4f}、"
          f"比 {vol['ratio']:.2f})")
    print(f"  24/7 構造: max|始値−前日終値| = "
          f"{v['structure_24_7']['max_abs_open_minus_prev_close']:.2e}")
    print(f"所要 {time.perf_counter() - started:.0f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
