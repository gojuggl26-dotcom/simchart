"""README 用: perp フォーク (S0-perp) の解説図を作る。

    uv run python scripts/make_perp_fork_figure.py

図は 4 つのことを 1 枚で示す:

1. **時計の違い** — equity は 6.5 時間セッション × 年 252 日 (日境界に窓)、
   perp は 24 時間連続 × 年 365 日 (窓は構造的に存在しない)
2. **同じ乱数・違う時計** — 同一シード・同一 σ̄ で時計だけを変えると、価格経路は
   形が同一のまま振幅が √(252/365) = 0.831 倍になる。**ann_days を切り替え
   忘れると σ が 20% 静かにずれる** — 時間軸の単一情報源 (§4) が防ぐ事故そのもの
3. **S0-perp のチャート** — GBM のみ。それらしく見えないのが正解
4. **その根拠** — 実現ボラが平坦 / 日次リターンが正規 (テールは S1-perp 以降)

★図中のラベルは英語にしてある (matplotlib の既定フォントは日本語を描けず
豆腐になる)。解説は README 側の日本語で行う — 既存の stage_comparison.png と
同じ規約。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sp_stats  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_charts import ohlc_from_log_price  # noqa: E402

from simchart.config import Config  # noqa: E402
from simchart.pipeline import run  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "images"
N_DAYS = 365
STEPS = 1440
SIGMA = 0.60
CHART_SEED = 1227194455  # results/perp_S0/charts の chart 00 と同一
META_SEED = 20260827  # チャート 10 本を引いたメタシード (make_perp_charts.py と同じ)

EQUITY_BLUE = "#4c78a8"
PERP_ORANGE = "#e8871a"


def _perp_cfg(**over) -> Config:
    kw = dict(stage="S0", market_type="perp_clob", session_type="24h",
              n_days=N_DAYS, steps_per_day=STEPS, sigma_bar=SIGMA)
    kw.update(over)
    return Config(**kw)


def _equity_cfg(**over) -> Config:
    kw = dict(stage="S0", n_days=N_DAYS, steps_per_day=STEPS, sigma_bar=SIGMA)
    kw.update(over)
    return Config(**kw)


# ---------------------------------------------------------------------------
def panel_calendar(ax) -> None:
    """取引カレンダーの模式図 (3 日分)。"""
    for d in range(3):
        # equity: 6.5h のセッション → 残り 17.5h は市場が閉じている
        ax.barh(1.0, 6.5, left=d * 24, height=0.42, color=EQUITY_BLUE, zorder=3)
        # perp: 24h 連続
        ax.barh(0.0, 24.0, left=d * 24, height=0.42, color=PERP_ORANGE, zorder=3)
    # equity の日境界 = 窓 (S4 のオーバーナイト・ギャップ)
    for d in range(1, 3):
        ax.annotate(
            "gap", xy=(d * 24, 1.0), xytext=(d * 24, 1.52),
            ha="center", va="bottom", fontsize=8, color="#c4453c",
            arrowprops=dict(arrowstyle="-|>", color="#c4453c", lw=1.0),
        )
    ax.set_xlim(-1, 73)
    ax.set_ylim(-0.6, 1.95)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["perp_clob\n24/7", "equity\n6.5h session"], fontsize=9)
    ax.set_xticks([0, 24, 48, 72])
    ax.set_xticklabels(["day 0", "day 1", "day 2", "day 3"], fontsize=8)
    ax.set_xlabel("wall clock (hours)", fontsize=9)
    ax.set_title(
        "(1) Different clock: 23,400 s/day x 252 vs 86,400 s/day x 365",
        fontsize=10, loc="left",
    )
    ax.text(
        36, -0.5,
        "perp has no session boundary -> no overnight gap by construction"
        "  (measured max |open[d] - close[d-1]| = 0 exactly)",
        ha="center", va="center", fontsize=8, color="0.3",
    )
    ax.grid(axis="x", alpha=0.15, lw=0.5)


def panel_same_seed(ax) -> tuple[float, float]:
    """同じ乱数を違う時計で読む (σ̄ を揃えた対照)。"""
    r_e = run(_equity_cfg(seed=42))
    r_p = run(_perp_cfg(seed=42))
    # 乱数の消費列が同一であることを図の前提として確認する
    assert (
        r_e.meta["l2"]["diffusion_digest"] == r_p.meta["l2"]["diffusion_digest"]
    ), "同一シードでも拡散乱数が違う (market_type が鍵に入っている?)"

    c_e = np.asarray(r_e.observation.log_price)[::STEPS]
    c_p = np.asarray(r_p.observation.log_price)[::STEPS]
    ret_e, ret_p = np.diff(c_e), np.diff(c_p)
    ratio = float(ret_p.std(ddof=1) / ret_e.std(ddof=1))
    theory = math.sqrt(252.0 / 365.0)

    sd_e, sd_p = float(ret_e.std(ddof=1)), float(ret_p.std(ddof=1))
    ann_e, ann_p = sd_e * math.sqrt(252.0), sd_p * math.sqrt(365.0)
    wrong = sd_p * math.sqrt(252.0)  # perp の日次を株式の時計で年率化した場合

    ax.plot(np.exp(c_e - c_e[0]) * 100.0, lw=1.0, color=EQUITY_BLUE,
            label=f"equity clock (252 d/yr): daily SD {sd_e * 100:.2f}%")
    ax.plot(np.exp(c_p - c_p[0]) * 100.0, lw=1.0, color=PERP_ORANGE,
            label=f"perp clock (365 d/yr): daily SD {sd_p * 100:.2f}%")
    ax.axhline(100.0, color="0.6", ls=":", lw=0.7)
    ax.set_ylabel("price", fontsize=9)
    ax.set_xlabel("day", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, lw=0.5)
    ax.set_title(
        "(2) Same random numbers, same sigma_bar - only the clock differs",
        fontsize=10, loc="left",
    )
    ax.text(
        0.02, 0.04,
        f"daily amplitude ratio = sqrt(252/365) = {theory:.3f} (measured {ratio:.3f})\n"
        f"annualized with its OWN clock, each recovers sigma_bar:"
        f"  {ann_e:.3f} / {ann_p:.3f}  (0.60 target; gap is 365-day sampling noise)\n"
        f"annualized with the WRONG clock: {wrong:.3f}"
        f"  ->  {100 * (wrong / ann_p - 1):+.0f}% silent error",
        transform=ax.transAxes, fontsize=8, color="0.25", va="bottom",
    )
    del r_e, r_p
    return ratio, theory


def panel_chart(ax, ax_rv) -> dict:
    """S0-perp のチャート例 (ローソク足) と日次実現ボラ。"""
    cfg = _perp_cfg(seed=CHART_SEED)
    r = run(cfg)
    lp = np.asarray(r.observation.log_price)
    o, h, l, c = (np.exp(v) for v in ohlc_from_log_price(lp, N_DAYS, STEPS))

    x = np.arange(N_DAYS)
    up = c >= o
    ax.vlines(x, l, h, color="0.35", lw=0.45)
    ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.72, color="#2a9d5c", lw=0)
    ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.72, color="#c4453c", lw=0)
    ax.set_ylabel("price", fontsize=9)
    ax.tick_params(labelsize=8, labelbottom=False)
    ax.grid(alpha=0.15, lw=0.5)
    ax.set_title(
        f"(3) An S0-perp chart (seed {CHART_SEED}, 365 days, GBM only) - candles are"
        f" uniformly sized: no clustering, no jumps.\n"
        f"      This 'unrealistic' look is the correct S0 answer; tails and"
        f" clustering arrive endogenously in S1-perp and later.",
        fontsize=10, loc="left",
    )

    # 日次実現ボラ (5 分足 288 本/日)
    bar_steps = max(int(round(STEPS / 288)), 1)
    sub = lp[::bar_steps]
    per_day = (sub.shape[0] - 1) // N_DAYS
    rr = np.diff(sub[: N_DAYS * per_day + 1])
    rv = np.sqrt((rr**2).reshape(N_DAYS, per_day).sum(axis=1) * 365.0)
    ax_rv.plot(x, rv * 100.0, lw=0.7, color="#1f4e79")
    ax_rv.axhline(SIGMA * 100.0, color="r", ls=":", lw=0.9)
    ax_rv.set_ylabel("real. vol %", fontsize=9)
    ax_rv.set_xlabel("day  (24/7: no gaps - open[d] == close[d-1] exactly)", fontsize=9)
    ax_rv.tick_params(labelsize=8)
    ax_rv.grid(alpha=0.15, lw=0.5)
    sd_log_rv = float(np.log(rv).std(ddof=1))
    noise = math.sqrt(1.0 / (2.0 * per_day))
    ax_rv.text(
        0.005, 0.06,
        f"sd(log RV) = {sd_log_rv:.4f} vs estimator noise alone {noise:.4f}"
        f"  ->  volatility really is constant (S1-perp will make it stochastic)",
        transform=ax_rv.transAxes, fontsize=8, color="0.25",
    )
    del r
    return {"sd_log_rv": sd_log_rv, "noise": noise}


def panel_qq(ax) -> dict:
    """10 本プールした日次リターンの正規 QQ (テールが無いことの可視化)。

    ★ここで回す 10 本は ``make_perp_charts.py`` と同じメタシードから引くので、
    results/perp_S0/charts の 10 本と**同一のチャート**になる (数字が食い違わない)。
    日次終値も返して README 用のギャラリー図に使い回す (追加実行なし)。
    """
    rng = np.random.default_rng(META_SEED)
    seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=10)]
    rets, closes = [], []
    for s in seeds:
        r = run(_perp_cfg(seed=s))
        c = np.asarray(r.observation.log_price)[::STEPS]
        rets.append(np.diff(c))
        closes.append(np.exp(c - c[0]) * 100.0)
        del r
    pooled = np.concatenate(rets)
    z = (pooled - pooled.mean()) / pooled.std(ddof=1)
    n = z.size
    q_emp = np.sort(z)
    q_theory = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.plot(q_theory, q_emp, ".", ms=2.0, color=PERP_ORANGE, alpha=0.6)
    lim = float(max(abs(q_theory).max(), abs(q_emp).max())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("normal quantile", fontsize=9)
    ax.set_ylabel("empirical quantile", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, lw=0.5)
    kurt = float(sp_stats.kurtosis(pooled, fisher=False, bias=False))
    se = math.sqrt(24.0 / n)
    ax.set_title(
        "(4) Daily returns are normal - no fat tails yet", fontsize=10, loc="left"
    )
    ax.text(
        0.03, 0.97,
        f"pooled {n:,} daily returns (10 charts)\n"
        f"kurtosis {kurt:.3f}  (expected 3, SE {se:.3f})\n"
        f"fat tails arrive with S1-perp (stochastic vol) and jumps",
        transform=ax.transAxes, fontsize=7.5, va="top", color="0.25",
    )
    return {"kurtosis": kurt, "se": se, "n": n,
            "seeds": seeds, "closes": closes, "rets": rets}


def figure_gallery(seeds, closes, rets, path: Path) -> None:
    """README 用: 同じ 10 本のチャート一覧 (終値)。"""
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.0))
    for k, ax in enumerate(axes.ravel()):
        ann = float(rets[k].std(ddof=1) * math.sqrt(365.0)) * 100.0
        ax.plot(closes[k], lw=0.9, color="#1f4e79")
        ax.axhline(100.0, color="r", ls=":", lw=0.6)
        ax.set_title(f"{k:02d}  seed {seeds[k]}  ann.vol {ann:.0f}%", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.12, lw=0.4)
    fig.suptitle(
        "S0-perp: 10 charts from random seeds (365 days, 24/7, GBM only, "
        "sigma = 60%/yr) - no gaps, no clustering, no jumps",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.5, 12.6))
    gs = fig.add_gridspec(
        4, 2, height_ratios=[0.85, 1.45, 1.65, 0.60],
        width_ratios=[1.75, 1.0], hspace=0.38, wspace=0.16,
    )
    ax_cal = fig.add_subplot(gs[0, :])
    ax_seed = fig.add_subplot(gs[1, 0])
    ax_qq = fig.add_subplot(gs[1, 1])
    ax_chart = fig.add_subplot(gs[2, :])
    # ★実現ボラはチャートの**真下に全幅・同一 x 軸**で置く (半幅だと日付が
    # 揃わず「別の期間の図」に見える)
    ax_rv = fig.add_subplot(gs[3, :], sharex=ax_chart)

    panel_calendar(ax_cal)
    ratio, theory = panel_same_seed(ax_seed)
    rvs = panel_chart(ax_chart, ax_rv)
    qq = panel_qq(ax_qq)

    fig.suptitle(
        "simchart perp fork (S0-perp): same codebase, different market type",
        fontsize=13, y=0.995,
    )
    path = OUT / "perp_fork_overview.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    path2 = OUT / "perp_s0_charts.png"
    figure_gallery(qq["seeds"], qq["closes"], qq["rets"], path2)

    print(f"{path}")
    print(f"{path2}")
    print(f"  amplitude ratio measured {ratio:.4f} / theory {theory:.4f}")
    print(f"  sd(log RV) {rvs['sd_log_rv']:.4f} vs noise {rvs['noise']:.4f}")
    print(f"  pooled kurtosis {qq['kurtosis']:.4f} (SE {qq['se']:.4f}, n={qq['n']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
