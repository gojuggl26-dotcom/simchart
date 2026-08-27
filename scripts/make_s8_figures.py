"""S8 (メタオーダー分割) の図を作る。

README に載せる 3 枚:

1. ``s8_signs.png`` — 符号 ACF の冪則 ((11))・メタオーダー長の離散 Pareto 裾・
                        プール占有の経路 (whale 滞留)
2. ``s8_deficit.png`` — インパクト赤字: R(ℓ) 増大・G(ℓ) 非減衰 ((15) の赤字)・
                        約定時間 VR の超拡散 (S7 対照)・サイズ応答の線形性
3. ``s8_price.png`` — 同一シードの S7 vs S8 ミッド経路 + 複数シードの
                        トレンド多様性 (whale の出方で実現超拡散が変わる)

ラベルは英語 (既存の慣習)。数値の正は results/S8/metrics.json。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simchart import Config, run
from simchart.validation.memory import acf
from simchart.validation.suite import _meta_metrics

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
BLUE, ORANGE, GREY, GREEN, RED = "#1f4e79", "#d1701e", "#8a8a8a", "#2e7d32", "#b03030"
S = 23400.0


def _signs(r, burn_days: float):
    s = np.asarray(r.events.meta["agg_trade_side"], dtype=np.float64)
    t = np.asarray(r.events.meta["agg_trade_t"])
    return s[t >= burn_days * S]


def fig_signs(r8, cfg8, r7) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    burn = cfg8.book_burn_in_days

    # 符号 ACF (log-log)
    ax = axes[0]
    for r, color, label in ((r7, GREY, "S7 (symmetric signs)"), (r8, BLUE, "S8")):
        s = _signs(r, burn)
        base = acf(s, max_lag=1000)
        lags = np.asarray(base["lags"], dtype=float)
        vals = np.array([np.nan if v is None else v for v in base["values"]])
        keep = (lags >= 1) & np.isfinite(vals) & (vals > 0)
        ax.loglog(lags[keep][::1], vals[keep][::1], ".", ms=2.5, color=color,
                  label=label, alpha=0.7)
    xs = np.geomspace(1, 1000, 50)
    c1 = 0.13
    ax.loglog(xs, c1 * xs ** (-0.6), "--", color=ORANGE, lw=1.6,
              label=r"$C(1)\,\ell^{-(\alpha-1)}$, $\alpha$=1.6")
    ax.set_xlabel("lag (aggressor orders, log)")
    ax.set_ylabel("sign ACF (log; positive values)")
    ax.set_title("stylized fact #11 arrives: power-law sign memory from metaorder splitting\n"
                 "(S7's points are just noise scatter around zero)", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.set_ylim(1e-4, 0.5)

    # 長さ分布 CCDF
    ax = axes[1]
    n_tot = r8.events.meta["metaorders"]["n_total"]
    vals, counts = np.unique(n_tot, return_counts=True)
    ccdf = 1.0 - np.cumsum(counts) / counts.sum()
    keep = ccdf > 0
    ax.loglog(vals[keep], np.concatenate([[1.0], ccdf[keep][:-1]]), ".", ms=4,
              color=BLUE, label="measured $P(N \\geq n)$")
    xs = np.geomspace(1, vals.max(), 50)
    ax.loglog(xs, xs ** (-1.6), "--", color=ORANGE, label=r"$n^{-\alpha}$, $\alpha$=1.6")
    ax.set_xlabel("metaorder length N (children)")
    ax.set_ylabel("CCDF")
    ax.set_title("discrete Pareto tail is exact by construction\n"
                 f"(floor sampler; MLE $\\hat\\alpha$ recovers to ±0.01, "
                 f"max N = {int(n_tot.max())})", fontsize=9.5)
    ax.legend(fontsize=8.5)

    # プール占有
    ax = axes[2]
    pool = np.asarray(r8.events.meta["pool_grid"])
    spd = int(round(S / r8.observation.step_seconds))
    days = np.arange(pool.size) / spd
    keep = (days >= 30) & (days <= 90)
    ax.plot(days[keep], pool[keep], color=BLUE, lw=0.7)
    ax.axhline(np.median(pool), color=ORANGE, ls="--", lw=1.4,
               label=f"median = {np.median(pool):.0f}")
    ax.set_xlabel("trading day (60-day excerpt)")
    ax.set_ylabel("active metaorders in pool")
    ax.set_title("the pool is lean (median ~3) but heavy-tailed:\n"
                 "spikes are whale episodes — a big metaorder lingers for days\n"
                 "because uniform picking gives it only a 1/A share", fontsize=9.5)
    ax.legend(fontsize=8.5)

    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(IMAGES / "s8_signs.png", dpi=140)
    plt.close(fig)


def fig_deficit(r8, cfg8, r7, m8) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    burn = cfg8.book_burn_in_days

    # R(l) と G(l)
    ax = axes[0]
    resp = m8["response_mid"]
    prop = m8["propagator_mid"]
    lags = np.asarray(resp["lags"], dtype=float)
    rv = np.array([np.nan if v is None else v for v in resp["values"]]) * 1e4
    gv = np.array([np.nan if v is None else v for v in prop["propagator"]]) * 1e4
    ax.plot(lags, rv, color=BLUE, lw=1.4, label=r"response $R(\ell)$")
    # G は打ち切りラグ近傍 (>170) に NNLS の端点アーティファクトが出るので表示を切る
    show = lags <= 170
    ax.plot(lags[show], gv[show], color=GREEN, lw=1.4,
            label=r"propagator $G(\ell)$ (solved)")
    gtar = gv[4] * (lags / 5.0) ** (-(1 - 0.585) / 2)
    ax.plot(lags[4:], gtar[4:], "--", color=ORANGE, lw=1.4,
            label=r"efficient target $\ell^{-(1-\gamma)/2}$")
    ax.set_xlabel("lag (aggressor orders)")
    ax.set_ylabel("bp")
    ax.set_title("propagator (#15) deficit: impact should DECAY like the orange line to keep\n"
                 "prices diffusive — instead G is flat-to-rising (β̂ ≈ −0.25 vs\n"
                 "target +0.21) and R grows without saturating", fontsize=9.5)
    ax.legend(fontsize=8.5)

    # 約定時間 VR
    ax = axes[1]
    for r, color, label in ((r7, GREY, "S7"), (r8, BLUE, "S8")):
        pm = np.asarray(r.events.meta["agg_trade_prev_mid_tick"], dtype=np.float64)
        t = np.asarray(r.events.meta["agg_trade_t"])
        pm = pm[(t >= burn * S)]
        pm = pm[np.isfinite(pm)]
        lp = np.log(r.events.meta["base_price"] + r.events.meta["tick_size"] * pm)
        v1 = np.diff(lp).var()
        ns = np.array([3, 10, 30, 100, 300, 1000, 3000])
        vr = [float((lp[n:] - lp[:-n]).var() / (n * v1)) for n in ns]
        ax.semilogx(ns, vr, "o-", color=color, label=label)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(1.3, color=RED, lw=1.0, ls="--", label="gate: VR(1000) > 1.3")
    ax.set_xlabel("aggregation window (trades, log)")
    ax.set_ylabel("variance ratio (trade time)")
    ax.set_title("superdiffusion in trade time: the ZI spring wins below ~30\n"
                 "trades, the sign memory wins above — S7 stays at 1", fontsize=9.5)
    ax.legend(fontsize=8.5)

    # サイズ応答
    ax = axes[2]
    bins = m8["impact_vs_size"]["bins"]
    x = [(b["n_lo"] * b["n_hi"]) ** 0.5 for b in bins]
    y = [b["mean_impact"] * 1e4 for b in bins]
    e = [b["se"] * 1e4 for b in bins]
    ax.errorbar(x, y, yerr=e, fmt="o", color=BLUE, ms=5, label="binned mean impact")
    xs = np.geomspace(1, max(x), 30)
    ax.loglog(xs, y[0] * xs ** 1.0, "--", color=GREY, lw=1.2, label="slope 1 (linear)")
    ax.loglog(xs, y[0] * xs ** 0.5, ":", color=ORANGE, lw=1.6,
              label="slope 0.5 (sqrt law — S10 goal)")
    ax.set_xlabel("metaorder size N (children, log)")
    ax.set_ylabel("mean signed impact (bp, log)")
    ax.set_title("square-root law (#16) deficit: impact is ~linear in size (slope 0.91) because\n"
                 "each child adds a constant kick — no liquidity response yet",
                 fontsize=9.5)
    ax.legend(fontsize=8.5)

    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(IMAGES / "s8_deficit.png", dpi=140)
    plt.close(fig)


def fig_price(cfg8) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

    # S7 vs S8 (同一シード)
    ax = axes[0]
    cfg7 = Config.load(ROOT / "configs" / "s7.yaml").replace(
        n_days=cfg8.n_days, steps_per_day=cfg8.steps_per_day,
        book_window_half_ticks=cfg8.book_window_half_ticks, seed=cfg8.seed,
    )
    r7 = run(cfg7)
    r8 = run(cfg8)
    for r, color, label in ((r7, GREY, "S7 (no sign memory)"), (r8, BLUE, "S8")):
        obs = r.observation
        days = obs.t / S
        stride = max(1, int(600.0 / obs.step_seconds))
        ax.plot(days[::stride], np.exp(obs.log_price[::stride]), color=color,
                lw=0.8, label=label)
    ax.set_xlabel("trading day")
    ax.set_ylabel("mid price")
    ax.set_title("same seed, same book, same Hawkes clock — only the SIGNS\n"
                 "changed, and the price now trends (impact deficit §8)", fontsize=9.5)
    ax.legend(fontsize=9)

    # 複数シードの S8 経路
    ax = axes[1]
    for i, seed in enumerate((43, 44, 45, 46, 47)):
        r = run(cfg8.replace(seed=seed))
        obs = r.observation
        days = obs.t / S
        stride = max(1, int(600.0 / obs.step_seconds))
        ax.plot(days[::stride], np.exp(obs.log_price[::stride]), lw=0.8,
                label=f"seed {seed}")
    ax.set_xlabel("trading day")
    ax.set_ylabel("mid price")
    ax.set_title("realized superdiffusion is whale luck: some seeds trend hard,\n"
                 "some meander — that's why the VR gate uses the 10-seed median",
                 fontsize=9.5)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s8_price.png", dpi=140)
    plt.close(fig)


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("S8/S7 を生成中 (250 日)...", flush=True)
    cfg8 = Config.load(ROOT / "configs" / "s8.yaml")
    r8 = run(cfg8)
    cfg7 = Config.load(ROOT / "configs" / "s7.yaml").replace(
        n_days=cfg8.n_days, steps_per_day=cfg8.steps_per_day,
        book_window_half_ticks=cfg8.book_window_half_ticks, seed=cfg8.seed,
    )
    r7 = run(cfg7)
    m8 = _meta_metrics(r8, cfg8)
    print("図 1/3: 符号 ACF・長さ分布・プール", flush=True)
    fig_signs(r8, cfg8, r7)
    print("図 2/3: インパクト赤字", flush=True)
    fig_deficit(r8, cfg8, r7, m8)
    print("図 3/3: 価格経路", flush=True)
    fig_price(cfg8)
    for name in ("s8_signs", "s8_deficit", "s8_price"):
        p = IMAGES / f"{name}.png"
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
