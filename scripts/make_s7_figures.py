"""S7 (Hawkes 注文流) の図を作る。

README に載せる 3 枚:

1. ``s7_three_way.png``  — 分岐比 3 経路 (Filimonov–Sornette の罠の実証) と
                           強度 λ(t) の再構成 (バーストの可視化)
2. ``s7_clustering.png`` — S6 vs S7: 分ごとの件数・間隔の生存関数・Fano の窓依存
3. ``s7_seasonality.png``— 日内 U 字 (件数 vs φ_λ) と出来高 ACF

ラベルは英語 (既存の慣習)。数値の正は results/S7/metrics.json。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simchart import Config, run
from simchart.validation.hawkes import (
    branching_three_ways,
    excitation_pass,
    hawkes_mle,
    marks_from_eventlog,
    phi_lookup,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
BLUE, ORANGE, GREY, GREEN, RED = "#1f4e79", "#d1701e", "#8a8a8a", "#2e7d32", "#b03030"
S = 23400.0


def _results(n_days: int = 120):
    cfg7 = Config.load(ROOT / "configs" / "s7.yaml").replace(n_days=n_days, steps_per_day=390)
    cfg6 = Config.load(ROOT / "configs" / "s6.yaml").replace(n_days=n_days, steps_per_day=390)
    return run(cfg7), cfg7, run(cfg6), cfg6


def _phi_table(cfg: Config) -> np.ndarray:
    from simchart.layers.l0_calendar import build_calendar
    from simchart.rng import RNGRegistry

    cal = build_calendar(cfg, RNGRegistry(cfg.seed))
    u = (np.arange(4096, dtype=np.float64) + 0.5) / 4096
    return np.asarray(cal.phi_lambda_of_u(u))


def fig_three_way(r7, cfg7) -> None:
    times, marks = marks_from_eventlog(r7.events)
    t_end = cfg7.n_days * S
    a = np.asarray(cfg7.hawkes_a)
    betas = 1.0 / np.asarray(cfg7.hawkes_tau_seconds)
    w = np.asarray(cfg7.hawkes_weights)
    n_design = float(np.max(np.abs(np.linalg.eigvals(a))))
    phi = _phi_table(cfg7)
    three = branching_three_ways(times, marks, t_end, betas, w, S, phi, n_design)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax = axes[0]
    labels = ["raw\n(no deseason.)", "true $\\varphi_\\lambda$", r"estimated $\hat\varphi_\lambda$"]
    vals = [three["n_hat_raw"], three["n_hat_true_phi"], three["n_hat_est_phi"]]
    colors = [RED, BLUE, GREEN]
    ax.bar(labels, vals, color=colors, width=0.55)
    ax.axhline(n_design, color="k", lw=1.2, ls="--", label=f"design n = {n_design:.3f}")
    ax.axhspan(n_design - 0.05, n_design + 0.05, color=BLUE, alpha=0.10,
               label=r"gate $\pm0.05$ (true-$\varphi$)")
    for x, v in enumerate(vals):
        ax.annotate(f"{v:.4f}", (x, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.78, 0.93)
    ax.set_ylabel(r"re-estimated branching ratio $\hat n$")
    ax.set_title(
        "the Filimonov–Sornette trap, demonstrated on our own engine:\n"
        "skip deseasonalization and the intraday U-shape masquerades as\n"
        "self-excitation (+0.066) — this is why S4's machinery exists",
        fontsize=9.5,
    )
    ax.legend(fontsize=8.5, loc="upper right")

    # 強度の再構成 (最も活発な 20 分)
    ax = axes[1]
    fit = hawkes_mle(times, marks, t_end, betas, w, phi_table=phi, session_seconds=S)
    e_mat, _, _ = excitation_pass(times, marks, betas, w, t_end)
    phi_i = phi_lookup(times, phi, S)
    lam = (
        phi_i * fit["mu_hat_per_sec"].sum()
        + e_mat @ fit["a_hat"].sum(axis=1)
    )
    # burn 後で強度最大のイベントを中心に 20 分
    burn = cfg7.book_burn_in_days * S
    idx_ok = np.flatnonzero(times >= burn)
    center = times[idx_ok[np.argmax(lam[idx_ok])]]
    lo, hi = center - 600.0, center + 600.0
    m_win = (times >= lo) & (times <= hi)
    tw = (times[m_win] - lo) / 60.0
    base = (phi_i * fit["mu_hat_per_sec"].sum())[m_win] * 60.0
    ax.semilogy(tw, lam[m_win] * 60.0, color=BLUE, lw=0.9, label=r"fitted $\lambda(t)$")
    ax.semilogy(tw, base, color=ORANGE, lw=1.4, ls="--",
                label=f"seasonal baseline (≈{base.mean():.0f}/min)")
    ax.plot(tw, np.full(tw.size, base.min() * 0.45), "|", color=GREY, ms=7, alpha=0.6,
            label="events")
    ax.set_xlabel("minutes (20-min window around the largest burst)")
    ax.set_ylabel("total intensity (events / min, log)")
    ax.set_title(
        "reconstructed intensity over the event stream (ticks at bottom):\n"
        "bursts reach 2 orders of magnitude above baseline and decay on the\n"
        "0.5s/10s/300s kernel scales — they are cascades, not baseline moves",
        fontsize=9.5,
    )
    ax.legend(fontsize=8.5, loc="upper left")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s7_three_way.png", dpi=140)
    plt.close(fig)


def fig_clustering(r7, cfg7, r6, cfg6) -> None:
    t7, _ = marks_from_eventlog(r7.events)
    t6, _ = marks_from_eventlog(r6.events)
    burn = cfg7.book_burn_in_days * S
    horizon = cfg7.n_days * S

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    # 分ごとの件数 (同じ 1 日)
    ax = axes[0]
    day = int(burn / S) + 3
    for t, color, label in ((t6, GREY, "S6 (Poisson)"), (t7, BLUE, "S7 (Hawkes)")):
        sel = (t >= day * S) & (t < (day + 1) * S)
        c, e = np.histogram(t[sel], bins=np.arange(day * S, (day + 1) * S + 60.0, 60.0))
        ax.plot((e[:-1] - day * S) / 3600.0, c, color=color, lw=0.9, label=label,
                alpha=0.9 if color == BLUE else 0.8)
    ax.set_xlabel("hours into the session (same simulated day)")
    ax.set_ylabel("events / minute")
    ax.set_title("same seed, same day — the flow now arrives in bursts", fontsize=9.5)
    ax.legend(fontsize=8.5)

    # 間隔の生存関数
    ax = axes[1]
    for t, color, label in ((t6, GREY, "S6"), (t7, BLUE, "S7")):
        d = np.diff(t[t >= burn])
        d = d[d > 0]
        x = np.sort(d) / d.mean()
        sf = 1.0 - np.arange(1, x.size + 1) / x.size
        keep = sf > 1e-6
        ax.semilogy(x[keep][::200], sf[keep][::200], color=color, lw=1.1, label=label)
    xs = np.linspace(0, 12, 100)
    ax.semilogy(xs, np.exp(-xs), "k--", lw=0.9, label="exponential")
    ax.set_xlim(0, 12)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("inter-event time / mean")
    ax.set_ylabel("survival function (log)")
    ax.set_title("S7 has excess mass at BOTH ends (bursts + lulls),\n"
                 f"CV² = 6.4 vs 1.0 — KS rejects exponential at p ≈ 0", fontsize=9.5)
    ax.legend(fontsize=8.5)

    # Fano vs 窓
    ax = axes[2]
    wins = np.array([10, 30, 60, 180, 600, 1800])
    for t, color, label in ((t6, GREY, "S6"), (t7, BLUE, "S7")):
        f = []
        for win in wins:
            c, _ = np.histogram(t[t >= burn], bins=np.arange(burn, horizon + win, win))
            f.append(c.var() / c.mean())
        ax.loglog(wins, f, "o-", color=color, label=label)
    n = 0.83
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(1.0 / (1.0 - n) ** 2, color=ORANGE, lw=1.2, ls="--",
               label=r"$(1-n)^{-2}$ = 34.6 (homogeneous limit)")
    ax.set_xlabel("counting window (seconds, log)")
    ax.set_ylabel("Fano factor (log)")
    ax.set_title("overdispersion grows with the window; beyond ~30 min the\n"
                 "seasonal U-shape adds variance on top of the Hawkes limit", fontsize=9.5)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(IMAGES / "s7_clustering.png", dpi=140)
    plt.close(fig)


def fig_seasonality(r7, cfg7) -> None:
    times, _ = marks_from_eventlog(r7.events)
    burn = cfg7.book_burn_in_days * S
    t_end = cfg7.n_days * S
    phi = _phi_table(cfg7)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    t_b = times[times >= burn]
    u = np.mod(t_b / S, 1.0)
    n_bins = 26
    counts, edges = np.histogram(u, bins=np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    days = (t_end - burn) / S
    ax.bar(centers * 6.5, counts / days / (S / n_bins / 60.0), width=6.5 / n_bins * 0.9,
           color=BLUE, alpha=0.75, label="measured events/min")
    phi_c = phi[np.minimum((centers * phi.size).astype(int), phi.size - 1)]
    mean_rate = t_b.size / days / 390.0
    ax.plot(centers * 6.5, phi_c * mean_rate, color=ORANGE, lw=2.0,
            label=r"$\varphi_\lambda(u)\times$ mean rate")
    ax.set_xlabel("hours into the session")
    ax.set_ylabel("events / minute")
    ax.set_title("the U-shape rides on the BASELINE only (kernels untouched,\n"
                 "so n is exactly preserved across the day) — corr = 0.994", fontsize=9.5)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    meta = r7.events.meta
    agg_t = np.asarray(meta["agg_trade_t"])
    agg_sz = np.asarray(meta["agg_trade_size"])
    keep = agg_t >= burn
    edges = np.arange(burn, t_end + 60.0, 60.0)
    vol, _ = np.histogram(agg_t[keep], bins=edges, weights=agg_sz[keep])
    d = vol - vol.mean()
    denom = float(d @ d)
    lags = np.arange(1, 31)
    acf = [float(d[:-k] @ d[k:]) / denom for k in lags]
    ax.bar(lags, acf, color=BLUE, width=0.8)
    ax.axhline(0, color="k", lw=0.8)
    se = 1.0 / np.sqrt(vol.size)
    ax.axhspan(-2 * se, 2 * se, color=GREY, alpha=0.3, label=r"$\pm2/\sqrt{N}$")
    # 帰属の分解 (実測): φ で正規化すると lag1 +0.35→+0.18、lag30 +0.18→+0.01。
    # つまり短ラグの半分が Hawkes カスケード、カーネル射程 (5 分) を超える裾は
    # ほぼ全て季節性 — どちらも S7 で注文流に入った成分で、対照の S6 は平坦。
    ax.set_xlabel("lag (minutes)")
    ax.set_ylabel("ACF of per-minute traded volume")
    ax.set_title(
        "volume clustering appears (ACF(1) = +0.34; flat in S6). φ-normalizing\n"
        "splits it: +0.18 at lag 1 is Hawkes cascades, the slow tail beyond the\n"
        "5-min kernel range is the seasonal U-shape (+0.01 residual at lag 30)",
        fontsize=9.5,
    )
    ax.legend(fontsize=8.5)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s7_seasonality.png", dpi=140)
    plt.close(fig)


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("S7/S6 を生成中 (120 日 x 390)...", flush=True)
    r7, cfg7, r6, cfg6 = _results()
    print("図 1/3: 分岐比 3 経路と強度再構成", flush=True)
    fig_three_way(r7, cfg7)
    print("図 2/3: クラスタリング (S6 対照)", flush=True)
    fig_clustering(r7, cfg7, r6, cfg6)
    print("図 3/3: 日内 U 字と出来高 ACF", flush=True)
    fig_seasonality(r7, cfg7)
    for name in ("s7_three_way", "s7_clustering", "s7_seasonality"):
        p = IMAGES / f"{name}.png"
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
