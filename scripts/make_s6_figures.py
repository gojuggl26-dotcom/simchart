"""S6 (ZI 板) の図を作る。

README に載せる 3 枚:

1. ``s6_book.png`` — デプスプロファイル・スプレッド分布・配置べき則
2. ``s6_decoupling.png`` — ミッドと p* の切断 (κ=0): 経路の重ね描き + リターン相関
3. ``s6_vr.png`` — ミッド VR の 2 レジーム (1/δ 以下の subdiffusion と日次の拡散)

ラベルは英語 (既存の慣習)。数値の正は results/S6/metrics.json。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simchart import Config, run
from simchart.types import EventType

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"
BLUE, ORANGE, GREY, GREEN = "#1f4e79", "#d1701e", "#8a8a8a", "#2e7d32"


def _result(n_days: int = 120):
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs" / "s6.yaml")
    cfg = cfg.replace(n_days=n_days, steps_per_day=390)
    return run(cfg), cfg


def fig_book(r, cfg) -> None:
    burn = cfg.book_burn_in_days * 23400.0
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    # デプスプロファイル
    ax = axes[0]
    b = r.book
    mask = b.t > burn
    prof_b = np.nanmean(np.where(np.isnan(b.bid_px[mask]), np.nan, b.bid_sz[mask]), axis=0)
    prof_a = np.nanmean(np.where(np.isnan(b.ask_px[mask]), np.nan, b.ask_sz[mask]), axis=0)
    lv = np.arange(1, prof_b.size + 1)
    ax.bar(lv - 0.2, prof_b, width=0.4, color=BLUE, label="bid")
    ax.bar(lv + 0.2, prof_a, width=0.4, color=ORANGE, label="ask")
    ax.set_xlabel("level (1 = best)")
    ax.set_ylabel("mean depth (lots)")
    ax.set_title(
        "depth peaks AWAY from best (front depletion by\nmarket orders — the ZI health check)",
        fontsize=10,
    )
    ax.legend(fontsize=9)

    # スプレッド分布
    ax = axes[1]
    ev = r.events
    bb = ev.meta["best_bid_tick"]
    ba = ev.meta["best_ask_tick"]
    ok = (bb >= 0) & (ba >= 0) & (ev.t > burn)
    spread = (ba[ok] - bb[ok]).astype(int)
    vals, counts = np.unique(spread, return_counts=True)
    keep = vals <= 15
    ax.bar(vals[keep], counts[keep] / counts.sum(), color=BLUE)
    ax.set_xlabel("spread (ticks)")
    ax.set_ylabel("probability")
    ax.set_title(
        f"small-tick regime: median {np.median(spread):.0f} ticks\n"
        f"(never zero or negative — no-cross invariant)",
        fontsize=10,
    )

    # 配置距離分布 (対数)
    ax = axes[2]
    lo_mask = ev.event_type == int(EventType.LIMIT_ADD)
    bb_prev = np.concatenate([[np.nan], bb[:-1].astype(float)])
    ba_prev = np.concatenate([[np.nan], ba[:-1].astype(float)])
    base_price = ev.meta["base_price"]
    ticks = np.round((ev.price[lo_mask] - base_price) / cfg.tick_size)
    delta = np.where(ev.side[lo_mask] > 0, bb_prev[lo_mask] - ticks, ticks - ba_prev[lo_mask])
    pos = delta[np.isfinite(delta) & (delta >= 1) & (delta <= 200)]
    edges = np.unique(np.round(np.geomspace(1, 200, 24)).astype(int))
    counts, _ = np.histogram(pos, bins=np.append(edges, 400))
    dens = counts / np.diff(np.append(edges, 400)) / pos.size
    m = counts > 20
    ax.loglog(edges[m] + cfg.book_place_offset, dens[m], "o", color=BLUE, ms=5,
              label="measured")
    xs = np.geomspace(edges[0] + 3, 200, 50)
    ax.loglog(xs, dens[m][0] * (xs / (edges[m][0] + 3.0)) ** (-(1 + cfg.book_mu_place)),
              "--", color=ORANGE, label=rf"spec slope $-(1+\mu)$, $\mu$={cfg.book_mu_place}")
    ax.set_xlabel(r"$\Delta + \Delta_0$ (ticks from own best)")
    ax.set_ylabel("placement density")
    ax.set_title("power-law placement with truncation\n(estimated μ within ±0.2 of spec)",
                 fontsize=10)
    ax.legend(fontsize=8.5)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Zero-intelligence book at κ=0 (Smith et al. 2003 baseline)", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGES / "s6_book.png", dpi=140)
    plt.close(fig)


def fig_decoupling(r, cfg) -> None:
    ev = r.events
    obs = r.observation
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    days = obs.t / 23400.0
    stride = 39  # 10 分ごと
    ax.plot(days[::stride], np.exp(obs.log_price[::stride]), color=BLUE, lw=0.8,
            label="ZI mid (observed price)")
    ax.plot(days[::stride], np.exp(r.price.log_p_star[::stride]), color=ORANGE, lw=0.8,
            label=r"latent $p^*$ (L2, frozen)")
    ax.set_xlabel("trading day")
    ax.set_ylabel("price")
    ax.set_title(
        "two unrelated worlds at κ=0 — the mid knows nothing about p*\n"
        "(p* is still interpolated at every event and recorded, §10)",
        fontsize=10,
    )
    ax.legend(fontsize=9)

    ax = axes[1]
    bb = ev.meta["best_bid_tick"]
    ba = ev.meta["best_ask_tick"]
    ok = (bb >= 0) & (ba >= 0)
    mid = 0.5 * (bb[ok] + ba[ok]).astype(float)
    ps = np.asarray(ev.meta["log_pstar"])[ok]
    st = 200
    dm = np.diff(mid[::st])
    dp = np.diff(ps[::st])
    good = (dm != 0) | (dp != 0)
    c = float(np.corrcoef(dm[good], dp[good])[0, 1])
    ax.scatter(dp[good], dm[good], s=4, alpha=0.3, color=GREY)
    ax.set_xlabel(r"$\Delta \log p^*$")
    ax.set_ylabel(r"$\Delta$ mid (ticks)")
    ax.set_title(
        f"corr of RETURNS = {c:+.4f} (S10 will move this)\n"
        "levels are never compared — sample corr of two random\nwalks is arcsine-distributed, not near 0",
        fontsize=10,
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s6_decoupling.png", dpi=140)
    plt.close(fig)


def fig_vr(r, cfg) -> None:
    from simchart.validation.scaling import variance_ratio

    obs = r.observation
    burn_days = int(cfg.book_burn_in_days)
    bars = obs.to_bars(60.0)
    vr_min = variance_ratio(bars.log_price[burn_days:], (2, 4, 8, 16, 32, 64))
    daily = obs.to_bars(obs.session_seconds).log_price_flat()
    vr_day = variance_ratio(daily[burn_days:], (2, 4, 8, 16, 32, 64))

    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    q_min = [row["q"] for row in vr_min["table"] if row.get("vr") is not None]
    v_min = [row["vr"] for row in vr_min["table"] if row.get("vr") is not None]
    q_day = [row["q"] for row in vr_day["table"] if row.get("vr") is not None]
    v_day = [row["vr"] for row in vr_day["table"] if row.get("vr") is not None]
    ax.semilogx([q * 1.0 for q in q_min], v_min, "o-", color=BLUE,
                label="minute bars (q in minutes)")
    ax.semilogx([q * 390.0 for q in q_day], v_day, "s-", color=GREEN,
                label="daily bars (q in days, x = minutes equivalent)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axvline(0.2 * 390, color=ORANGE, lw=1.5, ls="--",
               label=r"order lifetime $1/\delta$ = 0.2 day (94 min)")
    ax.set_xlabel("aggregation horizon (minutes, log)")
    ax.set_ylabel("variance ratio")
    ax.set_title(
        "the ZI book is a spring below the order lifetime (subdiffusion, VR→0.23)\n"
        "and diffusive above it (daily VR ≈ 1) — mean reversion here is physics, not a bug",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s6_vr.png", dpi=140)
    plt.close(fig)


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("S6 を生成中 (120 日 x 390)...", flush=True)
    r, cfg = _result()
    print("図 1/3: 板の構造", flush=True)
    fig_book(r, cfg)
    print("図 2/3: κ=0 の切断", flush=True)
    fig_decoupling(r, cfg)
    print("図 3/3: VR の 2 レジーム", flush=True)
    fig_vr(r, cfg)
    for name in ("s6_book", "s6_decoupling", "s6_vr"):
        p = IMAGES / f"{name}.png"
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
