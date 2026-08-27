"""S9 (queue-reactive) の図を作る。

README に載せる 3 枚:

1. ``s9_state.png`` — 状態依存の実在: スプレッド別の板内配置率・ノイズ条件付き
                        戻り曲線 (S8 対照)・OBI ビン別の次ミッド変化 ((10))
2. ``s9_deficit.png`` — 赤字の片側縮小: 約定時間 VR 曲線 (S8 vs S9)・
                        G(ℓ) の比較・赤字台帳 (S8 → S9 → S10 目標)
3. ``s9_micro.png`` — η の 3 系列と帯・signature (ミッド vs 約定 — 構造分離)・
                        tick 距離デプス (S8 対照)

ラベルは英語 (既存の慣習)。数値の正は results/S9/metrics.json。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simchart import Config, run
from simchart.types import EventType
from simchart.validation.micro import estimate_eta, propagator_fit
from simchart.validation.suite import _qr_metrics

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
BLUE, ORANGE, GREY, GREEN, RED = "#1f4e79", "#d1701e", "#8a8a8a", "#2e7d32", "#b03030"
S = 23400.0


def _prep(r, cfg):
    ev = r.events
    burn = cfg.book_burn_in_days * S
    pm = np.asarray(ev.meta["agg_trade_prev_mid_tick"], dtype=np.float64)
    t = np.asarray(ev.meta["agg_trade_t"])
    keep = (t >= burn) & np.isfinite(pm)
    lp = np.log(ev.meta["base_price"] + ev.meta["tick_size"] * pm[keep])
    sd = np.asarray(ev.meta["agg_trade_side"], dtype=np.float64)[keep]
    sz = np.asarray(ev.meta["agg_trade_size"], dtype=np.float64)[keep]
    mt = np.asarray(ev.meta["agg_trade_meta"], dtype=np.float64)[keep]
    return ev, burn, pm[keep], lp, sd, sz, mt


def fig_state(r9, cfg9, r8, cfg8, m9) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    # 板内配置率 vs スプレッド
    ax = axes[0]
    rates9 = m9["state_diag"]["inspread_rate_by_spread"]
    from simchart.validation.suite import _qr_metrics as _qm

    labels = {"s2_3": "2", "s3_5": "3-4", "s5_9": "5-8", "s9_30": "9+"}
    xs = [labels.get(k, k) for k in rates9]
    ax.bar(xs, [rates9[k] for k in rates9], color=BLUE, width=0.6,
           label="S9 (spread-dependent)")
    ax.axhline(0.14, color=GREY, ls="--", lw=1.4,
               label="S6-S8 baseline (state-blind ~0.14)")
    ax.set_xlabel("spread just before placement (ticks)")
    ax.set_ylabel("fraction placed inside the spread")
    ax.set_title("the decision layer reads the state: a stressed (wide) spread\n"
                 "pulls limit orders inside — the mean-reversion engine (§5)",
                 fontsize=9.5)
    ax.legend(fontsize=8.5)

    # ノイズ条件付き戻り
    ax = axes[1]
    for (r, cfg, color, label) in ((r8, cfg8, GREY, "S8"), (r9, cfg9, BLUE, "S9")):
        ev, burn, pmk, lp, sd, sz, mt = _prep(r, cfg)
        idxn = np.flatnonzero(mt < 0)
        hs = np.array([1, 2, 3, 5, 10, 20, 50, 100])
        prof = []
        for h in hs:
            sel = idxn[idxn + h < pmk.size]
            prof.append(float(np.mean(sd[sel] * (pmk[sel + h] - pmk[sel]))))
        ax.plot(hs, prof, "o-", color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("trades after a NOISE trade (log)")
    ax.set_ylabel("mean signed mid move (ticks)")
    ax.set_title("conditioning on noise trades isolates the book's pull-back\n"
                 "(unconditional curves rise — future same-sign flow)", fontsize=9.5)
    ax.legend(fontsize=9)

    # OBI ビン
    ax = axes[2]
    ev, burn, pmk, lp, sd, sz, mt = _prep(r9, cfg9)
    tr = r9.events.event_type == int(EventType.TRADE)
    tr_t = r9.events.t[tr]
    starts = np.flatnonzero(np.concatenate([[True], np.diff(tr_t) > 0]))
    trade_idx = np.flatnonzero(tr)
    pre_idx = np.maximum(trade_idx[starts] - 2, 0)
    db = np.asarray(r9.events.meta["depth_bid"])[pre_idx].astype(np.float64)
    da = np.asarray(r9.events.meta["depth_ask"])[pre_idx].astype(np.float64)
    imb_full = (db - da) / np.maximum(db + da, 1e-9)
    t_agg = np.asarray(r9.events.meta["agg_trade_t"])
    keep = (t_agg >= burn) & np.isfinite(
        np.asarray(r9.events.meta["agg_trade_prev_mid_tick"], dtype=np.float64)
    )
    imb = imb_full[keep]
    pm_k = np.asarray(r9.events.meta["agg_trade_prev_mid_tick"], dtype=np.float64)[keep]
    dm = pm_k[1:] - pm_k[:-1]
    ik = imb[:-1]
    edges = np.linspace(-1, 1, 11)
    centers, means, ses = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ik >= a) & (ik < b) & np.isfinite(dm)
        if m.sum() > 200:
            centers.append(0.5 * (a + b))
            means.append(float(dm[m].mean()))
            ses.append(float(dm[m].std() / np.sqrt(m.sum())))
    ax.errorbar(centers, means, yerr=ses, fmt="o-", color=BLUE)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("order book imbalance I (10-tick depth)")
    ax.set_ylabel("mean next mid change (ticks)")
    ax.set_title("stylized fact #10 emerges MECHANICALLY (no sign bias):\n"
                 "thin queues die first, so imbalance predicts the next move\n"
                 f"corr = {m9['obi']['corr_h1']:+.3f}", fontsize=9.5)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s9_state.png", dpi=140)
    plt.close(fig)


def fig_deficit(r9, cfg9, r8, cfg8) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    # VR 曲線
    ax = axes[0]
    ns = np.array([3, 10, 30, 100, 300, 1000, 3000])
    for (r, cfg, color, label) in ((r8, cfg8, GREY, "S8"), (r9, cfg9, BLUE, "S9")):
        ev, burn, pmk, lp, sd, sz, mt = _prep(r, cfg)
        v1 = np.diff(lp).var()
        vr = [float((lp[n:] - lp[:-n]).var() / (n * v1)) for n in ns]
        ax.semilogx(ns, vr, "o-", color=color, label=label)
    ax.axhline(1.0, color=GREEN, lw=1.2, ls="--", label="S10 target (κ anchor)")
    ax.set_xlabel("aggregation window (trades, log)")
    ax.set_ylabel("variance ratio (trade time)")
    ax.set_title("one side of the deficit closes: spread relaxation trims the\n"
                 "mid horizons, but without a displacement anchor the long end\n"
                 "stays superdiffusive — that anchor is S10's κ", fontsize=9.5)
    ax.legend(fontsize=8.5)

    # G(l)
    ax = axes[1]
    for (r, cfg, color, label) in ((r8, cfg8, GREY, "S8"), (r9, cfg9, BLUE, "S9")):
        ev, burn, pmk, lp, sd, sz, mt = _prep(r, cfg)
        p = propagator_fit(sd, sz, lp, 200, (5, 150))
        gv = np.array([np.nan if v is None else v for v in p["propagator"]]) * 1e4
        lags = np.arange(1, gv.size + 1)
        show = lags <= 170
        ax.plot(lags[show], gv[show], color=color, lw=1.4,
                label=f"{label} (β̂={p['beta']:+.2f})")
    gtar = gv[4] * (np.arange(1, 171) / 5.0) ** (-(1 - 0.58) / 2)
    ax.plot(np.arange(5, 171), gtar[4:], "--", color=ORANGE, lw=1.4,
            label=r"efficient $\ell^{-(1-\gamma)/2}$")
    ax.set_xlabel("lag (aggressor orders)")
    ax.set_ylabel("G (bp)")
    ax.set_title("the propagator bends slightly toward decay (β̂ −0.25 → −0.21)\n"
                 "— direction is right, magnitude needs the S10 anchor", fontsize=9.5)
    ax.legend(fontsize=8.5)

    # 赤字台帳
    ax = axes[2]
    cats = ["VR(1000)", "β", "size slope"]
    s8v = [3.21, -0.252, 0.907]
    s9v = [2.68, -0.209, 0.923]
    s10t = [1.0, 0.21, 0.55]
    x = np.arange(3)
    w = 0.27
    ax.bar(x - w, s8v, w, color=GREY, label="S8")
    ax.bar(x, s9v, w, color=BLUE, label="S9")
    ax.bar(x + w, s10t, w, color=GREEN, alpha=0.6, label="S10 target")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title("the impact-deficit ledger: S9 moves VR and β in the right\n"
                 "direction; the size slope is structurally stuck at ~0.92\n"
                 "(spread relaxation is too fast to discriminate whale sizes)",
                 fontsize=9.5)
    ax.legend(fontsize=8.5)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s9_deficit.png", dpi=140)
    plt.close(fig)


def fig_micro(r9, cfg9, r8, cfg8, m9) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    # η の 3 系列
    ax = axes[0]
    names = ["last px\n(per aggressor)", "trade rows", "mid"]
    vals9 = [m9["eta_trade"]["eta"], m9["eta_trade_rows"]["eta"], m9["eta_mid"]["eta"]]
    ax.bar(names, vals9, color=[BLUE, BLUE, GREY], width=0.55)
    ax.axhspan(0.05, 0.35, color=GREEN, alpha=0.15, label="empirical band [0.05, 0.35]")
    ax.axhline(0.5, color="k", lw=0.8, ls=":", label="η = 0.5 (no bounce)")
    ax.set_ylabel(r"effective $\eta$ (continuation / 2·alternation)")
    ax.set_title("η is a TRANSACTION-price concept (Robert–Rosenbaum):\n"
                 "trade series sit in the empirical band; the mid (grey) is a\n"
                 "different, smoother object — don't mix the series", fontsize=9.5)
    ax.legend(fontsize=8)

    # signature: mid vs trades
    ax = axes[1]
    obs = r9.observation
    step = obs.step_seconds
    burn = cfg9.book_burn_in_days * S
    lp_o = obs.log_price[int(burn / step):]
    secs = [1, 5, 15, 60, 300, 900]
    rv_mid = []
    for sec in secs:
        stride = int(round(sec / step))
        rr = np.diff(lp_o[::stride])
        rv_mid.append(float(rr.var() / sec))
    ax.loglog(secs, np.array(rv_mid) * 1e9, "o-", color=BLUE, label="mid")
    ev = r9.events
    agg_t = np.asarray(ev.meta["agg_trade_t"])
    agg_px = np.asarray(ev.meta["agg_trade_log_vwap"])
    keep = agg_t >= burn
    rv_tr = []
    for sec in secs:
        grid = np.arange(burn, cfg9.n_days * S, sec)
        idx = np.searchsorted(agg_t[keep], grid, side="right") - 1
        v = idx >= 0
        rr = np.diff(agg_px[keep][idx[v]])
        rv_tr.append(float(rr.var() / sec))
    ax.loglog(secs, np.array(rv_tr) * 1e9, "s-", color=ORANGE, label="trade VWAP")
    ax.set_xlabel("sampling interval (s, log)")
    ax.set_ylabel("RV per second ×1e9 (log)")
    ax.set_title("signature plots split: trades fall (bounce 合格 empirical shape),\n"
                 "the mid RISES at 300–900s — the residual superdiffusion that\n"
                 "§9.2 forbids us to fill; decreasing-mid is an S10 target",
                 fontsize=9.5)
    ax.legend(fontsize=8.5)

    # tick 距離デプス
    ax = axes[2]
    prof9 = m9["depth_tick_profile"]["profile"]
    from simchart.validation.suite import _qr_metrics as _qm2
    xs9 = [int(k) for k in prof9]
    ax.plot(xs9, [prof9[k] for k in prof9], "o-", color=BLUE, label="S9")
    # S8 側 (同関数を直接)
    b8 = r8.book
    keep8 = b8.t > burn
    tick = r8.events.meta["tick_size"]
    prof = {}
    for px_side, sz_side in ((b8.bid_px, b8.bid_sz), (b8.ask_px, b8.ask_sz)):
        px_k = px_side[keep8]
        sz_k = sz_side[keep8]
        best = px_k[:, 0:1]
        d_ticks = np.abs(np.round((px_k - best) / tick)) + 1
        for dd in range(1, 31):
            m = d_ticks == dd
            if int(m.sum()) > 200:
                prof.setdefault(dd, []).append(float(np.nanmean(sz_k[m])))
    avg8 = {d: float(np.mean(v)) for d, v in prof.items()}
    ax.plot(sorted(avg8), [avg8[d] for d in sorted(avg8)], "s--", color=GREY,
            label="S8")
    ax.set_xlabel("distance from best (ticks)")
    ax.set_ylabel("mean depth (lots)")
    ax.set_title("the hump (#20) stands at 3–4 ticks — front depletion plus\n"
                 "in-spread placement shape it (the cancel tilt was NOT needed\n"
                 "in the small-tick regime, and hurt when tried)", fontsize=9.5)
    ax.legend(fontsize=8.5)

    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(IMAGES / "s9_micro.png", dpi=140)
    plt.close(fig)


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("S9/S8 を生成中 (250 日)...", flush=True)
    cfg9 = Config.load(ROOT / "configs" / "s9.yaml")
    cfg8 = Config.load(ROOT / "configs" / "s8.yaml")
    r9 = run(cfg9)
    r8 = run(cfg8)
    m9 = _qr_metrics(r9, cfg9)
    print("図 1/3: 状態依存", flush=True)
    fig_state(r9, cfg9, r8, cfg8, m9)
    print("図 2/3: 赤字の片側縮小", flush=True)
    fig_deficit(r9, cfg9, r8, cfg8)
    print("図 3/3: η・signature・ハンプ", flush=True)
    fig_micro(r9, cfg9, r8, cfg8, m9)
    for name in ("s9_state", "s9_deficit", "s9_micro"):
        p = IMAGES / f"{name}.png"
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
