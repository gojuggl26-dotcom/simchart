"""S10: 結合 (κ > 0) の測定群。

中心は 4 つ:
- **乖離 d = log p* − log p_obs の定常性** (AR(1) 半減期・SD)
- **伝達率 T(h)** = Var[Δ_h log p_obs] / Var[Δ_h log p*]
- **残差符号の γ** — 生成時バイアス E[ε|d] = tanh(κ·d/s) を引いた符号列で
  分割構造 (⑪) の保存を判定する。raw の C(ℓ) には情報チャネル (追跡ハーディング
  のこぶ、d 緩和スケール ℓ~30-1000) が重畳するのは結合の物理でありバグではない
  (S10a の解剖 — results/S10a/DECISION.md)
- **p* 追随** (日次レベル/リターン相関)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import na, num, ok

__all__ = [
    "gap_metrics",
    "transmission",
    "residual_sign_acf",
    "pstar_tracking",
    "vol_activity_link",
]

_SESSION = 23400.0


def gap_metrics(result, cfg) -> dict[str, Any]:
    """d = log p* − log p_obs の定常性: SD・1 分 AR(1)・半減期・平均。"""
    obs = result.observation
    step = float(obs.step_seconds)
    burn = int(cfg.book_burn_in_days * _SESSION / step)
    d = (result.price.log_p_star - obs.log_price)[burn:]
    if d.size < 10_000:
        return na(f"標本が足りません (n={d.size})")
    stride = max(1, int(round(60.0 / step)))
    dm = d[::stride]
    phi = float(np.corrcoef(dm[:-1], dm[1:])[0, 1])
    hl = float(-np.log(2.0) / np.log(phi)) if 0.0 < phi < 1.0 else None
    return ok(
        num(hl),
        halflife_min=num(hl),
        ar1_1min=num(phi),
        sd_bp=num(float(d.std() * 1e4)),
        mean_bp=num(float(d.mean() * 1e4)),
        stationary=bool(hl is not None),
        n_minutes=int(dm.size),
    )


def transmission(result, cfg, horizons_sec=(60.0, 600.0, 23400.0, 117000.0)) -> dict[str, Any]:
    """伝達率 T(h)。短 h は板ノイズで >1、長 h は追随で → 1 (指示書 §5.2)。"""
    obs = result.observation
    step = float(obs.step_seconds)
    burn = int(cfg.book_burn_in_days * _SESSION / step)
    lp = obs.log_price[burn:]
    ps = result.price.log_p_star[burn:]
    rows: dict[str, Any] = {}
    for h in horizons_sec:
        stride = int(round(h / step))
        if stride < 1 or lp.size // stride < 60:
            continue
        do = np.diff(lp[::stride])
        dp = np.diff(ps[::stride])
        if dp.var() <= 0:
            continue
        key = f"h{int(h)}s"
        rows[key] = num(float(do.var() / dp.var()))
    if not rows:
        return na("有効なホライズンがありません")
    t_daily = rows.get(f"h{int(_SESSION)}s")
    return ok(t_daily, T=rows, T_daily=t_daily)


def residual_sign_acf(result, cfg, fit_range=(2, 1000)) -> dict[str, Any]:
    """生成時バイアスを引いた残差符号の γ (⑪ 保存の判定計器) + raw の記録。"""
    from .memory import acf_powerlaw_fit

    ev = result.events
    meta = ev.meta if isinstance(ev.meta, dict) else {}
    s_arr = np.asarray(meta.get("agg_trade_side", np.empty(0)), dtype=np.float64)
    t = np.asarray(meta.get("agg_trade_t", np.empty(0)))
    mt = np.asarray(meta.get("agg_trade_meta", np.empty(0)), dtype=np.float64)
    keep = t >= cfg.book_burn_in_days * _SESSION
    s_k = s_arr[keep]
    if s_k.size < 20_000:
        return na(f"攻撃注文が足りません (n={s_k.size})")
    fit_raw = acf_powerlaw_fit(s_k, fit_range, max_lag=fit_range[1])
    d0 = s_k - s_k.mean()
    c1_raw = float(d0[:-1] @ d0[1:]) / float(d0 @ d0)
    kappa = float(cfg.kappa)
    if kappa <= 0:
        tr0 = (30, fit_range[1]) if fit_range[1] > 60 else fit_range
        fit_tail0 = acf_powerlaw_fit(s_k, tr0, max_lag=tr0[1])
        return ok(
            fit_raw.get("gamma"),
            gamma_raw=num(fit_raw.get("gamma")),
            gamma_resid=num(fit_raw.get("gamma")),
            gamma_resid_tail=num(fit_tail0.get("gamma")),
            c1_raw=num(c1_raw),
            note="κ=0: raw = 残差",
        )
    mo = meta.get("metaorders") or {}
    tick = float(meta.get("tick_size", cfg.tick_size))
    bp = float(meta.get("base_price", 0.0))
    obs = result.observation
    idx = np.clip(
        (np.asarray(mo["t_first"]) / obs.step_seconds).astype(np.int64),
        0, result.price.log_p_star.size - 1,
    )
    sgrid = np.exp(result.price.log_vol) * np.sqrt(
        float(cfg.kappa_tau_meta_sec) / (252.0 * _SESSION)
    )
    mu_meta = np.tanh(
        kappa
        * (result.price.log_p_star[idx] - np.log(bp + tick * np.asarray(mo["mid_first"])))
        / sgrid[idx]
    )
    mt_k = mt[keep].astype(np.int64)
    mu_row = np.where(mt_k >= 0, mu_meta[np.clip(mt_k, 0, mu_meta.size - 1)], 0.0)
    resid = s_k - mu_row
    fit_res = acf_powerlaw_fit(resid, fit_range, max_lag=fit_range[1])
    # ★テール窓 (30,1000): κ ハーディングの残滓は短ラグ (ℓ < 30) に集中し、
    # ℓ=2 からのフィットは初期減衰が寝て γ̂ を過小評価する (1000 日実測:
    # (2,1000) 0.50 → (30,1000) 0.61 = S8 の 0.614 を回復)。⑪ 保存の判定計器は
    # テール側 — run length の裾指数はテールの傾きが担う。
    tail_range = (30, fit_range[1]) if fit_range[1] > 60 else fit_range
    fit_tail = acf_powerlaw_fit(resid, tail_range, max_lag=tail_range[1])
    dr = resid - resid.mean()
    c1_res = float(dr[:-1] @ dr[1:]) / float(dr @ dr)
    return ok(
        fit_res.get("gamma"),
        gamma_resid=num(fit_res.get("gamma")),
        gamma_resid_r2=num(fit_res.get("r2")),
        gamma_resid_tail=num(fit_tail.get("gamma")),
        gamma_resid_tail_r2=num(fit_tail.get("r2")),
        tail_range=list(tail_range),
        c1_resid=num(c1_res),
        gamma_raw=num(fit_raw.get("gamma")),
        gamma_raw_r2=num(fit_raw.get("r2")),
        c1_raw=num(c1_raw),
        mean_abs_bias=num(float(np.mean(np.abs(mu_meta)))),
        n=int(s_k.size),
    )


def vol_activity_link(result, cfg) -> dict[str, Any]:
    """⑦ ボラ・出来高リンク (S10c): 日次 RV(p_obs) と日次出来高/スプレッドの相関。

    主計器は **log-log Pearson** (日次 RV は裾が重く (Hill≈3.4)、レベル相関は
    少数の鯨日に支配される — S8/S9 で繰り返し確認した計器ノイズと同型)。
    レベル相関・イベント数版・日内スプレッド曲線 (§7.4) も記録する。
    """
    from ..types import EventType

    obs = result.observation
    step = float(obs.step_seconds)
    spd = int(round(_SESSION / step))
    burn_d = int(cfg.book_burn_in_days)
    n_days = int(cfg.n_days)

    lp = obs.log_price
    stride = max(1, int(round(60.0 / step)))
    r1m = np.diff(lp[::stride])
    day_of_r = (np.arange(r1m.size) * stride + stride) // spd
    rv = np.bincount(day_of_r.astype(np.int64), weights=r1m**2, minlength=n_days)[:n_days]

    t_ev = np.asarray(result.events.t)
    etype = np.asarray(result.events.event_type)
    size = np.asarray(result.events.size)
    day_ev = (t_ev / _SESSION).astype(np.int64)
    is_tr = etype == int(EventType.TRADE)
    vol = np.bincount(day_ev[is_tr], weights=size[is_tr], minlength=n_days)[:n_days]
    n_ev = np.bincount(day_ev, minlength=n_days)[:n_days].astype(np.float64)

    ev_meta = result.events.meta if isinstance(result.events.meta, dict) else {}
    sp_daily = np.full(n_days, np.nan)
    sp_curve = None
    if "best_bid_tick" in ev_meta:
        bb = np.asarray(ev_meta["best_bid_tick"], dtype=np.float64)
        ba = np.asarray(ev_meta["best_ask_tick"], dtype=np.float64)
        ok_sp = (bb >= 0) & (ba >= 0)
        sp = (ba - bb)[ok_sp]
        d_sp = day_ev[ok_sp]
        s_sum = np.bincount(d_sp, weights=sp, minlength=n_days)[:n_days]
        s_cnt = np.bincount(d_sp, minlength=n_days)[:n_days]
        sp_daily = np.where(s_cnt > 0, s_sum / np.maximum(s_cnt, 1), np.nan)
        n_bins = 26
        u_bin = np.clip(
            ((t_ev[ok_sp] % _SESSION) / _SESSION * n_bins).astype(np.int64), 0, n_bins - 1
        )
        keep_b = d_sp >= burn_d
        sp_curve = (
            np.bincount(u_bin[keep_b], weights=sp[keep_b], minlength=n_bins)
            / np.maximum(np.bincount(u_bin[keep_b], minlength=n_bins), 1)
        )

    k = slice(burn_d, n_days)
    rv_k, vol_k, nev_k, spd_k = rv[k], vol[k], n_ev[k], sp_daily[k]
    good = (rv_k > 0) & (vol_k > 0) & np.isfinite(spd_k)
    if good.sum() < 30:
        return na(f"日数が足りません (n={int(good.sum())})")

    def _corr(a, b) -> float:
        return float(np.corrcoef(a[good], b[good])[0, 1])

    c_log = _corr(np.log(rv_k), np.log(vol_k))
    return ok(
        num(c_log),
        corr_rv_volume_log=num(c_log),
        corr_rv_volume_level=num(_corr(rv_k, vol_k)),
        corr_rv_nevents_log=num(_corr(np.log(rv_k), np.log(nev_k))),
        corr_rv_spread=num(_corr(np.log(rv_k), spd_k)),
        spread_mean_ticks=num(float(np.nanmean(spd_k))),
        spread_intraday_curve=(
            [float(x) for x in sp_curve] if sp_curve is not None else None
        ),
        n_days_used=int(good.sum()),
    )


def pstar_tracking(result, cfg) -> dict[str, Any]:
    """p* 追随: 日次レベル相関 (ゲート > 0.9) とリターン相関 (記録)。"""
    obs = result.observation
    step = float(obs.step_seconds)
    burn = int(cfg.book_burn_in_days * _SESSION / step)
    spd = int(round(_SESSION / step))
    lp_d = obs.log_price[burn::spd]
    ps_d = result.price.log_p_star[burn::spd]
    if lp_d.size < 60:
        return na(f"日数が足りません (n={lp_d.size})")
    c_lvl = float(np.corrcoef(lp_d, ps_d)[0, 1])
    c_ret = float(np.corrcoef(np.diff(lp_d), np.diff(ps_d))[0, 1])
    return ok(
        num(c_lvl),
        corr_daily_level=num(c_lvl),
        corr_daily_return=num(c_ret),
        n_days=int(lp_d.size),
    )