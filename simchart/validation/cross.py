"""資産間 (非同期観測) の検証。

S0 は単一資産なので ``not_applicable`` になる。S13 で多資産を入れたときに、
共通因子から出るはずの相関が非同期観測でも正しく測れているかを確認するために
Hayashi-Yoshida 推定量を使う。等間隔に揃えてから相関を取ると Epps 効果
(細かい粒度ほど相関が消える見かけの現象) を自分で作り込んでしまうため、
最初から同期化しない推定量で測る。

S13 で追加した測定器 :

- :func:`epps_curve` サンプリング間隔別の相関 (前値サンプル約定値)
- :func:`lead_lag_profile` 交差相関のピーク位置 (創発リードラグ §6)
- :func:`conditional_correlation` 危機時 vs 平常時の日次相関 (§7.2)
- :func:`vol_correlation_by_horizon` ボラ相関の水平依存性 (§4.5)
- :func:`factor_decomposition_check` β の実測 (記録)
- :func:`theoretical_daily_corr` 因子構造の理論相関 (ゲート daily_corr_matches の基準)
- :func:`cross_asset_metrics` 上記を束ねて MultiAssetResult から計算する
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .base import na, num, ok

__all__ = [
    "hayashi_yoshida",
    "hayashi_yoshida_lead_lag",
    "epps_curve",
    "lead_lag_profile",
    "conditional_correlation",
    "vol_correlation_by_horizon",
    "factor_decomposition_check",
    "theoretical_daily_corr",
    "crisis_day_mask",
    "cross_asset_metrics",
]

_SINGLE_ASSET = (
    "資産が 1 本しかないため測定できません。多資産は S13 で導入します。"
)


def _as_series(obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """``Observation`` / ``(t, log_price)`` / ``dict`` を ``(t, log_price)`` に揃える。"""
    if obj is None:
        return None
    if hasattr(obj, "t") and hasattr(obj, "log_price"):
        t, p = obj.t, obj.log_price
    elif isinstance(obj, dict) and "t" in obj:
        t, p = obj["t"], obj.get("log_price", obj.get("p"))
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        t, p = obj
    else:
        return None
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    if t_arr.size < 2 or t_arr.size != p_arr.size:
        return None
    return t_arr, p_arr


def hayashi_yoshida(asset1: Any, asset2: Any, lag: float = 0.0) -> dict:
    """Hayashi-Yoshida の共分散・相関推定量 (非同期観測でも不偏)。

    区間 ``(t1_{i-1}, t1_i]`` と ``(t2_{j-1}, t2_j]`` が重なる収益率の積をすべて
    足し上げる。区間の重なり判定は ``t1_{i-1} < t2_j`` かつ ``t2_{j-1} < t1_i``。

    Parameters
    ----------
    asset1, asset2:
        :class:`~simchart.types.Observation`、``(t, log_price)`` のタプル、または
        同名キーを持つ辞書。
    lag:
        資産 2 の時刻に加えるシフト (秒)。正なら資産 2 を遅らせる = 資産 1 が
        先行しているかを見る。リード・ラグ曲線を描くのに使う。
    """
    s1 = _as_series(asset1)
    s2 = _as_series(asset2)
    if s1 is None or s2 is None:
        return na(_SINGLE_ASSET)

    t1, p1 = s1
    t2, p2 = s2
    t2 = t2 + float(lag)

    d1 = np.diff(p1)
    rv1 = float(np.sum(d1**2))
    rv2 = float(np.sum(np.diff(p2) ** 2))
    if rv1 <= 0 or rv2 <= 0:
        return na("いずれかの資産の実現分散が 0 です")

    # 区間 i (i=1..n1) に重なる区間 j の範囲を二分探索で求める。
    # 条件 b[j] > a[i-1] かつ b[j-1] < a[i] (a=t1, b=t2)
    j_lo = np.searchsorted(t2, t1[:-1], side="right")  # 最初の j で b[j] > a[i-1]
    j_lo = np.maximum(j_lo, 1)
    j_hi = np.searchsorted(t2, t1[1:], side="left")  # 最後の j で b[j-1] < a[i]
    j_hi = np.minimum(j_hi, t2.size - 1)

    valid = j_lo <= j_hi
    if not np.any(valid):
        return na("重なる観測区間がありません")
    contribution = p2[j_hi[valid]] - p2[j_lo[valid] - 1]
    cov = float(np.sum(d1[valid] * contribution))
    n_pairs = int(np.sum(j_hi[valid] - j_lo[valid] + 1))

    corr = cov / np.sqrt(rv1 * rv2)
    return ok(
        num(corr),
        covariance=num(cov),
        correlation=num(corr),
        realized_var_1=num(rv1),
        realized_var_2=num(rv2),
        n_intervals_1=int(d1.size),
        n_intervals_2=int(t2.size - 1),
        n_overlapping_pairs=n_pairs,
        lag=float(lag),
    )


def hayashi_yoshida_lead_lag(asset1: Any, asset2: Any, lags: Sequence[float]) -> dict:
    """複数のシフトで HY 相関を計算し、リード・ラグ曲線を返す。

    最大相関を与えるシフトが 0 から有意に離れていれば、どちらかが先行している。
    S13 で共通因子を入れたとき、意図しない先行関係 (実装上の時間ずれ) が
    生じていないかの確認に使う。
    """
    rows = []
    for lag in lags:
        res = hayashi_yoshida(asset1, asset2, lag=float(lag))
        if res["status"] != "ok":
            return res
        rows.append({"lag": float(lag), "correlation": res["correlation"]})
    if not rows:
        return na("ラグが指定されていません")
    best = max(rows, key=lambda row: abs(row["correlation"] or 0.0))
    return ok(
        best["correlation"],
        best_lag=best["lag"],
        best_correlation=best["correlation"],
        correlation_at_zero=next(
            (row["correlation"] for row in rows if row["lag"] == 0.0), None
        ),
        table=rows,
    )


# ---------------------------------------------------------------------------
# S13: 測定器
# ---------------------------------------------------------------------------
def _prev_tick_bars(
    t_ev: np.ndarray,
    p_ev: np.ndarray,
    interval_sec: float,
    session_seconds: float,
    n_days: int,
    t0: float,
) -> np.ndarray:
    """約定系列を前値サンプルでバー化する (セッション内リターン、(n_days, n_bars))。

    Epps 効果の古典的な測定対象 = 取引時刻でしか更新されない価格の同期
    サンプリング。板ミッドのグリッド系列を使うと非同期性が減って Epps が
    小さく見えるため、約定 (VWAP) 系列で測る。
    """
    n_bars = int(session_seconds // interval_sec)
    offsets = np.arange(n_bars + 1, dtype=np.float64) * interval_sec
    day_starts = t0 + np.arange(n_days, dtype=np.float64) * session_seconds
    query = (day_starts[:, None] + offsets[None, :]).ravel()
    idx = np.searchsorted(t_ev, query, side="right") - 1
    invalid = idx < 0  # 最初の約定より前 — 前値が定義できない
    np.clip(idx, 0, t_ev.size - 1, out=idx)
    prices = p_ev[idx]
    prices[invalid] = np.nan  # NaN は diff で隣接リターンへ伝播し、相関から外れる
    return np.diff(prices.reshape(n_days, n_bars + 1), axis=1)


def epps_curve(
    trades_i: tuple[np.ndarray, np.ndarray],
    trades_j: tuple[np.ndarray, np.ndarray],
    intervals_sec: Sequence[float],
    session_seconds: float,
    n_days: int,
    t0: float,
) -> dict:
    """サンプリング間隔別の相関 (Epps 曲線、設計要件)。

    検証は出たかではなく比率で行う (§10): 1 分相関 / 日次相関 < 0.7。
    日次相関も同じ約定系列の前値サンプルで測る (基準を揃える)。
    """
    ti, pi = trades_i
    tj, pj = trades_j
    if ti.size < 100 or tj.size < 100:
        return na("約定が少なすぎます (前値サンプルが成立しない)")
    table = []
    corr_by_interval: dict[float, float | None] = {}
    for dt in list(intervals_sec) + [session_seconds]:
        ri = _prev_tick_bars(ti, pi, float(dt), session_seconds, n_days, t0).ravel()
        rj = _prev_tick_bars(tj, pj, float(dt), session_seconds, n_days, t0).ravel()
        m = np.isfinite(ri) & np.isfinite(rj)
        if m.sum() < 30:
            corr_by_interval[float(dt)] = None
            continue
        c = float(np.corrcoef(ri[m], rj[m])[0, 1])
        corr_by_interval[float(dt)] = c
        table.append({"interval_sec": float(dt), "correlation": c, "n": int(m.sum())})
    c_daily = corr_by_interval.get(float(session_seconds))
    c_1min = corr_by_interval.get(60.0)
    ratio = (c_1min / c_daily) if (c_1min is not None and c_daily) else None
    # 単調性: 間隔の増加とともに相関が増加・飽和 (§10 epps_monotone)。
    vals = [row["correlation"] for row in table]
    steps = [b - a for a, b in zip(vals, vals[1:])]
    monotone_ok = bool(vals and all(s >= -0.03 for s in steps))
    return ok(
        ratio,
        ratio_1min_over_daily=ratio,
        correlation_1min=c_1min,
        correlation_daily=c_daily,
        monotone_ok=monotone_ok,
        min_step=min(steps) if steps else None,
        table=table,
    )


def lead_lag_profile(
    r_i: np.ndarray, r_j: np.ndarray, max_lag: int
) -> dict:
    """バーリターンの交差相関プロファイル (2 次元 (n_days, n_bars) 入力)。

    ``corr(r_i(t), r_j(t+ℓ))`` を ℓ = −max_lag..+max_lag で計算する。
    ℓ > 0 側が厚ければ i が先行 (j が遅れて追随)。セッション境界はまたがない。
    """
    if r_i.shape != r_j.shape or r_i.ndim != 2:
        return na("バー行列の形状が一致しません")
    n_days, n_bars = r_i.shape
    if n_bars <= max_lag + 2:
        return na("バー数がラグに対して不足しています")
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = r_i[:, : n_bars - lag].ravel()
            b = r_j[:, lag:].ravel()
        else:
            a = r_i[:, -lag:].ravel()
            b = r_j[:, : n_bars + lag].ravel()
        rows.append({"lag": lag, "correlation": float(np.corrcoef(a, b)[0, 1])})
    corr0 = next(r["correlation"] for r in rows if r["lag"] == 0)
    best = max(rows, key=lambda r: r["correlation"])
    pos = sum(r["correlation"] for r in rows if r["lag"] > 0)
    neg = sum(r["correlation"] for r in rows if r["lag"] < 0)
    return ok(
        float(best["lag"]),
        peak_lag=int(best["lag"]),
        peak_correlation=best["correlation"],
        correlation_at_zero=corr0,
        asymmetry_i_leads=float(pos - neg),
        sum_pos_lags=float(pos),
        sum_neg_lags=float(neg),
        table=rows,
    )


def crisis_day_mask(
    episodes: Sequence, step_sec: float, n_days: int,
    seconds_per_day: float = 23400.0,
) -> np.ndarray:
    """危機エピソード (スナップショット区間) を日マスクへ変換する (±1 日パッド)。"""
    mask = np.zeros(n_days, dtype=bool)
    for a, b in episodes or []:
        d0 = int(a * step_sec / seconds_per_day)
        d1 = int(b * step_sec / seconds_per_day)
        mask[max(d0 - 1, 0) : min(d1 + 2, n_days)] = True
    return mask


def conditional_correlation(
    daily_i: np.ndarray,
    daily_j: np.ndarray,
    crisis_mask: np.ndarray,
) -> dict:
    """危機日 (ペア和集合) とそれ以外の日次リターン相関 (§7.2)。"""
    n = min(daily_i.size, daily_j.size, crisis_mask.size)
    di, dj, m = daily_i[:n], daily_j[:n], crisis_mask[:n]
    n_crisis = int(m.sum())
    if n_crisis < 8 or (n - n_crisis) < 8:
        return na(f"危機日 {n_crisis} / 平常日 {n - n_crisis} が不足しています")
    c_c = float(np.corrcoef(di[m], dj[m])[0, 1])
    c_n = float(np.corrcoef(di[~m], dj[~m])[0, 1])
    return ok(
        c_c - c_n,
        corr_crisis=c_c,
        corr_normal=c_n,
        increase=c_c - c_n,
        n_crisis_days=n_crisis,
        n_normal_days=int(n - n_crisis),
    )


def vol_correlation_by_horizon(
    rv_i: np.ndarray, rv_j: np.ndarray, horizons_days: Sequence[int] = (1, 5, 20)
) -> dict:
    """log RV の水平別相関 (§4.5 — 遅い成分の共有なら長期ほど高い)。"""
    n = min(rv_i.size, rv_j.size)
    li = np.log(np.maximum(rv_i[:n], 1e-300))
    lj = np.log(np.maximum(rv_j[:n], 1e-300))
    table = []
    by_h: dict[int, float | None] = {}
    for h in horizons_days:
        k = n // int(h)
        if k < 12:
            by_h[int(h)] = None
            continue
        bi = li[: k * h].reshape(k, h).mean(axis=1)
        bj = lj[: k * h].reshape(k, h).mean(axis=1)
        c = float(np.corrcoef(bi, bj)[0, 1])
        by_h[int(h)] = c
        table.append({"horizon_days": int(h), "correlation": c, "n_blocks": int(k)})
    hs = [int(h) for h in horizons_days if by_h.get(int(h)) is not None]
    increasing = (
        bool(by_h[hs[-1]] > by_h[hs[0]]) if len(hs) >= 2 else None
    )
    return ok(
        by_h.get(hs[-1]) if hs else None,
        increasing_with_horizon=increasing,
        corr_short=by_h.get(hs[0]) if hs else None,
        corr_long=by_h.get(hs[-1]) if hs else None,
        table=table,
    )


def factor_decomposition_check(
    daily_latent: Sequence[np.ndarray], factor_daily: np.ndarray, betas: Sequence[float]
) -> dict:
    """β の実測 (§9 — 記録のみ)。潜在日次リターンを既知の共通因子日次集計に回帰する。

    シミュレータ内部では z_F が既知なので、β̂_i の相対比が設計比と整合するかを
    見る (絶対水準はボラ水準の重みで β_i·E[σ_i]·√dt 倍にスケールする)。
    """
    slopes = []
    for r in daily_latent:
        n = min(r.size, factor_daily.size)
        f = factor_daily[:n]
        slopes.append(float(np.dot(r[:n] - r[:n].mean(), f - f.mean()) / np.dot(f - f.mean(), f - f.mean())))
    base = slopes[0] if slopes and slopes[0] != 0 else None
    ratios = [s / base if base else None for s in slopes]
    beta_ratios = [float(b) / float(betas[0]) if betas[0] else None for b in betas]
    return ok(
        None,
        slopes=slopes,
        slope_ratios=ratios,
        design_beta_ratios=beta_ratios,
        betas_design=[float(b) for b in betas],
    )


def theoretical_daily_corr(config) -> dict:
    """因子構造の理論日次相関 (ゲート daily_corr_matches の基準、§4.5/§10)。

    潜在日次リターンについて:

        corr_ij = β_i β_j · D · (1−j_s) + s_J · j_s

    - D = 固有ボラ成分の対数正規希釈 E[σ_iσ_j]/√(E σ_i² E σ_j²)
        = (E[√M]²)^{k−k_c} · exp(−v_idio) (共有成分は分子分母で相殺)
        v_idio = (1−f_c)·var_slow + var_rough (離散実分散)
    - j_s = ジャンプの QV シェア、s_J = 共通強度シェア (共通ジャンプは全資産
      同一サイズ = 相関 1 の成分)

    これは拡散相関の希釈とコジャンプの寄与だけの一次理論。
    日次集計のボラ加重の高次項は含まない (実測との差が ±0.05 に入るかが
    ゲートそのもの)。
    """
    from ..layers.l2_price import (
        rough_discrete_stationary_variance,
        solve_eta_rough,
        solve_m0,
    )

    m0 = solve_m0(config.msm_k, config.vol_var_target_msm)
    e_sqrt_m = 0.5 * (math.sqrt(m0) + math.sqrt(2.0 - m0))
    k_idio = config.msm_k - config.msm_k_common
    d_msm = (e_sqrt_m * e_sqrt_m) ** k_idio
    v_idio = (1.0 - config.ou_common_share) * config.vol_var_target_slow
    var_rough = 0.0
    if config.enable_rough:
        theta_r = math.log(2.0) / config.rough_half_life_days
        dt_days = config.rough_grid_seconds / config.seconds_per_day
        eta = solve_eta_rough(config.rough_hurst, theta_r, config.vol_var_target_rough)
        var_rough = rough_discrete_stationary_variance(
            config.rough_hurst, theta_r, dt_days, eta
        )
    dilution = d_msm * math.exp(-(v_idio + var_rough))
    j_s = config.jump_qv_share_target if config.enable_jump else 0.0
    s_j = config.jump_common_share
    betas = config.factor_betas
    pairs = {}
    for i in range(len(betas)):
        for j in range(i + 1, len(betas)):
            c = betas[i] * betas[j] * dilution * (1.0 - j_s) + s_j * j_s
            pairs[f"{i}-{j}"] = c
    return ok(
        None,
        pairs=pairs,
        dilution_lognormal=dilution,
        d_msm_idio=d_msm,
        v_idio_gauss=v_idio + var_rough,
        jump_qv_share=j_s,
        jump_common_share=s_j,
    )


def cross_asset_metrics(multi, config) -> dict:
    """MultiAssetResult からクロス資産の全測定を計算する (設計要件/§10)。"""
    pl = multi.payloads
    n_assets = len(pl)
    if n_assets < 2:
        return na("資産が 1 本しかありません")
    session = pl[0].session_seconds
    theory = theoretical_daily_corr(config)

    pairs: dict[str, Any] = {}
    hy_errors: list[float] = []
    epps_ratios: list[float] = []
    epps_monotone: list[bool] = []
    corr_errs_latent: list[float] = []
    vol_corrs: list[float] = []
    vol_horizon_increasing: list[bool] = []
    cc_increases: list[float] = []
    cc_breadth_increases: list[float] = []
    cc_bigf_increases: list[float] = []

    # 条件付けマスク (資産横断で 1 回だけ作る):
    #  - breadth: ≥2 資産が同時に危機 (検出器ベース = リターン選択バイアスなし。
    #    実データの市場全体の危機期間の観測可能な対応物)
    #  - big_f: |z_F| 日次集計の上位 10% (潜在 — §7.1 の機構実在の記録用。
    #    リターン自身での条件付けは楕円切断の機械的相関上昇を作るため使わない)
    n_dm = min(min(p.daily_ret_obs.size, p.n_days) for p in pl)
    masks_all = [
        crisis_day_mask(p.crisis_episodes, p.crisis_step_sec, n_dm) for p in pl
    ]
    breadth_mask = np.sum(np.stack(masks_all), axis=0) >= 2
    fd = np.abs(np.asarray(multi.factor_daily)[:n_dm])
    bigf_mask = fd > np.quantile(fd, 0.9) if fd.size >= n_dm else None

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            key = f"{i}-{j}"
            pi, pj = pl[i], pl[j]
            # --- 真の相関 (潜在 1 分リターンの実現相関) ---
            r1i = np.diff(pi.pstar_1min)
            r1j = np.diff(pj.pstar_1min)
            n1 = min(r1i.size, r1j.size)
            true_corr = float(np.corrcoef(r1i[:n1], r1j[:n1])[0, 1])
            # --- HY (p* を各資産の実約定時刻でサンプル — 推定量+非同期性の検定) ---
            hy_rows = {}
            for thin, label in ((1, "full"), (2, "half"), (4, "quarter")):
                hy = hayashi_yoshida(
                    (pi.trade_t[::thin], pi.pstar_at_trades[::thin]),
                    (pj.trade_t[::thin], pj.pstar_at_trades[::thin]),
                )
                hy_rows[label] = hy
                if hy.get("status") == "ok":
                    hy_errors.append(abs(hy["correlation"] - true_corr))
            # --- HY (板の約定値 — マイクロ構造減衰込み、記録) ---
            hy_book = hayashi_yoshida(
                (pi.trade_t, pi.trade_log_vwap), (pj.trade_t, pj.trade_log_vwap)
            )
            # --- Epps 曲線 (約定値の前値サンプル) ---
            epps = epps_curve(
                (pi.trade_t, pi.trade_log_vwap),
                (pj.trade_t, pj.trade_log_vwap),
                (60.0, 300.0, 900.0, 1800.0),
                session,
                min(pi.n_days, pj.n_days),
                pi.obs_t0,
            )
            if epps.get("status") == "ok" and epps.get("ratio_1min_over_daily") is not None:
                epps_ratios.append(epps["ratio_1min_over_daily"])
                epps_monotone.append(bool(epps["monotone_ok"]))
            # --- 日次相関 (潜在=理論の直接対象 / 観測=板経由の記録) ---
            n_d = min(pi.daily_ret_latent.size, pj.daily_ret_latent.size)
            c_lat = float(
                np.corrcoef(pi.daily_ret_latent[:n_d], pj.daily_ret_latent[:n_d])[0, 1]
            )
            n_o = min(pi.daily_ret_obs.size, pj.daily_ret_obs.size)
            c_obs = float(
                np.corrcoef(pi.daily_ret_obs[:n_o], pj.daily_ret_obs[:n_o])[0, 1]
            )
            c_theory = theory["pairs"].get(key)
            if c_theory is not None:
                corr_errs_latent.append(abs(c_lat - c_theory))
            # --- ボラ相関 (潜在 log σ) と水平依存 ---
            nv = min(pi.log_vol_sub.size, pj.log_vol_sub.size)
            v_corr = float(
                np.corrcoef(pi.log_vol_sub[:nv], pj.log_vol_sub[:nv])[0, 1]
            )
            vol_corrs.append(v_corr)
            vbh = vol_correlation_by_horizon(pi.rv_daily_obs, pj.rv_daily_obs)
            if vbh.get("increasing_with_horizon") is not None:
                vol_horizon_increasing.append(bool(vbh["increasing_with_horizon"]))
            # --- 危機時相関 (3 条件付け: 和集合 / ブレッドス / 潜在 big|z_F|) ---
            n_dd = min(n_o, pi.n_days, pj.n_days, n_dm)
            mask = masks_all[i][:n_dd] | masks_all[j][:n_dd]
            cc = conditional_correlation(
                pi.daily_ret_obs[:n_dd], pj.daily_ret_obs[:n_dd], mask
            )
            if cc.get("status") == "ok":
                cc_increases.append(cc["increase"])
            cc_breadth = conditional_correlation(
                pi.daily_ret_obs[:n_dd], pj.daily_ret_obs[:n_dd],
                breadth_mask[:n_dd],
            )
            if cc_breadth.get("status") == "ok":
                cc_breadth_increases.append(cc_breadth["increase"])
            cc_bigf = (
                conditional_correlation(
                    pi.daily_ret_latent[:n_dd], pj.daily_ret_latent[:n_dd],
                    bigf_mask[:n_dd],
                )
                if bigf_mask is not None
                else na("factor_daily が不足")
            )
            if cc_bigf.get("status") == "ok":
                cc_bigf_increases.append(cc_bigf["increase"])
            # --- リードラグ (観測ミッドの 300 秒バー、±2 時間) ---
            # 60 秒バーは板の内生ノイズ (アンカー付き ZI 歩行 + バウンス) が
            # 短スケールの観測相関を ~0.003 まで沈め、CCF 全体が雑音になる
            # (事前測定 #2 実測)。κ 差の追跡ラグは数十分スケールなので、
            # 信号シェアが立つ 5 分バー × ±24 ラグ (±2h) で測る。
            spd = int(round(session / pi.obs_step_sec))
            stride = int(round(300.0 / pi.obs_step_sec))
            nd = min(pi.n_days, pj.n_days)
            bars_i = np.diff(
                pi.obs_log_price_f32[: nd * spd + 1 : stride].astype(np.float64)
            )
            bars_j = np.diff(
                pj.obs_log_price_f32[: nd * spd + 1 : stride].astype(np.float64)
            )
            n_bars_day = spd // stride
            nb = (bars_i.size // n_bars_day) * n_bars_day
            ll = lead_lag_profile(
                bars_i[:nb].reshape(-1, n_bars_day),
                bars_j[:nb].reshape(-1, n_bars_day),
                max_lag=24,
            )
            pairs[key] = {
                "beta_product_design": float(pl[i].beta * pl[j].beta),
                "true_corr_latent_1min": true_corr,
                "hy_pstar_at_trades": hy_rows,
                "hy_book_vwap": hy_book,
                "epps": epps,
                "daily_corr_latent": c_lat,
                "daily_corr_obs": c_obs,
                "daily_corr_theory": c_theory,
                "vol_corr_latent": v_corr,
                "vol_corr_by_horizon": vbh,
                "conditional_corr": cc,
                "conditional_corr_breadth": cc_breadth,
                "conditional_corr_latent_bigf": cc_bigf,
                "lead_lag": ll,
            }

    per_asset = [
        {
            "asset_index": p.asset_index,
            "beta": p.beta,
            "tick_size": p.tick_size,
            "kappa": p.kappa,
            "n_trades": int(p.trade_t.size),
            "spread_median_ticks": p.spread_median,
            "throughput_events_per_sec": p.throughput,
            "var_log_sigma_path": p.var_log_sigma_path,
            "n_crisis_episodes": len(p.crisis_episodes),
        }
        for p in pl
    ]
    fdc = factor_decomposition_check(
        [p.daily_ret_latent for p in pl], multi.factor_daily,
        [p.beta for p in pl],
    )
    # marginal_preservation (§4.2/§10): 共有分割の予算恒等式。E[M]=1 なので
    # MSM の分割は周辺を厳密に保存し、OU はガウス分散の加法で保存する。
    # 経路の実現 Var(log σ) はエポックで ±15-20% 揺れる (S1 実測) ため記録。
    total_design = (
        (config.vol_var_target_msm if config.enable_msm else 0.0)
        + (config.vol_var_target_slow if config.enable_slow_ou else 0.0)
        + (config.vol_var_target_rough if config.enable_rough else 0.0)
        + (config.vol_var_target_chaos if config.enable_chaos_vol else 0.0)
    )
    k_c = config.msm_k_common
    split_sum = (
        (config.vol_var_target_msm * k_c / config.msm_k if config.enable_msm else 0.0)
        + (
            config.vol_var_target_msm * (config.msm_k - k_c) / config.msm_k
            if config.enable_msm else 0.0
        )
        + (
            config.ou_common_share * config.vol_var_target_slow
            + (1.0 - config.ou_common_share) * config.vol_var_target_slow
            if config.enable_slow_ou else 0.0
        )
        + (config.vol_var_target_rough if config.enable_rough else 0.0)
        + (config.vol_var_target_chaos if config.enable_chaos_vol else 0.0)
    )
    path_vars = [p.var_log_sigma_path for p in pl]
    finite_vars = [v for v in path_vars if v == v]
    marginal = {
        "budget_identity_ok": bool(abs(split_sum - total_design) < 1e-12),
        "total_design": total_design,
        "split_sum": split_sum,
        "var_log_sigma_path_per_asset": path_vars,
        "var_log_sigma_path_spread_rel": (
            (max(finite_vars) - min(finite_vars)) / (sum(finite_vars) / len(finite_vars))
            if len(finite_vars) >= 2 else None
        ),
    }
    return {
        "status": "ok",
        "n_assets": n_assets,
        "theory": theory,
        "pairs": pairs,
        "per_asset": per_asset,
        "factor_decomposition": fdc,
        "marginal": marginal,
        "summary": {
            "hy_max_abs_err": max(hy_errors) if hy_errors else None,
            "epps_ratio_median": float(np.median(epps_ratios)) if epps_ratios else None,
            "epps_monotone_all": bool(epps_monotone and all(epps_monotone)),
            "daily_corr_latent_max_abs_err": (
                max(corr_errs_latent) if corr_errs_latent else None
            ),
            "vol_corr_min": min(vol_corrs) if vol_corrs else None,
            "vol_corr_max": max(vol_corrs) if vol_corrs else None,
            "vol_corr_horizon_increasing_all": bool(
                vol_horizon_increasing and all(vol_horizon_increasing)
            ),
            "crisis_corr_increase_median": (
                float(np.median(cc_increases)) if cc_increases else None
            ),
            "crisis_corr_increase_breadth_median": (
                float(np.median(cc_breadth_increases))
                if cc_breadth_increases else None
            ),
            "crisis_corr_increase_latent_bigf_median": (
                float(np.median(cc_bigf_increases)) if cc_bigf_increases else None
            ),
            "n_breadth_days": int(breadth_mask[:n_dm].sum()),
        },
    }
