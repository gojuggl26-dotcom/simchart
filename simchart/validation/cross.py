"""資産間 (非同期観測) の検証。

S0 は単一資産なので ``not_applicable`` になる。S13 で多資産を入れたときに、
共通因子から出るはずの相関が**非同期観測でも正しく測れているか**を確認するために
Hayashi-Yoshida 推定量を使う。等間隔に揃えてから相関を取ると Epps 効果
(細かい粒度ほど相関が消える見かけの現象) を自分で作り込んでしまうため、
最初から同期化しない推定量で測る。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import na, num, ok

__all__ = ["hayashi_yoshida", "hayashi_yoshida_lead_lag"]

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
    # 条件 b[j] > a[i-1] かつ b[j-1] < a[i]  (a=t1, b=t2)
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
