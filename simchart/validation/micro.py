"""マイクロストラクチャー (注文流・インパクト) の検証。

S0 では全部 ``not_applicable`` になる。それでも**スタブではなく本実装を S0 で
書いておく**のには理由がある。

インパクト整合性 ``beta = (1 - gamma) / 2``
-------------------------------------------
注文符号の自己相関が ``C(l) ~ l^{-gamma}`` と長く尾を引くのに、価格が拡散的で
あり続ける (分散比 ~ 1) ためには、propagator が ``G(l) ~ l^{-beta}`` で
``beta = (1 - gamma) / 2`` を満たしていなければならない。相関の効果を減衰する
インパクトがちょうど打ち消す、というのがこの関係の中身である。

これは **S8 で実装する機能ではなく、S8 の設計を縛る制約**である。メタオーダーの
分割則 (gamma を決める) とインパクトの減衰 (beta を決める) を独立にチューニング
すると、必ずどちらかが壊れる。測定手段を先に用意しておけば、S8 で「どちらを
動かすべきか」が即座に判る。後付けできない類のものなので S0 で書く。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize, stats

from .base import na, num, ok
from .memory import acf, power_law_fit

__all__ = [
    "sign_acf",
    "response_function",
    "propagator_fit",
    "impact_consistency",
    "sqrt_law_check",
    "branching_ratio_reestimate",
]

_NO_FLOW = (
    "注文流イベントが無いため測定できません。S6 で板層、S8 でメタオーダー分割を"
    "入れた段階で初めて意味を持ちます。"
)


def _clean_signs(signs) -> np.ndarray | None:
    if signs is None:
        return None
    arr = np.asarray(signs, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    return arr


def sign_acf(signs, max_lag: int = 200, fit_lag_range: tuple[int, int] = (5, 200)) -> dict:
    """注文符号の自己相関と、その冪則減衰指数 gamma。

    実データでは ``C(l) ~ l^{-gamma}`` で ``gamma`` は 0.5 前後 (長期記憶)。
    これはメタオーダーの分割によって生じる。
    """
    eps = _clean_signs(signs)
    if eps is None:
        return na(_NO_FLOW)
    if eps.size < 100:
        return na(f"約定数が足りません (n={eps.size})")

    base = acf(eps, max_lag=min(max_lag, eps.size - 1))
    if base["status"] != "ok":
        return base
    lags = np.asarray(base["lags"], dtype=np.float64)
    values = np.array([np.nan if v is None else v for v in base["values"]], dtype=np.float64)
    fit = power_law_fit(lags, values, fit_lag_range)
    return ok(
        base["lag1"],
        lag1=base["lag1"],
        n=base["n"],
        gamma=fit.get("gamma"),
        gamma_fit=fit,
        conf95=base["conf95"],
        lags=base["lags"],
        values=base["values"],
    )


def response_function(signs, log_price, max_lag: int = 200) -> dict:
    """応答関数 ``R(l) = E[(p_{n+l} - p_n) * eps_n]``。

    符号つきの平均価格変化。実データでは緩やかに増加して飽和する。
    """
    eps = _clean_signs(signs)
    price = _clean_signs(log_price)
    if eps is None or price is None:
        return na(_NO_FLOW)
    if eps.size != price.size:
        return na(f"符号 ({eps.size}) と価格 ({price.size}) の長さが一致しません")
    n = eps.size
    if n < max_lag + 50:
        return na(f"約定数が足りません (n={n}, max_lag={max_lag})")

    lags = np.arange(1, max_lag + 1)
    values = np.empty(max_lag, dtype=np.float64)
    for i, lag in enumerate(lags):
        values[i] = float(np.mean((price[lag:] - price[:-lag]) * eps[: n - lag]))
    return ok(
        num(values[0]),
        n=int(n),
        lags=lags.tolist(),
        values=[num(v) for v in values],
    )


def propagator_fit(
    signs,
    sizes,
    log_price,
    max_lag: int = 200,
    fit_lag_range: tuple[int, int] = (5, 200),
) -> dict:
    """transient impact model の propagator ``G(l)`` を推定し、減衰指数 beta を出す。

    モデルは ``p_t = sum_{n<t} G(t-n) * v_n * eps_n + noise``。応答関数と符号の
    自己共分散のあいだには

    ``R(l) = sum_{j>=1} G(j) * [C(|l-j|) - C(j)]``

    という関係が成り立つ (``C`` は符号の自己共分散、``C(0)=1``)。これを
    ``l = 1..L`` について並べた線形系を**非負最小二乗**で解いて ``G`` を得る。
    非負制約を入れるのは、符号つきインパクトが負になる解は物理的に意味が無く、
    無制約だと打ち切り誤差で振動するため。

    ``L`` で打ち切っている分だけ大きなラグの ``G`` にバイアスが乗る。``beta`` の
    当てはめ範囲 ``fit_lag_range`` の上限を ``L`` より十分小さく取ること。
    """
    eps = _clean_signs(signs)
    price = _clean_signs(log_price)
    if eps is None or price is None:
        return na(_NO_FLOW)
    if eps.size != price.size:
        return na(f"符号 ({eps.size}) と価格 ({price.size}) の長さが一致しません")

    n = eps.size
    if n < 5 * max_lag:
        return na(f"約定数が足りません (n={n}, max_lag={max_lag})")

    if sizes is not None:
        size_arr = np.asarray(sizes, dtype=np.float64).ravel()
        if size_arr.size != n:
            return na("サイズ列の長さが符号列と一致しません")
        mean_size = float(np.mean(np.abs(size_arr)))
        v = np.abs(size_arr) / mean_size if mean_size > 0 else np.ones(n)
    else:
        v = np.ones(n, dtype=np.float64)

    flow = v * eps
    resp = response_function(eps, price, max_lag=max_lag)
    if resp["status"] != "ok":
        return resp
    r_vec = np.array([np.nan if x is None else x for x in resp["values"]], dtype=np.float64)
    if not np.all(np.isfinite(r_vec)):
        return na("応答関数に有限でない値があります")

    cov = acf(flow, max_lag=2 * max_lag)
    if cov["status"] != "ok":
        return cov
    c_full = np.array([np.nan if x is None else x for x in cov["values"]], dtype=np.float64)

    lags = np.arange(1, max_lag + 1)
    design = c_full[np.abs(lags[:, None] - lags[None, :])] - c_full[lags][None, :]
    solution, residual = optimize.nnls(design, r_vec)

    fit = power_law_fit(lags.astype(np.float64), solution, fit_lag_range)
    return ok(
        fit.get("gamma"),
        beta=fit.get("gamma"),
        beta_fit=fit,
        n_trades=int(n),
        max_lag=int(max_lag),
        nnls_residual=num(residual),
        response_scale=num(r_vec[0]),
        propagator=[num(g) for g in solution],
        lags=lags.tolist(),
    )


def impact_consistency(
    gamma: float | None, beta: float | None, gamma_se: float | None = None,
    beta_se: float | None = None,
) -> dict:
    """``beta = (1 - gamma) / 2`` の整合性を確認する。

    符号の長期記憶 (gamma) と propagator の減衰 (beta) が拡散的な価格と両立する
    ための条件。片方だけ合わせても価格が拡散にならない。
    """
    if gamma is None or beta is None:
        return na(
            "gamma または beta が未推定です。符号自己相関 (S8) と propagator (S8) の"
            "両方が揃って初めて判定できます。"
        )
    predicted = (1.0 - float(gamma)) / 2.0
    diff = float(beta) - predicted
    z = None
    if beta_se and gamma_se:
        se = float(np.sqrt(beta_se**2 + (gamma_se / 2.0) ** 2))
        if se > 0:
            z = diff / se
    return ok(
        num(diff),
        gamma=num(gamma),
        beta=num(beta),
        beta_predicted=num(predicted),
        difference=num(diff),
        z=num(z) if z is not None else None,
    )


def sqrt_law_check(metaorders: Sequence[Mapping[str, Any]] | None) -> dict:
    """メタオーダーの平方根則 ``I ~ sigma * (Q/V)^delta``、``delta ~ 0.5``。

    Parameters
    ----------
    metaorders:
        各メタオーダーの記録。必要なキーは ``q`` (執行数量)、``v`` (同期間の
        市場出来高)、``sigma`` (同期間のボラ)、``impact`` (対数価格の変化)。

    Notes
    -----
    ``log(I / sigma) = a + delta * log(Q / V)`` を最小二乗する。実データの
    ``delta`` は 0.4〜0.6 に収まることが多い。
    """
    if not metaorders:
        return na(
            "メタオーダーの記録がありません。S8 でメタオーダー分割を実装した段階で"
            "測定できるようになります。"
        )
    q_list, ratio_list, y_list = [], [], []
    for row in metaorders:
        try:
            q = float(row["q"])
            v = float(row["v"])
            sigma = float(row["sigma"])
            impact = float(row["impact"])
        except (KeyError, TypeError, ValueError):
            continue
        if q <= 0 or v <= 0 or sigma <= 0 or impact <= 0:
            continue
        q_list.append(q)
        ratio_list.append(q / v)
        y_list.append(impact / sigma)
    if len(ratio_list) < 20:
        return na(f"有効なメタオーダーが足りません (n={len(ratio_list)})")

    x = np.log(np.array(ratio_list))
    y = np.log(np.array(y_list))
    slope, intercept, rvalue, _, stderr = stats.linregress(x, y)
    return ok(
        num(slope),
        delta=num(slope),
        se=num(stderr),
        intercept=num(intercept),
        r2=num(rvalue**2),
        n=len(ratio_list),
        deviation_from_half=num(slope - 0.5),
        z_vs_half=num((slope - 0.5) / stderr) if stderr > 0 else None,
    )


def branching_ratio_reestimate(
    events,
    window_seconds: Sequence[float] | None = None,
    t_span: tuple[float, float] | None = None,
    target: float | None = None,
) -> dict:
    """イベント時刻から Hawkes の分岐比 n を再推定する (Fano 比法)。

    分岐比 n の Hawkes 過程では、十分長い窓での事象数の分散平均比が
    ``Var / Mean -> 1 / (1 - n)^2`` に収束する。したがって
    ``n_hat = 1 - sqrt(Mean / Var)``。カーネル形状を仮定しないので、
    S7 で入れた n が S8〜S11 の改変を経ても保たれているかの確認に使える
    (**設定値をそのまま報告するのではなく、出力から測り直す**ことに意味がある)。

    Parameters
    ----------
    events:
        イベント時刻の配列、または ``t`` 属性を持つオブジェクト
        (:class:`~simchart.types.EventLog`)。
    target:
        設定した分岐比。渡すと差分も報告する。
    """
    times = getattr(events, "t", events)
    arr = _clean_signs(times)
    if arr is None:
        return na(
            "イベントが無いため分岐比を再推定できません。自己励起は S7 で導入します。"
        )
    arr = np.sort(arr[np.isfinite(arr)])
    if arr.size < 200:
        return na(f"イベント数が足りません (n={arr.size})")

    t0, t1 = (float(arr[0]), float(arr[-1])) if t_span is None else (float(t_span[0]), float(t_span[1]))
    span = t1 - t0
    if span <= 0:
        return na("イベントの時間範囲が 0 です")

    if window_seconds is None:
        window_seconds = [span / k for k in (2000, 1000, 500, 200, 100, 50)]

    rows = []
    for w in window_seconds:
        w = float(w)
        n_windows = int(span // w)
        if w <= 0 or n_windows < 30:
            continue
        edges = t0 + np.arange(n_windows + 1, dtype=np.float64) * w
        counts = np.histogram(arr, bins=edges)[0].astype(np.float64)
        mean = float(counts.mean())
        var = float(counts.var(ddof=1))
        if mean <= 0 or var <= 0:
            continue
        ratio = mean / var
        n_hat = 1.0 - float(np.sqrt(ratio)) if ratio <= 1.0 else None
        rows.append(
            {
                "window_seconds": w,
                "n_windows": int(n_windows),
                "mean_count": num(mean),
                "var_count": num(var),
                "fano": num(var / mean),
                "branching_ratio": num(n_hat) if n_hat is not None else None,
            }
        )

    usable = [row for row in rows if row["branching_ratio"] is not None]
    if not usable:
        return na("有効な窓が得られませんでした (過分散が観測されていない)", table=rows)
    # 窓が長いほど漸近式に近い。最長の窓の推定値を代表値にする。
    best = max(usable, key=lambda row: row["window_seconds"])
    out = ok(
        best["branching_ratio"],
        branching_ratio=best["branching_ratio"],
        window_seconds=best["window_seconds"],
        n_events=int(arr.size),
        span_seconds=num(span),
        table=rows,
    )
    if target is not None:
        out["target"] = float(target)
        out["difference"] = num(best["branching_ratio"] - float(target))
    return out
