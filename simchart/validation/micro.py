"""マイクロストラクチャー (注文流・インパクト) の検証。

S0 では全部 ``not_applicable`` になる。それでも**スタブではなく本実装を S0 で
書いておく**のには理由がある。

インパクト整合性 ``beta = (1 - gamma) / 2``
-------------------------------------------
注文符号の自己相関が ``C(l) ~ l^{-gamma}`` と長く尾を引くのに、価格が拡散的で
あり続ける (分散比 ~ 1) ためには、propagator が ``G(l) ~ l^{-beta}`` で
``beta = (1 - gamma) / 2`` を満たしていなければならない。相関の効果を減衰する
インパクトがちょうど打ち消す、というのがこの関係の中身である。

これは S8 で実装する機能ではなく、S8 の設計を縛る制約である。メタオーダーの
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
    "spread_distribution",
    "depth_profile",
    "queue_length_distribution",
    "order_size_check",
    "placement_check",
    "book_liveness",
    "interevent_times",
    "obi",
    # --- S8 ---
    "metaorder_length_check",
    "pool_stationarity",
    "iceberg_stats",
    # --- S9 ---
    "estimate_eta",
    "obi_predictive",
    "mean_reversion_profile",
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
    ``l = 1..L`` について並べた線形系を非負最小二乗で解いて ``G`` を得る。
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


def metaorder_length_check(
    lengths, alpha_spec: float, n_min: int = 1
) -> dict:
    """メタオーダー長の離散 Pareto 適合と α の MLE (S8)。

    生成則は ``N = floor(N_min·(1−u)^{-1/α})`` で、離散裾
    ``P(N ≥ n) = (N_min/n)^α`` が厳密に成り立つ。推定は連続近似 (Hill) ではなく
    離散モデルの MLE: ``P(N = n) = (N_min/n)^α − (N_min/(n+1))^α``。
    SE は対数尤度の数値曲率から。あわせて固定点の裾確率の二項 z も返す
    (MLE が壊れても裾のずれを直接見られるように)。
    """
    arr = np.asarray(lengths, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr) & (arr >= n_min)]
    m = arr.size
    if m < 1000:
        return na(f"メタオーダーが足りません (n={m})")

    def negll(a: float) -> float:
        if a <= 1.0001 or a >= 5.0:
            return 1e300
        p = (n_min / arr) ** a - (n_min / (arr + 1.0)) ** a
        if (p <= 0).any():
            return 1e300
        return -float(np.sum(np.log(p)))

    res = optimize.minimize_scalar(negll, bounds=(1.01, 4.0), method="bounded")
    a_hat = float(res.x)
    h = 1e-4
    curv = (negll(a_hat + h) - 2.0 * negll(a_hat) + negll(a_hat - h)) / h**2
    se = float(1.0 / np.sqrt(curv)) if curv > 0 else None

    tail_z = {}
    for n0 in (2, 5, 20, 100):
        p_th = (n_min / n0) ** alpha_spec
        if p_th * m < 20:
            continue
        obs = float((arr >= n0).mean())
        tail_z[f"z_at_{n0}"] = num(
            (obs - p_th) / np.sqrt(p_th * (1 - p_th) / m)
        )
    return ok(
        num(a_hat),
        alpha_hat=num(a_hat),
        alpha_spec=num(alpha_spec),
        difference=num(a_hat - alpha_spec),
        se=num(se),
        n=m,
        mean_length=num(arr.mean()),
        max_length=num(arr.max()),
        **tail_z,
    )


def pool_stationarity(pool_grid, burn_points: int = 0) -> dict:
    """メタオーダー・プール占有 (グリッド標本) の定常性 (S8)。

    ゲートは前半と後半の平均が ±10% 以内・増加トレンドなし。
    前後半比較そのものがトレンドの計器なので、OLS 勾配は記録に留める
    (系列は強く自己相関しており naive な傾き検定の p 値は意味を持たない)。
    α<2 の Pareto 長では占有の裾が重く (whale 滞留)、平均は少数の episode に
    引かれる — 中央値と分位も併記する。
    """
    arr = np.asarray(pool_grid, dtype=np.float64).ravel()[burn_points:]
    if arr.size < 1000:
        return na(f"標本が足りません (n={arr.size})")
    half = arr.size // 2
    m1 = float(arr[:half].mean())
    m2 = float(arr[half:].mean())
    overall = float(arr.mean())
    rel = (m2 - m1) / m1 if m1 > 0 else None
    x = np.arange(arr.size, dtype=np.float64)
    slope = float(np.polyfit(x, arr, 1)[0])
    trend_frac = slope * arr.size / overall if overall > 0 else None
    return ok(
        num(rel),
        mean_first_half=num(m1),
        mean_second_half=num(m2),
        rel_diff=num(rel),
        mean=num(overall),
        median=num(float(np.median(arr))),
        p95=num(float(np.percentile(arr, 95))),
        max=num(float(arr.max())),
        min=num(float(arr.min())),
        trend_frac_of_mean=num(trend_frac),
        n=int(arr.size),
    )


def iceberg_stats(diag: Mapping[str, Any] | None) -> dict:
    """アイスバーグの比率・隠れ量・補充回数 (S8。l3 診断の整形)。"""
    if not diag:
        return na("アイスバーグが無効か、診断がありません")
    n_ice = float(diag.get("n_iceberg_orders", 0))
    if n_ice <= 0:
        return na("アイスバーグ注文が 0 件です")
    refills = float(diag.get("refills", 0))
    hidden = float(diag.get("hidden_volume_in", 0.0))
    return ok(
        num(diag.get("iceberg_share_of_lo")),
        share_of_lo=num(diag.get("iceberg_share_of_lo")),
        n_orders=int(n_ice),
        refills=int(refills),
        refills_per_order=num(refills / n_ice),
        hidden_volume_in=num(hidden),
        hidden_per_order=num(hidden / n_ice),
        refill_volume=num(diag.get("refill_volume")),
    )


def estimate_eta(price_ticks) -> dict:
    """Robert–Rosenbaum の実効 η (S9 §8): 価格変化の継続/交替比 N_c/(2N_a)。

    ティック離散の価格列からゼロでない変化の方向列を作り、
    継続 (同方向) N_c と交替 (反転) N_a から η̂ = N_c / (2 N_a)。
    iid の ±1 変化なら η = 0.5、交替過多 (バウンス) で η < 0.5。
    経験値 0.1〜0.3 は取引価格系列の値 (R-R の枠組み自体が取引価格の
    離散化モデル)。ミッドに当てた値は別物として記録する — 系列を混ぜないこと。
    """
    arr = np.asarray(price_ticks, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 1000:
        return na(f"価格点が足りません (n={arr.size})")
    ch = np.diff(arr)
    ch = ch[ch != 0]
    if ch.size < 500:
        return na(f"価格変化が足りません (n={ch.size})")
    sg = np.sign(ch)
    cont = int((sg[1:] == sg[:-1]).sum())
    alt = int((sg[1:] != sg[:-1]).sum())
    if alt == 0:
        return na("交替が 0 件です (退化系列)")
    eta = cont / (2.0 * alt)
    # 変化方向の 1 次相関 (= 2 状態連鎖なら (cont−alt)/(cont+alt))
    corr_sign = (cont - alt) / (cont + alt)
    return ok(
        num(eta),
        eta=num(eta),
        n_continuations=cont,
        n_alternations=alt,
        n_changes=int(ch.size),
        change_sign_corr=num(corr_sign),
    )


def obi_predictive(imbalance, mid, horizons=(1, 2, 5, 10)) -> dict:
    """(10) OBI の予測相関: corr(I_k, m_{k+h} − m_k) (S9 §7)。

    ``imbalance`` と ``mid`` は同じイベント格子 (攻撃注文ごと等) に整列した列。
    I はその時点で観測できる板の量、リターンはその後のミッド変化 —
    時間契約 (説明変数の確定時刻 ≤ 目的変数の開始時刻) を満たす予測相関。
    """
    i_arr = np.asarray(imbalance, dtype=np.float64).ravel()
    m_arr = np.asarray(mid, dtype=np.float64).ravel()
    if i_arr.size != m_arr.size:
        return na(f"I ({i_arr.size}) とミッド ({m_arr.size}) の長さが一致しません")
    if i_arr.size < 5000:
        return na(f"標本が足りません (n={i_arr.size})")
    rows = {}
    for h in horizons:
        h = int(h)
        if h < 1 or h >= i_arr.size - 10:
            continue
        dm = m_arr[h:] - m_arr[:-h]
        ik = i_arr[:-h]
        good = np.isfinite(dm) & np.isfinite(ik)
        if good.sum() < 1000:
            continue
        rows[f"h{h}"] = num(float(np.corrcoef(ik[good], dm[good])[0, 1]))
    if not rows:
        return na("有効なホライズンがありません")
    first = rows.get("h1")
    return ok(first, corr_h1=first, horizons=rows, n=int(i_arr.size))


def mean_reversion_profile(signs, mid, meta_ids=None, horizons=(1, 2, 3, 5, 10, 20, 50)) -> dict:
    """約定後のミッド戻り曲線 R(h) = E[ε_k·(m_{k+h} − m_k)] (S9 §10)。

    無条件の R(h) は符号の長期記憶 (将来の同符号フロー) で増加し、板の
    復元力が見えない。``meta_ids`` を渡すと**ノイズトレード (meta_id < 0) に
    条件付け**る — ノイズの符号は iid なので将来フローと独立で、曲線は
    純粋な板応答 (インパクト → 部分的な戻り) になる。
    """
    s_arr = np.asarray(signs, dtype=np.float64).ravel()
    m_arr = np.asarray(mid, dtype=np.float64).ravel()
    if s_arr.size != m_arr.size:
        return na("符号とミッドの長さが一致しません")
    if meta_ids is not None:
        base_idx = np.flatnonzero(np.asarray(meta_ids) < 0)
        conditioned = "noise_trades"
    else:
        base_idx = np.arange(s_arr.size)
        conditioned = "all"
    if base_idx.size < 5000:
        return na(f"条件付け後の標本が足りません (n={base_idx.size})")
    prof = {}
    vals_list = []
    for h in horizons:
        h = int(h)
        sel = base_idx[base_idx + h < m_arr.size]
        v = s_arr[sel] * (m_arr[sel + h] - m_arr[sel])
        good = np.isfinite(v)
        if good.sum() < 1000:
            continue
        mean_v = float(v[good].mean())
        prof[f"h{h}"] = num(mean_v)
        vals_list.append(mean_v)
    if len(vals_list) < 4:
        return na("有効なホライズンが足りません")
    peak = max(vals_list[:2])
    tail = vals_list[-1]
    reversion_frac = (peak - tail) / peak if peak > 0 else None
    monotone_nondecreasing = all(
        b >= a - 1e-12 for a, b in zip(vals_list, vals_list[1:])
    )
    return ok(
        num(reversion_frac),
        profile=prof,
        conditioned_on=conditioned,
        impact_peak=num(peak),
        tail=num(tail),
        reversion_frac=num(reversion_frac),
        monotone_nondecreasing=bool(monotone_nondecreasing),
        n=int(base_idx.size),
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
    (設定値をそのまま報告するのではなく、出力から測り直すことに意味がある)。

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


# ---------------------------------------------------------------------------
# S6: ZI 板の測定 
# ---------------------------------------------------------------------------
def _burned(t: np.ndarray, burn_in_sec: float) -> np.ndarray:
    return np.asarray(t, dtype=np.float64) >= burn_in_sec


def spread_distribution(
    best_bid_tick: np.ndarray, best_ask_tick: np.ndarray, t: np.ndarray,
    burn_in_sec: float = 0.0,
) -> dict:
    """スプレッド (ティック単位) の分布。イベント時点で測る。"""
    bb = np.asarray(best_bid_tick)
    ba = np.asarray(best_ask_tick)
    mask = _burned(t, burn_in_sec) & (bb >= 0) & (ba >= 0)
    if mask.sum() < 100:
        return na(f"有効なイベントが足りません (n={int(mask.sum())})")
    s = (ba[mask] - bb[mask]).astype(np.float64)
    qs = np.percentile(s, [5, 25, 50, 75, 95, 99])
    return ok(
        num(float(np.median(s))),
        median=num(float(np.median(s))),
        mean=num(float(s.mean())),
        p5=num(qs[0]), p25=num(qs[1]), p75=num(qs[3]), p95=num(qs[4]), p99=num(qs[5]),
        n=int(s.size),
        n_nonpositive=int((s <= 0).sum()),
        min=num(float(s.min())),
    )


def depth_profile(book_snapshots, burn_in_sec: float = 0.0,
                  tick_size: float | None = None) -> dict:
    """best からの距離別の平均デプス (スナップショットのレベル順位ベース)。

    「デプスのピークが best (Δ=0) でない」のが ZI 板の健全な形 (設計要件
    depth_front_depletion): best は成行に最初に食われるので前方が消耗し、ピークは
    数レベル奥に来る。ピークが best にあるなら約定による前方消耗が働いていない。
    """
    b = book_snapshots
    if b.is_empty:
        return na("スナップショットがありません")
    mask = _burned(b.t, burn_in_sec)
    if mask.sum() < 10:
        return na("burn-in 後のスナップショットが足りません")
    bid_sz = np.where(np.isnan(b.bid_px[mask]), np.nan, b.bid_sz[mask])
    ask_sz = np.where(np.isnan(b.ask_px[mask]), np.nan, b.ask_sz[mask])
    prof_bid = np.nanmean(bid_sz, axis=0)
    prof_ask = np.nanmean(ask_sz, axis=0)
    prof = 0.5 * (prof_bid + prof_ask)
    peak = int(np.nanargmax(prof))
    # レベル順位 → ティック距離 (中央値) も出す (small tick では順位 != 距離)。
    tick_dist = None
    if tick_size:
        bb = b.bid_px[mask][:, 0]
        d = (bb[:, None] - b.bid_px[mask]) / tick_size
        tick_dist = [num(float(np.nanmedian(d[:, k]))) for k in range(d.shape[1])]
    return ok(
        float(peak),
        peak_level=int(peak),  # 0 = best
        peak_is_best=bool(peak == 0),
        profile_bid=[num(float(x)) for x in prof_bid],
        profile_ask=[num(float(x)) for x in prof_ask],
        profile_mean=[num(float(x)) for x in prof],
        median_tick_distance=tick_dist,
        n_snapshots=int(mask.sum()),
    )


def queue_length_distribution(book_snapshots, burn_in_sec: float = 0.0) -> dict:
    """best キュー長 (ロット) の分布。"""
    b = book_snapshots
    if b.is_empty:
        return na("スナップショットがありません")
    mask = _burned(b.t, burn_in_sec)
    q = np.concatenate([b.bid_sz[mask][:, 0], b.ask_sz[mask][:, 0]])
    q = q[np.isfinite(q) & (q > 0)]
    if q.size < 100:
        return na("有効なキュー観測が足りません")
    return ok(
        num(float(np.median(q))),
        median=num(float(np.median(q))),
        mean=num(float(q.mean())),
        p95=num(float(np.percentile(q, 95))),
        cv=num(float(q.std() / q.mean())),
        n=int(q.size),
    )


def order_size_check(sizes: np.ndarray, lot_values, lot_probs, w_round: float,
                     pareto_alpha: float) -> dict:
    """サイズ分布の仕様適合 。

    混合 (離散ロット + 切り上げ Pareto) なので素朴な KS は使えない (離散原子が
    あると KS の帰無分布が成り立たない)。代わりに (a) 各ロット点の質量が仕様の
    期待値どおりか (二項 z)、(b) 非ロット部の裾指数が Pareto α と整合するか、で見る。
    """
    s = np.asarray(sizes, dtype=np.float64)
    s = s[s > 0]
    if s.size < 1000:
        return na(f"標本が足りません (n={s.size})")
    n = s.size
    lot_values = np.asarray(lot_values, dtype=np.float64)
    lot_probs = np.asarray(lot_probs, dtype=np.float64)

    def pareto_ceil_mass(v: float) -> float:
        # ceil(Pareto(xm=1)) = v になる質量 = P(v-1 < X <= v) (v >= 2)。
        if v <= 1.0:
            return 0.0
        lo = max(v - 1.0, 1.0)
        return lo ** (-pareto_alpha) - v ** (-pareto_alpha)

    rows = []
    max_z = 0.0
    for v, p in zip(lot_values, lot_probs):
        expected = w_round * p + (1.0 - w_round) * pareto_ceil_mass(float(v))
        observed = float((s == v).mean())
        se = float(np.sqrt(expected * (1 - expected) / n))
        z = (observed - expected) / se if se > 0 else 0.0
        rows.append({"lot": num(v), "expected": num(expected),
                     "observed": num(observed), "z": num(z)})
        if abs(z) > max_z:
            max_z = abs(z)
    tail = s[~np.isin(s, lot_values)]
    tail_alpha = None
    if tail.size > 200:
        t_sorted = np.sort(tail)[::-1]
        k = max(int(0.5 * t_sorted.size), 50)
        top = t_sorted[:k]
        denom = float(np.mean(np.log(top / top[-1] + 1e-300)))
        tail_alpha = 1.0 / denom if denom > 0 else None
    return ok(
        num(max_z),
        max_abs_z=num(max_z),
        table=rows,
        tail_alpha=num(tail_alpha) if tail_alpha else None,
        spec_alpha=num(pareto_alpha),
        n=int(n),
    )


def placement_check(
    lo_price_tick: np.ndarray, lo_side: np.ndarray,
    prev_best_bid_tick: np.ndarray, prev_best_ask_tick: np.ndarray,
    mu_place_spec: float, place_offset: float, max_place: int,
) -> dict:
    """配置距離分布の仕様適合: べき指数の推定が仕様 ±0.2 。

    Δ = 発注直前の同サイド best からの板内距離。イベントログの best は
    イベント後の値なので、呼び出し側は前イベントの best を渡すこと (improvement は
    自分が best を書き換えるため)。推定は板内配置 (Δ >= 1) の対数ビン回帰:
    log P(Δ) = 定数 − (1+μ) log(Δ+Δ0)。
    """
    side = np.asarray(lo_side)
    px = np.asarray(lo_price_tick, dtype=np.float64)
    bb = np.asarray(prev_best_bid_tick, dtype=np.float64)
    ba = np.asarray(prev_best_ask_tick, dtype=np.float64)
    delta = np.where(side > 0, bb - px, px - ba)
    delta = delta[np.isfinite(delta)]
    pos = delta[(delta >= 1) & (delta <= max_place)]
    if pos.size < 1000:
        return na(f"板内配置の標本が足りません (n={pos.size})")
    edges = np.unique(np.round(np.geomspace(1, max_place / 2, 18)).astype(int))
    counts, _ = np.histogram(pos, bins=np.append(edges, edges[-1] * 2))
    widths = np.diff(np.append(edges, edges[-1] * 2)).astype(float)
    centers = edges.astype(float)
    dens = counts / widths / pos.size
    m = (counts > 20) & (dens > 0)
    if m.sum() < 5:
        return na("回帰に使えるビンが足りません")
    slope, intercept = np.polyfit(np.log(centers[m] + place_offset), np.log(dens[m]), 1)
    mu_hat = -slope - 1.0
    resid = np.log(dens[m]) - (slope * np.log(centers[m] + place_offset) + intercept)
    ss = float(((np.log(dens[m]) - np.log(dens[m]).mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss if ss > 0 else float("nan")
    return ok(
        num(mu_hat),
        mu_estimated=num(mu_hat),
        mu_spec=num(mu_place_spec),
        difference=num(mu_hat - mu_place_spec),
        fit_r2=num(r2),
        frac_inspread=num(float((delta < 0).mean())),
        frac_at_best=num(float((delta == 0).mean())),
        n_interior=int(pos.size),
    )


def book_liveness(empty_time_sec: float, horizon_sec: float,
                  n_reject_events: float, reject_vol: float) -> dict:
    """片側枯渇の頻度 (設計要件 — 時間比率 < 0.1% がゲート)。"""
    frac = empty_time_sec / horizon_sec if horizon_sec > 0 else float("nan")
    return ok(
        num(frac),
        empty_side_time_fraction=num(frac),
        empty_side_time_sec=num(empty_time_sec),
        mo_reject_events=num(n_reject_events),
        mo_reject_volume=num(reject_vol),
    )


def interevent_times(t: np.ndarray, burn_in_sec: float = 0.0,
                     window_sec: float = 1800.0) -> dict:
    """到着間隔の指数性 (S7 の比較基準 — 設計要件)。

    2 つの独立な検査を返す:
    - CV² = Var(Δt)/E[Δt]²。指数分布なら 1
    - 窓あたり件数の過分散指数 (Fano factor) Var(N)/E[N]。Poisson なら 1

    S7 で Hawkes を入れるとどちらも 1 を大きく超える (クラスタリング) ので、
    ここで ≈1 を記録しておくことが自己励起が入ったことの検定の対照になる。
    """
    tt = np.asarray(t, dtype=np.float64)
    tt = tt[tt >= burn_in_sec]
    if tt.size < 1000:
        return na(f"イベントが足りません (n={tt.size})")
    dt = np.diff(tt)
    dt = dt[dt > 0]
    cv2 = float(dt.var() / dt.mean() ** 2)
    n_windows = int((tt[-1] - tt[0]) / window_sec)
    fano = None
    if n_windows >= 30:
        counts, _ = np.histogram(tt, bins=n_windows)
        fano = float(counts.var(ddof=1) / counts.mean())
    return ok(
        num(cv2),
        cv2=num(cv2),
        fano_factor=num(fano) if fano is not None else None,
        window_sec=float(window_sec),
        n=int(dt.size),
        mean_interarrival_sec=num(float(dt.mean())),
    )


def obi(book_snapshots, levels: int = 5, burn_in_sec: float = 0.0) -> dict:
    """板不均衡 OBI = (D_bid − D_ask)/(D_bid + D_ask) の分布 (S9 で本格化する量)。"""
    b = book_snapshots
    if b.is_empty:
        return na("スナップショットがありません")
    mask = _burned(b.t, burn_in_sec)
    k = min(levels, b.n_levels)
    db = np.nansum(
        np.where(np.isnan(b.bid_px[mask][:, :k]), 0.0, b.bid_sz[mask][:, :k]), axis=1
    )
    da = np.nansum(
        np.where(np.isnan(b.ask_px[mask][:, :k]), 0.0, b.ask_sz[mask][:, :k]), axis=1
    )
    tot = db + da
    good = tot > 0
    if good.sum() < 100:
        return na("有効なスナップショットが足りません")
    x = (db[good] - da[good]) / tot[good]
    return ok(
        num(float(np.abs(x).mean())),
        mean=num(float(x.mean())),
        sd=num(float(x.std())),
        mean_abs=num(float(np.abs(x).mean())),
        p95_abs=num(float(np.percentile(np.abs(x), 95))),
        levels=int(k),
        n=int(good.sum()),
    )
