"""スケーリング (時間集計に対する振る舞い) の検証。

S0 での期待値
-------------
- 分散比 ~ 1 (どの集計幅でも分散が時間に比例)
- 尖度がどのスケールでも ~ 3 (**集計しても変わらない** = 単一フラクタル)
- ``zeta_q = q/2`` の直線 (多重フラクタルでない)
- signature plot が平坦 (マイクロストラクチャー・ノイズが無い)
- log P は単位根を棄却せず、リターンは棄却する

尖度がスケールで変わらないことは S0 の**中心的な主張**である。実データでは
高頻度ほど尖度が高く、集計すると 3 に近づく (集計正規性)。これを再現するには
テールがボラ過程とジャンプから内生的に出ている必要があり、革新項に t 分布などを
外生的に入れてしまうと全スケールで尖度が高いままになって永久に再現できない。

有限標本と閾値
--------------
粗いスケールほど標本数が減り、尖度の標準誤差は ``sqrt(24/n)`` で増える。1 日
スケール (500 標本) では標準誤差が 0.22 になり、ゲート [2.6, 3.4] は 1.8 標準誤差
しかない。そこで **標本数がしきい値未満のスケールは「記録のみ」** とし、ゲート
判定からは外す。閾値を緩めるのではなく測定対象を選ぶ、という考え方をとる。
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from .base import na, num, ok

__all__ = [
    "variance_ratio",
    "kurtosis_by_scale",
    "zeta_q",
    "signature_plot",
    "adf_test",
    "adf_combined",
]


def _as_2d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("入力は 1 次元または (セッション, 標本) の 2 次元である必要があります")
    return a


def _aggregate(r2d: np.ndarray, k: int) -> np.ndarray | None:
    """セッション内で ``k`` 本ずつリターンを足し合わせる (端数は捨てる)。"""
    n_rows, n_cols = r2d.shape
    n_blocks = n_cols // k
    if n_blocks < 1:
        return None
    return r2d[:, : n_blocks * k].reshape(n_rows, n_blocks, k).sum(axis=2)


def _scale_steps(scale_sec: float, step_seconds: float) -> int | None:
    ratio = scale_sec / step_seconds
    k = int(round(ratio))
    if k < 1 or abs(ratio - k) > 1e-9:
        return None
    return k


# ---------------------------------------------------------------------------
def variance_ratio(log_price: np.ndarray, qs=(2, 4, 8, 16, 32, 64)) -> dict:
    """Lo-MacKinlay の分散比 (重複標本版)。

    ``VR(q) = Var[q 期リターン] / (q * Var[1 期リターン])``。ランダムウォークなら 1。

    実装では Lo-MacKinlay の記法に従い、q 期リターンの二乗和を
    ``m = q * (項数) * (1 - q/N)`` で割った ``sigma_c^2(q)`` を作る。**この m に
    既に q が入っている**ので、比を取るときに改めて q で割ってはならない
    (割ると VR が 1/q になる)。

    ``log_price`` が 2 次元なら行をセッションとみなし、q 期リターンは**セッション
    内で完結するものだけ**を使う。平均は全標本から 1 つだけ推定するので、
    Lo-MacKinlay のバイアス補正項 ``(1 - q/N)`` は全体の標本数 N で入れる。

    ``z`` は等分散を仮定した漸近分散 ``2(2q-1)(q-1)/(3qN)`` に基づく統計量。
    S0 は等分散なのでこの仮定は正しい。S1 以降 (ボラが動く) では過小になるため、
    そのときは heteroskedasticity-robust 版に差し替えること。
    """
    lp = _as_2d(log_price)
    n_rows, n_cols = lp.shape
    if n_cols < 3:
        return na(f"1 セッションあたりの点数が足りません (n={n_cols})")
    r = np.diff(lp, axis=1)
    total = r.size
    mu = float(r.mean())
    var1 = float(((r - mu) ** 2).sum() / (total - 1))
    if var1 <= 0:
        return na("1 期リターンの分散が 0 です")

    rows = []
    for q in qs:
        q = int(q)
        if q < 2 or q >= n_cols:
            rows.append({"q": q, "status": "not_applicable", "vr": None, "z": None})
            continue
        y = lp[:, q:] - lp[:, :-q]
        n_terms = y.size
        denom = q * n_terms * (1.0 - q / total)
        var_q = float(((y - q * mu) ** 2).sum() / denom)
        vr = var_q / var1
        asym_var = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * total)
        z = (vr - 1.0) / np.sqrt(asym_var) if asym_var > 0 else None
        rows.append(
            {
                "q": q,
                "status": "ok",
                "vr": num(vr),
                "z": num(z),
                "n_overlapping": int(n_terms),
                "se": num(np.sqrt(asym_var)),
            }
        )

    valid = [row for row in rows if row["vr"] is not None]
    if not valid:
        return na("有効な q がありません", table=rows)
    devs = [abs(row["vr"] - 1.0) for row in valid]
    zs = [abs(row["z"]) for row in valid if row["z"] is not None]
    return ok(
        num(max(devs)),
        max_abs_dev=num(max(devs)),
        max_abs_z=num(max(zs)) if zs else None,
        min_vr=num(min(row["vr"] for row in valid)),
        max_vr=num(max(row["vr"] for row in valid)),
        n_returns=int(total),
        n_sessions=int(n_rows),
        table=rows,
    )


def kurtosis_by_scale(
    r: np.ndarray,
    scales,
    step_seconds: float = 1.0,
    min_obs_for_gate: int = 10_000,
) -> dict:
    """集計スケール別の尖度 (正規で 3)。

    ``r`` は最細粒度のリターン。2 次元ならセッション内でのみ集計する。
    ``scales`` は秒で与え、``step_seconds`` で刻み数に換算する。
    """
    r2d = _as_2d(r)
    rows = []
    for scale_sec in scales:
        k = _scale_steps(float(scale_sec), step_seconds)
        if k is None:
            rows.append(
                {"scale_sec": float(scale_sec), "status": "not_applicable",
                 "reason": "刻みの整数倍ではありません", "kurtosis": None}
            )
            continue
        agg = _aggregate(r2d, k)
        if agg is None or agg.size < 20:
            rows.append(
                {"scale_sec": float(scale_sec), "status": "not_applicable",
                 "reason": "標本数不足", "kurtosis": None}
            )
            continue
        flat = agg.ravel()
        n = int(flat.size)
        kurt = float(stats.kurtosis(flat, fisher=False, bias=False))
        rows.append(
            {
                "scale_sec": float(scale_sec),
                "status": "ok",
                "n": n,
                "kurtosis": num(kurt),
                "se_under_normal": num(np.sqrt(24.0 / n)),
                "std": num(flat.std(ddof=1)),
                "gated": bool(n >= min_obs_for_gate),
            }
        )

    gated = [row for row in rows if row.get("gated")]
    if not gated:
        return na("ゲート判定に足る標本数のスケールがありません", table=rows)
    devs = [abs(row["kurtosis"] - 3.0) for row in gated]
    return ok(
        num(max(devs)),
        max_abs_dev_from_3_gated=num(max(devs)),
        min_kurtosis_gated=num(min(row["kurtosis"] for row in gated)),
        max_kurtosis_gated=num(max(row["kurtosis"] for row in gated)),
        n_gated_scales=len(gated),
        gated_scales_sec=[row["scale_sec"] for row in gated],
        min_obs_for_gate=int(min_obs_for_gate),
        table=rows,
    )


def zeta_q(
    r: np.ndarray,
    qs=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    scales=(1, 5, 30, 60, 300, 900),
    step_seconds: float = 1.0,
    min_obs_for_gate: int = 10_000,
) -> dict:
    """構造関数のスケーリング指数 ``zeta_q``。

    ``S_q(tau) = E[|r_tau|^q] ~ tau^{zeta_q}`` を両対数回帰で推定し、さらに
    ``zeta_q`` を q に回帰する。単一フラクタル (ブラウン運動) なら
    ``zeta_q = q/2`` の**直線**、多重フラクタルなら上に凸の曲線になる。

    q は 3 までに留める。高次モーメントは標本誤差が急速に大きくなり、S1 以降で
    テールが太くなると推定量そのものが不安定になるため、段階間で比較できる
    範囲に揃えておく。
    """
    r2d = _as_2d(r)
    qs = tuple(float(q) for q in qs)

    scale_rows = []
    moments: dict[float, list[tuple[float, float]]] = {q: [] for q in qs}
    for scale_sec in scales:
        k = _scale_steps(float(scale_sec), step_seconds)
        if k is None:
            continue
        agg = _aggregate(r2d, k)
        if agg is None or agg.size < 20:
            continue
        flat = np.abs(agg.ravel())
        n = int(flat.size)
        gated = n >= min_obs_for_gate
        row = {"scale_sec": float(scale_sec), "n": n, "gated": bool(gated)}
        for q in qs:
            value = float(np.mean(flat**q))
            row[f"S_{q}"] = num(value)
            if gated and value > 0:
                moments[q].append((float(scale_sec), value))
        scale_rows.append(row)

    zetas: list[float] = []
    q_used: list[float] = []
    per_q = []
    for q in qs:
        points = moments[q]
        if len(points) < 3:
            per_q.append({"q": q, "status": "not_applicable", "zeta": None})
            continue
        tau = np.log([p[0] for p in points])
        s = np.log([p[1] for p in points])
        slope, intercept, rvalue, _, stderr = stats.linregress(tau, s)
        per_q.append(
            {
                "q": q,
                "status": "ok",
                "zeta": num(slope),
                "se": num(stderr),
                "r2": num(rvalue**2),
                "n_scales": len(points),
                "expected_bm": num(q / 2.0),
            }
        )
        zetas.append(float(slope))
        q_used.append(q)

    if len(zetas) < 3:
        return na("zeta_q を推定できたモーメント次数が足りません", per_q=per_q, scales=scale_rows)

    slope, intercept, rvalue, _, stderr = stats.linregress(np.array(q_used), np.array(zetas))
    residual = np.array(zetas) - (intercept + slope * np.array(q_used))
    return ok(
        num(rvalue**2),
        r2=num(rvalue**2),
        slope=num(slope),
        slope_se=num(stderr),
        intercept=num(intercept),
        expected_slope_bm=0.5,
        max_abs_residual=num(np.max(np.abs(residual))),
        max_abs_dev_from_q_over_2=num(
            np.max(np.abs(np.array(zetas) - np.array(q_used) / 2.0))
        ),
        qs=list(q_used),
        zetas=[num(z) for z in zetas],
        per_q=per_q,
        scales=scale_rows,
    )


def signature_plot(
    log_price: np.ndarray,
    scales,
    step_seconds: float = 1.0,
    seconds_per_year: float | None = None,
    min_obs_for_gate: int = 10_000,
) -> dict:
    """signature plot: 単位時間あたり実現分散を集計スケールの関数として見る。

    マイクロストラクチャー・ノイズがあると細かいスケールで実現分散が跳ね上がる
    (有名な右下がりの曲線)。S0 にはノイズが無いので**平坦**になるのが正解であり、
    S9 で uncertainty zones を入れて初めて傾きが出るはずである。

    判定はゲート対象スケールの中央値からの最大相対乖離で行う。
    """
    lp = _as_2d(log_price)
    r2d = np.diff(lp, axis=1)
    rows = []
    for scale_sec in scales:
        k = _scale_steps(float(scale_sec), step_seconds)
        if k is None:
            continue
        agg = _aggregate(r2d, k)
        if agg is None or agg.size < 20:
            continue
        n = int(agg.size)
        covered_seconds = float(n) * float(scale_sec)
        rv_per_second = float((agg**2).sum() / covered_seconds)
        row = {
            "scale_sec": float(scale_sec),
            "n": n,
            "rv_per_second": num(rv_per_second),
            "gated": bool(n >= min_obs_for_gate),
        }
        if seconds_per_year:
            row["implied_annual_vol"] = num(np.sqrt(rv_per_second * seconds_per_year))
        rows.append(row)

    gated = [row for row in rows if row["gated"] and row["rv_per_second"]]
    if len(gated) < 2:
        return na("ゲート判定に足るスケールがありません", table=rows)
    values = np.array([row["rv_per_second"] for row in gated], dtype=np.float64)
    reference = float(np.median(values))
    rel_dev = np.abs(values / reference - 1.0)
    for row, dev in zip(gated, rel_dev):
        row["rel_dev"] = num(dev)
    # 傾き (両対数) も出しておく。S9 以降はここが負になるはず。
    slope, _, rvalue, _, stderr = stats.linregress(
        np.log([row["scale_sec"] for row in gated]), np.log(values)
    )
    return ok(
        num(np.max(rel_dev)),
        max_rel_dev=num(np.max(rel_dev)),
        reference_rv_per_second=num(reference),
        log_log_slope=num(slope),
        log_log_slope_se=num(stderr),
        log_log_r2=num(rvalue**2),
        n_gated_scales=len(gated),
        table=rows,
    )


# ---------------------------------------------------------------------------
def adf_test(x: np.ndarray, maxlag: int = 10, regression: str = "c") -> dict:
    """拡張 Dickey-Fuller 検定 (帰無仮説: 単位根がある)。

    ``autolag`` は使わない。情報量規準でラグを選ぶと標本や段階が変わるたびに
    選ばれるラグが変わり、段階間で比較できなくなるため、ラグは固定する。
    """
    y = np.asarray(x, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if y.size < 50:
        return na(f"標本数が足りません (n={y.size})")
    result = adfuller(y, maxlag=maxlag, regression=regression, autolag=None)
    stat, pvalue, usedlag, nobs = result[0], result[1], result[2], result[3]
    crit = result[4]
    return ok(
        num(pvalue),
        stat=num(stat),
        pvalue=num(pvalue),
        usedlag=int(usedlag),
        nobs=int(nobs),
        regression=regression,
        critical_values={str(k): num(v) for k, v in crit.items()},
    )


def adf_combined(log_price_result: dict, returns_result: dict, alpha: float = 0.01) -> dict:
    """log P とリターンの ADF 結果を 1 つの判定にまとめる。

    期待は「log P では単位根を棄却しない、リターンでは棄却する」。片方だけを見ると
    「そもそも系列が定数だった」ような壊れ方を見逃す。
    """
    if log_price_result.get("status") != "ok" or returns_result.get("status") != "ok":
        return na("片方または両方の ADF 検定が実行できませんでした")
    p_level = log_price_result["pvalue"]
    p_diff = returns_result["pvalue"]
    if p_level is None or p_diff is None:
        return na("ADF の p 値が取得できませんでした")
    level_ok = p_level > alpha
    diff_ok = p_diff < alpha
    return ok(
        bool(level_ok and diff_ok),
        combined_ok=bool(level_ok and diff_ok),
        alpha=float(alpha),
        log_price_pvalue=num(p_level),
        log_price_not_rejected=bool(level_ok),
        returns_pvalue=num(p_diff),
        returns_rejected=bool(diff_ok),
    )
