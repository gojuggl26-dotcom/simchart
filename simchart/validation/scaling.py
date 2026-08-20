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

import math

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
    "vol_variance_budget",
    "msm_diagnostics",
    "kurtosis_decay_fit",
    "zeta_curvature",
    "daily_invariance_stats",
    "roughness_exponent",
    "realized_variance",
    "path_stationarity",
    "skewness_by_scale",
    "spectral_peak",
    "marginal_normality",
    "cross_seed_correlation",
]


def skewness_by_scale(r_daily: np.ndarray, scales_days=(1, 2, 5, 10, 20), min_indep: int = 100) -> dict:
    """歪度の集計スケール依存 (S3)。

    非対称ジャンプ由来の歪度は集計で 1/sqrt(k) より速く減衰する (3 次キュムラント
    は加法だが分散も増えるため)。日次で負、集計で 0 へ向かうのが期待。
    """
    x = np.asarray(r_daily, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 500:
        return na(f"日次リターンが足りません (n={x.size})")
    cs = np.concatenate([[0.0], np.cumsum(x)])
    rows = []
    for scale in scales_days:
        k = int(scale)
        if x.size // k < min_indep:
            rows.append({"scale_days": k, "status": "not_applicable", "skewness": None})
            continue
        agg = cs[k:] - cs[:-k]
        rows.append(
            {"scale_days": k, "status": "ok",
             "skewness": num(float(stats.skew(agg, bias=False))),
             "n_independent": int(x.size // k)}
        )
    valid = [row for row in rows if row.get("skewness") is not None]
    if not valid:
        return na("有効なスケールがありません", table=rows)
    return ok(
        valid[0]["skewness"],
        skewness_daily=valid[0]["skewness"],
        skewness_coarsest=valid[-1]["skewness"],
        toward_zero=bool(abs(valid[-1]["skewness"]) <= abs(valid[0]["skewness"]) + 0.1),
        table=rows,
    )


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


# ---------------------------------------------------------------------------
# S1 追加: ボラ過程の診断
# ---------------------------------------------------------------------------
def vol_variance_budget(components: dict[str, np.ndarray] | None) -> dict:
    """log sigma の成分別分散シェアを返す (指示書 §6 の予算検証)。

    Parameters
    ----------
    components:
        成分名 -> log sigma への加法寄与の系列 (同一長)。例:
        ``{"msm": half_log_msm, "slow_ou": x_slow}``。

    Notes
    -----
    シェアは Var(成分)/Var(合計) で定義する。成分間が独立なら共分散 ~ 0 で
    シェアの和は 1 に近いが、有限標本では共分散項の分だけずれる。その残差も
    ``cross_share`` として返す (大きければ独立性が壊れている兆候)。

    **1 経路の時間平均は遅い成分のせいで大きくゆらぐ** (5000 日でも実効独立標本
    ~30、SD ≈ ±15%)。ゲート判定にはアンサンブル断面 (ensemble.py) を使い、
    こちらは経路の記録として扱うこと。
    """
    if not components:
        return na("ボラ成分がありません (enable_msm / enable_slow_ou が無効)")
    arrays = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in components.items()}
    n = {k: a.size for k, a in arrays.items()}
    if len(set(n.values())) != 1:
        return na(f"成分の長さが揃っていません: {n}")
    total = np.sum(list(arrays.values()), axis=0)
    var_total = float(total.var())
    if var_total <= 0:
        return na("log sigma の分散が 0 です")
    shares = {k: float(a.var() / var_total) for k, a in arrays.items()}
    return ok(
        num(var_total),
        var_total=num(var_total),
        component_vars={k: num(float(a.var())) for k, a in arrays.items()},
        shares={k: num(v) for k, v in shares.items()},
        cross_share=num(1.0 - sum(shares.values())),
        n_samples=int(next(iter(n.values()))),
    )


def msm_diagnostics(msm_meta: dict | None, horizon_days: float | None = None) -> dict:
    """MSM の実測切替率と占有率を、指定パラメータと突き合わせる。

    生成時に記録された成分別の切替回数 (Poisson) と m0 側占有率を検定する。
    切替回数は Poisson(gamma_i * T) なので z = (観測 - 期待)/sqrt(期待)。
    占有率の期待は 1/2 だが、遅い成分は実効標本が少なくゆらぎが大きいので、
    z は切替回数ベースの実効標本数で正規化する。

    これは**切替動学**の検証を担当する。合成式と正規化の検証はアンサンブル断面
    (ensemble.py)、経路の分散は :func:`vol_variance_budget` が担当し、三者で
    役割を分担している。
    """
    if not msm_meta:
        return na("MSM が無効です (enable_msm=False)")
    n_switches = msm_meta.get("n_switches")
    expected = msm_meta.get("expected_switches")
    occupancy = msm_meta.get("occupancy_hi")
    if not n_switches or not expected:
        return na("MSM の診断情報が不完全です", keys=sorted(msm_meta))

    rows = []
    max_abs_z = 0.0
    for i, (obs, exp) in enumerate(zip(n_switches, expected)):
        z = (obs - exp) / np.sqrt(exp) if exp > 0 else None
        if z is not None:
            max_abs_z = max(max_abs_z, abs(z))
        # 占有率の実効標本 ~ 区間数 (切替回数 + 1)。
        occ = occupancy[i] if occupancy else None
        occ_z = None
        if occ is not None and obs + 1 > 1:
            occ_z = (occ - 0.5) / (0.5 / np.sqrt(obs + 1))
        rows.append(
            {
                "component": i + 1,
                "n_switches": int(obs),
                "expected_switches": num(exp),
                "z": num(z) if z is not None else None,
                "occupancy_hi": num(occ) if occ is not None else None,
                "occupancy_z": num(occ_z) if occ_z is not None else None,
            }
        )
    return ok(
        num(max_abs_z),
        max_abs_switch_z=num(max_abs_z),
        k=msm_meta.get("k"),
        m0=num(msm_meta.get("m0")),
        horizon_days=num(horizon_days if horizon_days is not None else msm_meta.get("horizon_days")),
        table=rows,
    )


def kurtosis_decay_fit(r_daily: np.ndarray, scales_days=(1, 2, 5, 10, 20, 50), min_obs: int = 100) -> dict:
    """尖度の集計スケール依存 (集計正規性) を定量化する。

    日次リターンを ``scales_days`` 日で集計し、超過尖度 (kurt - 3) を両対数回帰する。
    ボラ変動があると細かいスケールほど尖度が高く、集計で 3 に近づく (減衰指数が負)。

    判定は 2 つの条件の AND:
    (1) log(超過尖度) の log-log 回帰傾きが負
    (2) 最細スケールの尖度 > 最粗 (gated) スケールの尖度
    厳密な単調減少を要求しないのは、粗いスケールは標本数が少なく尖度推定が
    ノイジーで、隣接ペアの逆転が乱数だけで起きるため。

    集計は**重なり窓** (rolling sum) で行う。非重複ブロックだと粗いスケールの
    標本数が n/k に落ちて集計ノイズが支配的になる。重なり窓は独立標本数こそ
    増やさないが、ブロック切りの位相という人工的なノイズ源を消し、モーメント
    推定を平滑化する。
    """
    r = np.asarray(r_daily, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 200:
        return na(f"日次リターンが足りません (n={r.size})")

    cs = np.concatenate([[0.0], np.cumsum(r)])
    rows = []
    points = []
    for scale in scales_days:
        k = int(scale)
        n_windows = r.size - k + 1
        n_independent = r.size // k
        if n_independent < 30:
            rows.append({"scale_days": k, "status": "not_applicable", "kurtosis": None})
            continue
        agg = cs[k:] - cs[:-k]  # 重なり窓の k 日リターン (n_windows 本)
        kurt = float(stats.kurtosis(agg, fisher=False, bias=False))
        gated = n_independent >= min_obs
        rows.append(
            {
                "scale_days": k,
                "status": "ok",
                "n_windows": int(n_windows),
                "n_independent": int(n_independent),
                "kurtosis": num(kurt),
                "excess": num(kurt - 3.0),
                "gated": bool(gated),
            }
        )
        if gated and kurt > 3.0:
            points.append((k, kurt - 3.0))

    gated_rows = [row for row in rows if row.get("gated")]
    if len(gated_rows) < 3:
        return na("ゲート判定に足るスケールがありません", table=rows)

    slope = None
    slope_se = None
    r2 = None
    if len(points) >= 3:
        lx = np.log([p[0] for p in points])
        ly = np.log([p[1] for p in points])
        fit = stats.linregress(lx, ly)
        slope, slope_se, r2 = float(fit.slope), float(fit.stderr), float(fit.rvalue**2)

    finest = gated_rows[0]["kurtosis"]
    coarsest = gated_rows[-1]["kurtosis"]
    decreasing = bool(
        slope is not None and slope < 0 and finest is not None and coarsest is not None
        and finest > coarsest
    )
    return ok(
        num(slope) if slope is not None else None,
        decay_slope=num(slope) if slope is not None else None,
        decay_slope_se=num(slope_se) if slope_se is not None else None,
        decay_r2=num(r2) if r2 is not None else None,
        kurtosis_finest=num(finest),
        kurtosis_coarsest_gated=num(coarsest),
        decreasing=decreasing,
        n_fit_points=len(points),
        table=rows,
    )


def zeta_curvature(
    r_daily: np.ndarray,
    qs=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
    scales_days=(1, 2, 5, 10, 20, 50, 100),
    min_independent: int = 40,
) -> dict:
    """zeta_q の曲率 (q への 2 次回帰の 2 次係数 c2) を推定する。

    マルチフラクタルなら zeta_q は上に凸で c2 < 0。線形回帰の R^2 は小さな曲率に
    鈍感 (S1 の予算では S0 と分離できない) ため、曲率を直接推定する。モーメントは
    重なり窓で、q は 4 まで、スケールは 1..100 日 — いずれも曲率の信号を最大化する
    選択 (それでも S1 の分離は確率的。決定的な判定は S2 で行う)。

    :func:`zeta_q` (非重複・q<=3・ゲート用) とは役割が違うので別関数として追加
    してある。既存関数は変更しない。
    """
    r = np.asarray(r_daily, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 500:
        return na(f"日次リターンが足りません (n={r.size})")

    cs = np.concatenate([[0.0], np.cumsum(r)])
    zetas: list[float] = []
    used_q: list[float] = []
    per_q = []
    for q in qs:
        pts = []
        for k in scales_days:
            k = int(k)
            if r.size // k < min_independent:
                continue
            agg = np.abs(cs[k:] - cs[:-k])
            moment = float(np.mean(agg ** float(q)))
            if moment > 0:
                pts.append((k, moment))
        if len(pts) < 4:
            per_q.append({"q": float(q), "status": "not_applicable", "zeta": None})
            continue
        lx = np.log([p[0] for p in pts])
        ly = np.log([p[1] for p in pts])
        fit = stats.linregress(lx, ly)
        zetas.append(float(fit.slope))
        used_q.append(float(q))
        per_q.append(
            {"q": float(q), "status": "ok", "zeta": num(fit.slope),
             "se": num(fit.stderr), "n_scales": len(pts)}
        )

    if len(zetas) < 4:
        return na("zeta を推定できたモーメント次数が足りません", per_q=per_q)

    qa = np.array(used_q)
    za = np.array(zetas)
    design = np.column_stack([np.ones_like(qa), qa, qa**2])
    coef, residuals, _, _ = np.linalg.lstsq(design, za, rcond=None)
    lin = stats.linregress(qa, za)
    return ok(
        num(float(coef[2])),
        c2=num(float(coef[2])),
        c1=num(float(coef[1])),
        c0=num(float(coef[0])),
        linear_r2=num(lin.rvalue**2),
        qs=list(used_q),
        zetas=[num(z) for z in zetas],
        per_q=per_q,
        scales_days=[int(s) for s in scales_days],
    )


def roughness_exponent(
    log_vol: np.ndarray,
    scales_steps,
    qs=(0.5, 1.0, 1.5, 2.0, 3.0),
    min_pairs: int = 1000,
) -> dict:
    """粗さ指数 H を q 次モーメントスケーリングで推定する (S2 の中心的検証)。

        m(q, Delta) = E[|log sigma_{t+Delta} - log sigma_t|^q] ∝ Delta^{qH}

    log m を log Delta に回帰して zeta_q^vol を得、zeta_q を q に回帰した傾きが H。
    zeta_q が q に線形 (R^2) であることが「単一フラクタルな粗さ」の確認になる。

    ★測定窓は**ラフ成分が支配する帯域** (既定 5 分〜4 時間) に限ること。GPH の
    測定窓 (1〜100 日) と重ねると互いに汚染し、どちらも信用できなくなる
    (S2 指示書 §7)。``scales_steps`` は入力系列のステップ数で与える。
    """
    scales_steps = [int(s) for s in scales_steps]
    if not scales_steps:
        return na("測定スケールがありません (サブサンプルが無い段階)")
    x = np.asarray(log_vol, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < min_pairs + max(scales_steps):
        return na(f"標本数が足りません (n={x.size})")

    qs = tuple(float(q) for q in qs)
    log_scales: list[float] = []
    moments: dict[float, list[float]] = {q: [] for q in qs}
    used_scales: list[int] = []
    for scale in scales_steps:
        k = int(scale)
        if k < 1 or x.size - k < min_pairs:
            continue
        d = np.abs(x[k:] - x[:-k])
        if not np.any(d > 0):
            return na("log sigma が定数です (増分が 0)")
        used_scales.append(k)
        log_scales.append(math.log(k))
        for q in qs:
            moments[q].append(float(np.mean(d**q)))

    if len(used_scales) < 4:
        return na(f"有効なスケールが足りません (n={len(used_scales)})")

    ls = np.array(log_scales)
    per_q = []
    zetas: list[float] = []
    for q in qs:
        m = np.array(moments[q])
        if np.any(m <= 0):
            per_q.append({"q": q, "status": "not_applicable", "zeta": None})
            continue
        fit = stats.linregress(ls, np.log(m))
        per_q.append(
            {
                "q": q,
                "status": "ok",
                "zeta": num(fit.slope),
                "se": num(fit.stderr),
                "r2": num(fit.rvalue**2),
                "h_implied": num(fit.slope / q),
            }
        )
        zetas.append(float(fit.slope))

    if len(zetas) < 3:
        return na("zeta_q^vol を推定できた次数が足りません", per_q=per_q)

    q_used = np.array([row["q"] for row in per_q if row["status"] == "ok"])
    z_used = np.array(zetas)
    hfit = stats.linregress(q_used, z_used)
    return ok(
        num(hfit.slope),
        h=num(hfit.slope),
        h_se=num(hfit.stderr),
        linearity_r2=num(hfit.rvalue**2),
        intercept=num(hfit.intercept),
        scales_steps=used_scales,
        per_q=per_q,
    )


def realized_variance(returns: np.ndarray, steps_per_window: int) -> np.ndarray:
    """実現分散 (窓ごとの二乗リターン和) の系列を返す (S2 追加)。

    素の配列を返すユーティリティ。H の RV 側推定と、S11 の RV フィードバックで
    再利用する。端数は捨てる。
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    k = int(steps_per_window)
    if k < 1:
        raise ValueError("steps_per_window は正整数である必要があります")
    n_windows = r.size // k
    if n_windows < 1:
        raise ValueError(f"窓 ({k} ステップ) に対してリターンが足りません (n={r.size})")
    return (r[: n_windows * k].reshape(n_windows, k) ** 2).sum(axis=1)


def path_stationarity(
    y: np.ndarray, adf_maxlag: int = 10, max_adf_points: int = 20_000
) -> dict:
    """経路の定常性検査 (S2 の stationarity_Y ゲート)。

    非定常な fBm/Volterra を混入させた場合の最も分かりやすい症状は「後半の分散が
    前半より大きい」ことなので、(1) 前半・後半の平均差 (自己相関補正つき z)、
    (2) 分散比、(3) ADF (帰無: 単位根あり) の 3 点で見る。ADF は実効標本を保った
    まま点数を間引いて行う (数百万点の OLS を避ける)。

    分散比の許容 [0.85, 1.18] は、半減期 ~1 日・5000 日の fOU で実現分散の
    片側 SD が ~4%、比の SD が ~6% であることに基づく (~2.7 sigma)。
    """
    x = np.asarray(y, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 1000:
        return na(f"標本数が足りません (n={n})")

    half = n // 2
    first, second = x[:half], x[half : 2 * half]
    rho1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    rho1 = min(max(rho1, -0.999), 0.999)
    n_eff = max(half * (1.0 - rho1) / (1.0 + rho1), 4.0)
    pooled_var = 0.5 * (float(first.var(ddof=1)) + float(second.var(ddof=1)))
    se_mean_diff = math.sqrt(2.0 * pooled_var / n_eff)
    mean_diff_z = (float(second.mean()) - float(first.mean())) / se_mean_diff

    var_ratio = float(second.var(ddof=1) / first.var(ddof=1))

    stride = max(n // max_adf_points, 1)
    adf = adf_test(x[::stride], maxlag=adf_maxlag)
    adf_p = adf.get("pvalue")

    checks = {
        "mean_stable": bool(abs(mean_diff_z) < 4.0),
        "variance_stable": bool(0.85 <= var_ratio <= 1.18),
        "adf_rejects_unit_root": bool(adf_p is not None and adf_p < 0.01),
    }
    return ok(
        bool(all(checks.values())),
        stationary=bool(all(checks.values())),
        checks=checks,
        mean_first=num(first.mean()),
        mean_second=num(second.mean()),
        mean_diff_z=num(mean_diff_z),
        var_first=num(first.var(ddof=1)),
        var_second=num(second.var(ddof=1)),
        var_ratio=num(var_ratio),
        lag1_autocorr=num(rho1),
        n_eff=num(n_eff),
        adf_pvalue=num(adf_p) if adf_p is not None else None,
        adf_stride=int(stride),
    )


def daily_invariance_stats(result) -> dict:
    """時間スケール不変性の比較に使う日次統計 (尖度・GPH d・|r|ACF(1)・Var(log sigma))。

    2 つの解像度の実行を**同じ汎関数**で測るための最小セット。Var(log sigma) は
    生成時の分単位サブサンプル (物理時刻グリッドが解像度に依らず共通) から取る。
    """
    from .memory import acf as acf_fn
    from .memory import gph_estimator
    from .tails import basic_moments

    obs = result.observation
    bars = obs.to_bars(obs.session_seconds)
    r_daily = bars.returns()
    abs_r = np.abs(r_daily)

    moments = basic_moments(r_daily)
    gph = gph_estimator(abs_r, bandwidth_exponent=result.config.validation.daily_gph_bandwidth_exponent)
    acf_abs = acf_fn(abs_r, max_lag=min(100, abs_r.size - 1))

    sub = result.meta.get("l2", {}).get("vol_subsample")
    var_log_vol = None
    if sub is not None:
        log_vol_sub = np.asarray(sub["log_vol"], dtype=np.float64)
        var_log_vol = float(log_vol_sub.var())
    else:
        var_log_vol = float(np.asarray(result.price.log_vol).var())

    return {
        "kurtosis_daily": moments.get("kurtosis"),
        "gph_d": gph.get("d"),
        "acf_abs_lag1": acf_abs.get("lag1"),
        "var_log_vol": var_log_vol,
        "n_days": int(r_daily.size),
    }


def spectral_peak(
    x: np.ndarray,
    sample_spacing_days: float,
    period_range_days: tuple[float, float] = (2.0, 1000.0),
    daily_band: tuple[float, float] = (0.8, 1.2),
    nperseg: int | None = None,
) -> dict:
    """スペクトルの主要ピーク位置 (市場日単位) と日周期帯のパワー比 (S5 §4.2)。

    χ₂ の写像先の検証に使う: 主要ピークが 20〜40 日にあり、S4 で苦労して除去した
    日内季節性の帯域 (0.8〜1.2 日) に**新規のピークを立てていない**こと。
    後者は「日周期帯のパワーシェア」で測る — MG は 30〜40 日スケールの滑らかな
    振動なので、正しく写像されていれば日周期帯のパワーは実質ゼロになる。
    """
    from scipy import signal as sp_signal

    y = np.asarray(x, dtype=np.float64).ravel()
    if y.shape[0] < 256:
        return na(f"系列が短すぎます (n={y.shape[0]})")
    fs = 1.0 / float(sample_spacing_days)  # サンプル/日
    seg = nperseg if nperseg is not None else min(1 << 14, y.shape[0])
    freqs, psd = sp_signal.welch(y - y.mean(), fs=fs, nperseg=seg)

    lo_f = 1.0 / period_range_days[1]
    hi_f = min(1.0 / period_range_days[0], fs / 2)
    mask = (freqs >= lo_f) & (freqs <= hi_f) & (freqs > 0)
    if mask.sum() < 8:
        return na("探索帯域に周波数点が足りません")
    fpk = float(freqs[mask][np.argmax(psd[mask])])
    total = float(psd[freqs > 0].sum())
    band = (freqs >= 1.0 / daily_band[1]) & (freqs <= 1.0 / daily_band[0])
    daily_share = float(psd[band].sum()) / total if total > 0 and band.any() else 0.0
    # ピーク近傍への集中度 (±40%)
    near = (freqs >= fpk * 0.7) & (freqs <= fpk * 1.4)
    concentration = float(psd[near].sum()) / total if total > 0 else float("nan")

    return ok(
        num(1.0 / fpk),
        peak_period_days=num(1.0 / fpk),
        peak_frequency_per_day=num(fpk),
        daily_band_power_share=num(daily_share),
        concentration_pm40pct=num(concentration),
        resolvable_period_max_days=num(seg / fs),
        n_used=int(y.shape[0]),
    )


def marginal_normality(x: np.ndarray, mode_prominence: float = 0.05) -> dict:
    """周辺分布の形 (S5 §3.2 のチェック)。

    カオスアトラクタの周辺分布は有界でしばしば多峰。log σ に双峰成分を足すと
    log RV の分布が双峰化し、実証 (log RV はおおむね正規) と乖離する。ゲートは
    **合成後の log σ** に対して |超過尖度| < 1 かつ単峰を要求する (χ₂ 単体は
    多峰でもよい — 分散比 1:4 のガウス成分との合成で滑らかになるため)。

    山の数は KDE の局所最大 (最大密度の ``mode_prominence`` 倍を超えるもの) で
    数える。厳密な多峰性検定 (dip test) ではないが、ゲートの目的 (双峰化の検出)
    には十分で、閾値が明示されている分だけ再現しやすい。
    """
    y = np.asarray(x, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if y.shape[0] < 500:
        return na(f"標本が足りません (n={y.shape[0]})")
    z = (y - y.mean()) / y.std()
    sub = z[:: max(y.shape[0] // 20000, 1)]
    kde = stats.gaussian_kde(sub)
    grid = np.linspace(float(z.min()), float(z.max()), 512)
    dens = kde(grid)
    peaks = (
        (dens[1:-1] > dens[:-2])
        & (dens[1:-1] > dens[2:])
        & (dens[1:-1] > dens.max() * mode_prominence)
    )
    return ok(
        num(float(stats.kurtosis(y, fisher=True, bias=False))),
        excess_kurtosis=num(float(stats.kurtosis(y, fisher=True, bias=False))),
        skewness=num(float(stats.skew(y, bias=False))),
        n_modes=int(peaks.sum()),
        unimodal=bool(int(peaks.sum()) == 1),
        mode_prominence=float(mode_prominence),
        n=int(y.shape[0]),
    )


def cross_seed_correlation(paths: list[np.ndarray]) -> dict:
    """シード横断相関 (S5 §8 — S5 の中核ゲート)。

    χ₂ は決定論的なので全シードで同一。log σ = D(t) [決定論] + S_i(t) [シード固有]
    と書けるから、シード i≠j の同時刻相関は

        corr(log σ_i, log σ_j) = Var(D) / sqrt((Var(D)+Var(S_i))(Var(D)+Var(S_j)))

    となり、**内部状態に一切アクセスせずに** χ₂ の分散シェアを推定できる。
    呼び出し側は φ_σ (これも決定論成分) を**除去してから**渡すこと — 残すと
    φ の分散が分子に混ざり、シェアが過大に見える。

    45 対 (10 シード) の平均・範囲を返す。ペアは独立でない (シードごとの経路分散が
    複数ペアに共有される) ので、範囲も見て 1 ペアの外れに引きずられていないかを
    確認できるようにする。
    """
    if len(paths) < 2:
        return na(f"シードが足りません (n={len(paths)})")
    n = min(p.shape[0] for p in paths)
    mat = np.stack([np.asarray(p[:n], dtype=np.float64) for p in paths])
    c = np.corrcoef(mat)
    iu = np.triu_indices(len(paths), k=1)
    vals = c[iu]
    return ok(
        num(float(vals.mean())),
        mean=num(float(vals.mean())),
        median=num(float(np.median(vals))),
        min=num(float(vals.min())),
        max=num(float(vals.max())),
        n_pairs=int(vals.size),
        n_seeds=int(len(paths)),
        n_time_points=int(n),
        per_seed_variance=[num(float(v)) for v in mat.var(axis=1)],
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
