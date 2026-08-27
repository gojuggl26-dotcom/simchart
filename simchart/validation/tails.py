"""テール関連の検証。

S0 での期待値
-------------
革新項は正規なので、Hill 推定量は「大きく、k の取り方で不安定」になるのが正解。
実データのように alpha ~ 3〜4 で安定するのは S1〜S3 でボラ過程とジャンプを
入れてからであって、**S0 でそれが出たら革新項にファットテールを混ぜてしまった
という事故**である。QQ プロットは直線になる。どちらも記録のみでゲートにはしない
(Hill 推定量は正規標本に対して発散気味に振る舞うため、閾値を置く意味がない)。
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .base import na, num, ok

__all__ = [
    "basic_moments",
    "hill_estimator",
    "hill_profile",
    "qq_normal",
    "bns_jump_test",
    "hill_by_scale",
]


def bns_jump_test(r: np.ndarray, steps_per_window: int) -> dict:
    """Barndorff-Nielsen & Shephard の bipower variation によるジャンプ検出 (S3)。

    窓 (通常 1 日) ごとに RV = sum r^2 と BV = (pi/2) sum |r_i||r_{i-1}| を計算する。
    BV は連続部分の QV に一致的でジャンプに頑健なので、JV share は全期間集計の
    ``max(0, 1 - sum BV / sum RV)`` で推定する。窓別の z 統計
    (RV-BV)/sqrt(theta TQ / n) の有意窓割合も返す (記録)。
    """
    x = np.asarray(r, dtype=np.float64).ravel()
    k = int(steps_per_window)
    n_w = x.size // k
    if n_w < 30:
        return na(f"窓が足りません (n_windows={n_w})")
    panel = x[: n_w * k].reshape(n_w, k)
    rv = (panel**2).sum(axis=1)
    absr = np.abs(panel)
    bv = (np.pi / 2.0) * (absr[:, 1:] * absr[:, :-1]).sum(axis=1)
    # tripower quarticity (z 統計の分母)。mu_{4/3} = 2^{2/3} Gamma(7/6)/Gamma(1/2)
    mu_43 = 2 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.gamma(0.5)
    p43 = absr ** (4.0 / 3.0)
    tq = k * mu_43**-3 * (p43[:, 2:] * p43[:, 1:-1] * p43[:, :-2]).sum(axis=1)
    theta = (np.pi**2 / 4.0) + np.pi - 5.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (1.0 - bv / rv) / np.sqrt(theta * np.maximum(tq / np.maximum(bv, 1e-300) ** 2, 1.0 / k))
    total_rv = float(rv.sum())
    total_bv = float(bv.sum())
    jv_share = max(0.0, 1.0 - total_bv / total_rv) if total_rv > 0 else None
    sig = float(np.mean(z > 3.09)) if np.all(np.isfinite(z)) else None  # p<0.001 片側
    return ok(
        num(jv_share),
        jv_share=num(jv_share),
        total_rv=num(total_rv),
        total_bv=num(total_bv),
        n_windows=int(n_w),
        frac_windows_significant=num(sig) if sig is not None else None,
        mean_z=num(float(np.nanmean(z))),
    )


def hill_by_scale(r_daily: np.ndarray, scales_days=(1, 2, 5, 10, 20), k_frac: float = 0.05) -> dict:
    """Hill α の集計スケール依存 (S3 — (18) の定量化)。

    指数テールのジャンプ + ボラ混合では α がスケールとともに上昇する
    (集計でテールが相対的に薄れる)。べき則ジャンプだと α が不変になるので、
    その識別でもある。判定は「隣接スケールで非減少 (小さな逆転は許容) かつ
    最粗 > 最細」ではなく、単調性を回帰傾きで見る (ノイズ耐性)。
    """
    x = np.asarray(r_daily, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 500:
        return na(f"日次リターンが足りません (n={x.size})")
    cs = np.concatenate([[0.0], np.cumsum(x)])
    rows = []
    alphas = []
    used = []
    for scale in scales_days:
        k = int(scale)
        agg = cs[k:] - cs[:-k]  # 重なり窓
        n_indep = x.size // k
        res = hill_estimator(agg, k_frac=k_frac, tail="both")
        if res["status"] != "ok" or n_indep < 100:
            rows.append({"scale_days": k, "status": "not_applicable", "alpha": None})
            continue
        rows.append(
            {"scale_days": k, "status": "ok", "alpha": res["alpha"],
             "se": res["se_alpha"], "n_independent": int(n_indep)}
        )
        alphas.append(float(res["alpha"]))
        used.append(k)
    if len(alphas) < 3:
        return na("有効なスケールが足りません", table=rows)
    fit = stats.linregress(np.log(used), alphas)
    increasing = bool(fit.slope > 0 and alphas[-1] > alphas[0])
    return ok(
        num(fit.slope),
        alpha_daily=num(alphas[0]),
        alpha_coarsest=num(alphas[-1]),
        slope_vs_log_scale=num(fit.slope),
        slope_se=num(fit.stderr),
        increasing=increasing,
        k_frac=float(k_frac),
        table=rows,
    )


def _clean(r: np.ndarray) -> np.ndarray:
    arr = np.asarray(r, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def basic_moments(r: np.ndarray) -> dict:
    """平均・標準偏差・歪度・尖度と Jarque-Bera 検定。

    尖度は excess ではない値 (正規分布で 3) を ``value`` に入れる。
    ゲートが 2.7〜3.3 という表現なのでそれに合わせる。excess も併記する。
    """
    x = _clean(r)
    n = x.size
    if n < 10:
        return na(f"標本数が足りません (n={n})")
    kurt = float(stats.kurtosis(x, fisher=False, bias=False))
    skew = float(stats.skew(x, bias=False))
    jb_stat, jb_p = stats.jarque_bera(x)
    return ok(
        num(kurt),
        n=int(n),
        mean=num(x.mean()),
        std=num(x.std(ddof=1)),
        skewness=num(skew),
        kurtosis=num(kurt),
        excess_kurtosis=num(kurt - 3.0),
        # 正規標本での標本尖度の標準誤差。ゲート閾値が有効かどうかの判断材料。
        kurtosis_se_under_normal=num(np.sqrt(24.0 / n)),
        skewness_se_under_normal=num(np.sqrt(6.0 / n)),
        jarque_bera_stat=num(jb_stat),
        jarque_bera_pvalue=num(jb_p),
    )


def hill_estimator(r: np.ndarray, k_frac: float = 0.05, tail: str = "both") -> dict:
    """Hill 推定量によるテール指数 alpha の推定。

    Parameters
    ----------
    r:
        リターン系列。
    k_frac:
        上位何割を裾とみなすか。
    tail:
        ``"both"`` (|r|) / ``"right"`` (r>0) / ``"left"`` (r<0 の絶対値)。

    Notes
    -----
    ``alpha_hat = 1 / mean(log(x_(i) / x_(k+1)))`` (x は降順ソート、i=1..k)。
    標準誤差は漸近的に ``alpha / sqrt(k)``。**正規標本に対しては真の alpha が
    存在しない (テールが冪則でない) ため、推定値は k とともに増加し安定しない。**
    S0 でこれが安定していたら革新項を疑うこと。
    """
    x = _clean(r)
    if tail == "both":
        x = np.abs(x)
    elif tail == "right":
        x = x[x > 0]
    elif tail == "left":
        x = -x[x < 0]
    else:
        raise ValueError(f"tail は 'both' / 'right' / 'left' のいずれかです: {tail!r}")

    x = x[x > 0]
    n = x.size
    if not (0.0 < k_frac < 1.0):
        raise ValueError("k_frac は (0, 1) の範囲である必要があります")
    k = int(n * k_frac)
    if k < 10 or k >= n:
        return na(f"裾の標本数が足りません (n={n}, k={k})", n=int(n), k=int(k), tail=tail)

    order = np.argpartition(x, n - k - 1)[n - k - 1 :]
    top = np.sort(x[order])[::-1]  # 上位 k+1 個を降順に
    threshold = top[k]
    if threshold <= 0:
        return na("裾の閾値が 0 以下です", n=int(n), k=int(k), tail=tail)
    hill = float(np.mean(np.log(top[:k] / threshold)))
    if hill <= 0:
        return na("Hill 統計量が非正です (同値の集中)", n=int(n), k=int(k), tail=tail)
    alpha = 1.0 / hill
    return ok(
        num(alpha),
        alpha=num(alpha),
        xi=num(hill),
        se_alpha=num(alpha / np.sqrt(k)),
        k=int(k),
        k_frac=float(k_frac),
        n=int(n),
        threshold=num(threshold),
        tail=tail,
    )


def hill_profile(r: np.ndarray, k_fracs, tail: str = "both") -> dict:
    """複数の ``k_frac`` で Hill 推定量を並べ、安定性を見る。

    正規標本では alpha が k とともに単調に動く。冪則テールなら平坦域が出る。
    ``instability`` は推定値の (最大 - 最小) / 中央値。
    """
    rows = []
    for kf in k_fracs:
        res = hill_estimator(r, k_frac=float(kf), tail=tail)
        rows.append({"k_frac": float(kf), "status": res["status"], "alpha": res.get("alpha")})
    alphas = [row["alpha"] for row in rows if row["alpha"] is not None]
    if not alphas:
        return na("有効な Hill 推定値がありません", profile=rows, tail=tail)
    med = float(np.median(alphas))
    instability = (max(alphas) - min(alphas)) / med if med > 0 else None
    return ok(
        num(med),
        median_alpha=num(med),
        min_alpha=num(min(alphas)),
        max_alpha=num(max(alphas)),
        instability=num(instability) if instability is not None else None,
        profile=rows,
        tail=tail,
    )


def qq_normal(r: np.ndarray, n_points: int = 201) -> dict:
    """標準化リターンの正規 QQ。直線からのずれを数値で返す。

    ``r2`` は経験分位点を理論分位点に回帰した決定係数、``max_abs_dev`` は
    標準偏差単位での最大乖離。プロット用に分位点そのものも返す。
    """
    x = _clean(r)
    n = x.size
    if n < 100:
        return na(f"標本数が足りません (n={n})")
    z = (x - x.mean()) / x.std(ddof=1)
    probs = (np.arange(1, n_points + 1) - 0.5) / n_points
    theoretical = stats.norm.ppf(probs)
    empirical = np.quantile(z, probs)

    slope, intercept, rvalue, _, stderr = stats.linregress(theoretical, empirical)
    deviation = empirical - (intercept + slope * theoretical)
    return ok(
        num(rvalue**2),
        r2=num(rvalue**2),
        slope=num(slope),
        intercept=num(intercept),
        slope_stderr=num(stderr),
        max_abs_dev=num(np.max(np.abs(deviation))),
        n=int(n),
        probs=probs.tolist(),
        theoretical_quantiles=theoretical.tolist(),
        empirical_quantiles=empirical.tolist(),
    )
