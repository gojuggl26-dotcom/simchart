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

import numpy as np
from scipy import stats

from .base import na, num, ok

__all__ = ["basic_moments", "hill_estimator", "hill_profile", "qq_normal"]


def _clean(r: np.ndarray) -> np.ndarray:
    arr = np.asarray(r, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def basic_moments(r: np.ndarray) -> dict:
    """平均・標準偏差・歪度・尖度と Jarque-Bera 検定。

    尖度は **excess ではない**値 (正規分布で 3) を ``value`` に入れる。指示書 §8 の
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
