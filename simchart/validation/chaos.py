"""S5: カオス性の検証 (Lyapunov / 相関次元 / 0-1 test / BDS)。

役割の整理 (S5 設計要件)
--------------------------
これらの推定器は 3 つの対象に、期待される結果が違うまま適用される:

- χ₂ 単体: 正の Lyapunov・低い相関次元 (~2.1)・K ≈ 1 → critical ゲート
- 合成 log σ: 確率成分 (分散比 4:1) に埋もれて検出困難 → 記録のみ
- 観測価格: 検出不能 → 記録のみ

最後の 2 つは欠陥ではない。実データから低次元カオスが検出されないという実証結果と
整合的であり、カオスの価値は統計的識別可能性ではなく**決定論的な再現性と制御可能な
レジーム構造**にある。「価格からカオスが検出できないから失敗」と判断しないこと。

推定器についての注意
--------------------
- Rosenstein も GP も決定論的データを前提とする。確率過程に当てると Rosenstein
  は偽の正の傾きを出す (近傍が拡散で離れるため)。合成系列への適用値は
  実データ解析でこう見えるという記録であって、カオスの証拠でも反証でもない
- MG(17) は高コヒーレントな振動子 (実測: 周期 ~49.7 単位にパワーの 84%、
  位相相関は 8 周期後も 0.77)。埋め込み遅延はピーク周期の 1/4 程度、Theiler 窓は
  2 周期以上を要する
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .base import na, num, ok

__all__ = [
    "lyapunov_rosenstein",
    "correlation_dimension",
    "test_0_1_chaos",
    "bds_test",
]


def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """遅延埋め込み。行 = 時刻、列 = (x_t, x_{t-tau}, ..., x_{t-(m-1)tau})。"""
    n = x.shape[0] - (m - 1) * tau
    if n < 10:
        raise ValueError(f"埋め込み後の点数が足りません (n={n})")
    idx = np.arange(n)[:, None] + np.arange(m)[None, :] * tau
    return x[idx]


def _acf_first_below(x: np.ndarray, threshold: float = 1.0 / math.e, max_lag: int = 2000) -> int:
    """自己相関が閾値を最初に割るラグ (埋め込み遅延の既定値に使う)。"""
    d = x - x.mean()
    denom = float(d @ d)
    for lag in range(1, min(max_lag, x.shape[0] // 2)):
        if float(d[:-lag] @ d[lag:]) / denom < threshold:
            return lag
    return max_lag


def lyapunov_rosenstein(
    x: np.ndarray,
    dt: float,
    m: int = 5,
    tau_embed: int | None = None,
    theiler: int | None = None,
    horizon_units: float = 400.0,
    fit_range_units: tuple[float, float] = (10.0, 250.0),
    max_points: int = 20000,
) -> dict:
    """Rosenstein 法による最大 Lyapunov 指数 (1/時間単位)。

    各埋め込み点の最近傍 (時間的に ``theiler`` 以上離れたもの) を取り、両軌道の
    距離の対数の平均 ⟨ln d(i)⟩ を先読み時間 i について描き、初期の線形領域の傾きを
    λ とする。決定論なら距離は e^{λt} で伸びるので傾きが λ になる。

    MG(17) の文献値は ~0.006/単位。飽和 (アトラクタ直径への到達) は
    ln(直径/初期距離) ≈ 7 を λ で割った ~1000 単位あたりなので、既定の
    フィット窓 [10, 250] は線形領域に収まる。
    """
    y = np.asarray(x, dtype=np.float64).ravel()
    stride = max(int(math.ceil(y.shape[0] / max_points)), 1)
    y = y[::stride]
    dt_eff = dt * stride
    n = y.shape[0]
    if n < 2000:
        return na(f"系列が短すぎます (n={n})")

    tau_e = tau_embed if tau_embed is not None else max(_acf_first_below(y) // 1, 1)
    the = theiler if theiler is not None else max(int(round(100.0 / dt_eff)), tau_e * m)
    emb = _embed((y - y.mean()) / y.std(), m, tau_e)
    n_emb = emb.shape[0]
    horizon = int(round(horizon_units / dt_eff))
    usable = n_emb - horizon
    if usable < 500:
        return na(f"先読み分を除いた点数が足りません (usable={usable})")

    from scipy.spatial import cKDTree

    tree = cKDTree(emb[:usable])
    # Theiler 窓内の点を最近傍候補から外すため k を多めに取る。
    k = max(2 * the // max(tau_e, 1), 8)
    dists, idxs = tree.query(emb[:usable], k=min(k, usable))
    nn = np.full(usable, -1, dtype=np.int64)
    for col in range(1, idxs.shape[1]):
        cand = idxs[:, col]
        okmask = (nn < 0) & (np.abs(cand - np.arange(usable)) > the)
        nn[okmask] = cand[okmask]
    valid = nn >= 0
    ii = np.arange(usable)[valid]
    jj = nn[valid]
    if ii.size < 200:
        return na("Theiler 窓を満たす近傍対が足りません")

    steps = np.arange(0, horizon + 1, max(int(round(1.0 / dt_eff)), 1))
    mean_log_d = np.empty(steps.shape[0], dtype=np.float64)
    for si, h in enumerate(steps):
        diff = emb[ii + h] - emb[jj + h]
        d = np.sqrt((diff * diff).sum(axis=1))
        mean_log_d[si] = float(np.log(np.maximum(d, 1e-300)).mean())

    t_units = steps * dt_eff
    lo, hi = fit_range_units
    fmask = (t_units >= lo) & (t_units <= hi)
    if fmask.sum() < 5:
        return na("フィット窓に点が足りません")
    slope, intercept = np.polyfit(t_units[fmask], mean_log_d[fmask], 1)
    resid = mean_log_d[fmask] - (slope * t_units[fmask] + intercept)
    ss_tot = float(((mean_log_d[fmask] - mean_log_d[fmask].mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    return ok(
        num(slope),
        lyapunov_per_unit=num(slope),
        fit_r2=num(r2),
        tau_embed=int(tau_e),
        theiler=int(the),
        m=int(m),
        n_pairs=int(ii.size),
        dt_effective=float(dt_eff),
        fit_range_units=[float(lo), float(hi)],
        curve_t_units=t_units,
        curve_mean_log_d=mean_log_d,
    )


def correlation_dimension(
    x: np.ndarray,
    dt: float,
    m_values: Sequence[int] = (3, 4, 5, 6),
    tau_embed: int | None = None,
    theiler_units: float = 100.0,
    n_points: int = 5000,
    n_radii: int = 24,
) -> dict:
    """Grassberger-Procaccia 相関次元 D₂。

    相関積分 C(r) = (Theiler 窓外の点対のうち距離 ≤ r の割合) の log-log 傾きを
    スケーリング領域で測る。埋め込み次元 m を上げても傾きが飽和する値が D₂。
    MG(17) の文献値は ~2.1。

    スケーリング領域は距離分布の [2%, 20%] 分位で固定する。下限を下げすぎると
    点数不足のノイズ、上限を上げすぎるとアトラクタの縁の効果 (傾きの低下) を拾う。
    """
    y = np.asarray(x, dtype=np.float64).ravel()
    stride = max(int(math.ceil(y.shape[0] / (n_points + 2000))), 1)
    y = y[::stride]
    dt_eff = dt * stride
    tau_e = tau_embed if tau_embed is not None else max(_acf_first_below(y), 1)
    the = max(int(round(theiler_units / dt_eff)), 1)

    from scipy.spatial import cKDTree

    per_m: list[dict] = []
    slopes: list[float] = []
    for m in m_values:
        try:
            emb = _embed((y - y.mean()) / y.std(), int(m), tau_e)
        except ValueError as exc:
            return na(str(exc))
        n = emb.shape[0]
        if n > n_points:
            emb = emb[: n_points]
            n = n_points
        tree = cKDTree(emb)
        # 距離分布の分位からスケーリング領域を決める (小標本で近似)。
        sample_idx = np.linspace(0, n - 1, 400).astype(int)
        sd = np.sqrt(((emb[sample_idx, None, :] - emb[None, sample_idx, :]) ** 2).sum(-1))
        iu = np.triu_indices(sample_idx.size, k=1)
        keep = np.abs(sample_idx[iu[0]] - sample_idx[iu[1]]) > the
        pool = sd[iu][keep]
        if pool.size < 50:
            # Theiler 窓が系列長に対して大きすぎる (単位の取り違え等)。
            return na(
                f"Theiler 窓 ({the} サンプル) が系列長 ({n}) に対して大きすぎ、"
                f"時間的に独立な点対が取れません"
            )
        r_lo, r_hi = np.quantile(pool, [0.02, 0.20])
        radii = np.geomspace(max(r_lo, 1e-12), r_hi, n_radii)

        pairs = tree.query_pairs(r=float(radii[-1]), output_type="ndarray")
        if pairs.shape[0] < 100:
            per_m.append({"m": int(m), "status": "insufficient_pairs"})
            continue
        tmask = np.abs(pairs[:, 0] - pairs[:, 1]) > the
        pd = np.sqrt(((emb[pairs[tmask, 0]] - emb[pairs[tmask, 1]]) ** 2).sum(axis=1))
        if pd.size < 100:
            per_m.append({"m": int(m), "status": "insufficient_pairs"})
            continue
        counts = np.searchsorted(np.sort(pd), radii, side="right").astype(np.float64)
        cmask = counts > 10
        if cmask.sum() < 6:
            per_m.append({"m": int(m), "status": "insufficient_scaling_points"})
            continue
        slope, _ = np.polyfit(np.log(radii[cmask]), np.log(counts[cmask]), 1)
        per_m.append(
            {
                "m": int(m),
                "status": "ok",
                "slope": num(slope),
                "n_pairs": int(pd.size),
                "r_range": [num(radii[cmask][0]), num(radii[cmask][-1])],
            }
        )
        slopes.append(float(slope))

    if len(slopes) < 2:
        return na("傾きを推定できた埋め込み次元が足りません", per_m=per_m)
    # 埋め込み飽和後の値 = 上位 2 つの m の平均。
    d2 = float(np.mean(slopes[-2:]))
    return ok(
        num(d2),
        d2=num(d2),
        slopes_by_m=slopes,
        per_m=per_m,
        tau_embed=int(tau_e),
        theiler=int(the),
        saturated=bool(abs(slopes[-1] - slopes[-2]) < 0.4),
    )


def test_0_1_chaos(
    x: np.ndarray,
    subsample: int = 1,
    n_c: int = 64,
    n_cut_frac: float = 0.1,
) -> dict:
    """Gottwald-Melbourne の 0-1 test (修正版 D(n) を使用)。

    K ≈ 1 がカオス、K ≈ 0 が規則運動。translation 変数 (p, q) の平均二乗変位が
    カオスなら拡散的 (線形成長)、規則的なら有界になることを使う。

    - ``c`` は決定論的な固定格子 (乱数で引かない — このモジュールの系全体が
      乱数を消費しない規約のため)。共鳴を避けて [0.3π, 0.7π] に置く
    - 連続系のオーバーサンプリングは K を下方に偏らせるので、呼び出し側は
    特徴周期の 1/8 程度に間引いて渡すこと (``subsample``)
    """
    y = np.asarray(x, dtype=np.float64).ravel()[:: max(int(subsample), 1)]
    n = y.shape[0]
    if n < 500:
        return na(f"系列が短すぎます (n={n})")
    n_cut = max(int(n * n_cut_frac), 20)
    ns = np.arange(1, n_cut + 1)
    ybar = float(y.mean())

    cs = np.linspace(0.3 * math.pi, 0.7 * math.pi, int(n_c))
    ks: list[float] = []
    j = np.arange(n, dtype=np.float64)
    for c in cs:
        ang = c * j
        p = np.cumsum(y * np.cos(ang))
        q = np.cumsum(y * np.sin(ang))
        msd = np.empty(n_cut, dtype=np.float64)
        for i, h in enumerate(ns):
            dp = p[h:] - p[:-h]
            dq = q[h:] - q[:-h]
            msd[i] = float((dp * dp + dq * dq).mean())
        # 修正項: 振動成分を除去して収束を速める (Gottwald-Melbourne 2009)。
        corr_term = ybar**2 * (1.0 - np.cos(ns * c)) / (1.0 - math.cos(c))
        d = msd - corr_term
        # K_c = corr(n, D(n))
        if d.std() > 0:
            ks.append(float(np.corrcoef(ns, d)[0, 1]))
    if not ks:
        return na("K_c を計算できませんでした")
    return ok(
        num(float(np.median(ks))),
        K=num(float(np.median(ks))),
        K_iqr=num(float(np.subtract(*np.percentile(ks, [75, 25])))),
        n_c=int(n_c),
        n_used=int(n),
        n_cut=int(n_cut),
    )


# pytest が名前で誤収集しないための印 (関数名を保つため改名しない)。
test_0_1_chaos.__test__ = False  # type: ignore[attr-defined]


def bds_test(x: np.ndarray, max_dim: int = 4, max_points: int = 20000) -> dict:
    """BDS 検定 (iid の帰無仮説)。statsmodels 実装のラッパ。

    棄却 = カオスではない。BDS は iid からのあらゆる逸脱 (確率ボラ・
    自己相関・非線形依存すべて) に反応する。価格リターンで棄却されるのは確率ボラ
    だけで十分説明でき、カオスの証拠にはならない — 記録のみの位置づけはそのため。
    """
    from statsmodels.tsa.stattools import bds

    y = np.asarray(x, dtype=np.float64).ravel()
    stride = max(int(math.ceil(y.shape[0] / max_points)), 1)
    y = y[::stride]
    if y.shape[0] < 300:
        return na(f"系列が短すぎます (n={y.shape[0]})")
    stat, pvalue = bds(y, max_dim=int(max_dim))
    return ok(
        num(float(np.atleast_1d(stat)[-1])),
        statistics=np.atleast_1d(stat),
        p_values=np.atleast_1d(pvalue),
        dims=list(range(2, int(max_dim) + 1)),
        n_used=int(y.shape[0]),
        stride=int(stride),
    )
