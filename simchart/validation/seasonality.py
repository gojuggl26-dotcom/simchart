"""日内季節性の測定と除去 (S4 の中核成果物)。

なぜこれが S4 の本当の成果物なのか
----------------------------------
日内 U 字は**それ自体が観測可能な統計性質を大量に偽造する**。除去しないと:

- ``|r|`` の自己相関に日次周期のピークが立ち、ラグ方向に減衰する成分と混ざる。
  GPH や Local Whittle は低周波側の勾配を見るので、周期成分の漏れ (leakage) が
  長期記憶の推定値 ``d`` を汚す。
- S7 の Hawkes 分岐比 ``n`` が**系統的に過大推定**される (Filimonov-Sornette 2015)。
  活発な時間帯へのイベント集中は「外生強度の時間変化」なのに、季節性を入れずに
  当てると自己励起として説明されてしまう。臨界 (n→1) の誤検出はこの経路で起きる。
- 日内ボラの起伏そのものがマルチフラクタルの見かけを作る。

したがって S4 で追加すべきものは「季節性を入れた」ことではなく、
**「季節性を入れても、除去すれば S1〜S3 の構造がそのまま出てくる」ことを
示せる道具**である。ゲートはその除去の効き目を測る。

2 経路を必ず両方持つ
--------------------
1. **真値経路** (``true_phi_bars``): シミュレータは φ を知っているので、除去は
   厳密にできる。「除去したら S3 に戻る」という強い主張はこちらで検定する。
2. **推定経路** (``estimate_phi``): 実務では φ は未知で、データから推定する。
   推定誤差がどれだけ残差を汚すかを測る。S7 で実データに向かうときに効くのは
   こちら。真値経路だけを持っていると「道具として使えるか」が判らない。

推定は標本内 (in-sample) である
-------------------------------
φ̂ は全標本の日内平均から作る。これは記述統計 (Andersen-Bollerslev 1997) の
標準手順であり、**予測ではない**ので同時性の問題は生じない。ただし脱季節化した
系列を将来リターンの予測に使うなら、φ̂ は訓練期間だけから推定すること
(``fit_days`` 引数で期間を限れる)。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .base import na, ok

__all__ = [
    "bin_centers",
    "intraday_profile",
    "estimate_phi",
    "true_phi_bars",
    "deseasonalize",
    "phi_recovery",
    "phi_normalization_check",
    "acf_periodicity_test",
    "spectral_periodicity_test",
    "profile_flatness",
    "overnight_stats",
    "close_to_close_returns",
    "time_change_by_phi_lambda",
]

#: 正規分布の 0.75 分位。median(|z|) = 0.6745 sigma の逆数変換に使う。
_MAD_TO_SIGMA = 1.0 / 0.6744897501960817


# ---------------------------------------------------------------------------
# 日内プロファイルの測定
# ---------------------------------------------------------------------------
def bin_centers(n_bars: int) -> np.ndarray:
    """バー k が覆う ``u in [k/B, (k+1)/B)`` の中心。"""
    return (np.arange(n_bars, dtype=np.float64) + 0.5) / n_bars


def intraday_profile(
    r_2d: np.ndarray, method: str = "median_abs", min_days: int = 20
) -> dict[str, Any]:
    """バー位置ごとのリターン散らばりの生プロファイル (平滑化なし)。

    Parameters
    ----------
    r_2d:
        ``(n_days, n_bars)`` のセッション内リターン。**日をまたぐ差分を含めない**
        (``BarSeries.returns_2d()`` はその構造で返す)。
    method:
        ``"median_abs"`` (既定) / ``"mean_abs"`` / ``"rms"``。

        既定を中央値にしてあるのは **S3 のジャンプに対する頑健性**のため。
        ``rms`` は 1 本のジャンプでその位置のバーだけ跳ね上がり、φ̂ に偽のスパイクを
        作る。中央値は ``sigma`` の定数倍 (正規なら 0.6745 sigma) に収束するので、
        正規化で定数が消える以上、φ の推定には十分である。

    Notes
    -----
    どの方法でも「バー位置 k の散らばり = phi_k x (位置に依らない定数)」が成り立つ
    (季節性が**乗法**変調だから)。定数は正規化で落ちるので、φ の形だけが残る。
    """
    r = np.asarray(r_2d, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("intraday_profile は (n_days, n_bars) の 2 次元入力が必要です")
    n_days, n_bars = r.shape
    if n_days < min_days:
        return na(f"日数 {n_days} が下限 {min_days} 未満です", n_days=n_days)
    if n_bars < 4:
        return na(f"バー数 {n_bars} が少なすぎます", n_bars=n_bars)

    a = np.abs(r)
    if method == "median_abs":
        disp = np.median(a, axis=0) * _MAD_TO_SIGMA
    elif method == "mean_abs":
        disp = a.mean(axis=0) * math.sqrt(math.pi / 2.0)
    elif method == "rms":
        disp = np.sqrt((r**2).mean(axis=0))
    else:
        raise ValueError(f"未知の method={method!r} です")

    if not np.all(disp > 0):
        return na("散らばりが 0 のバー位置があります", n_zero=int((disp <= 0).sum()))

    return ok(
        disp,
        method=method,
        n_days=int(n_days),
        n_bars=int(n_bars),
        u=bin_centers(n_bars),
        # 「起伏があるか」の素朴な指標。脱季節化後はこれが 1 に近づく。
        max_min_ratio=float(disp.max() / disp.min()),
    )


def _design(u: np.ndarray, n_harmonics: int, include_slope: bool) -> np.ndarray:
    """``[1, (u-1/2)?, cos 2pi k u, sin 2pi k u, ...]`` の計画行列。

    生成側 (:func:`~simchart.layers.l0_calendar.fourier_profile`) と**同じ基底**を
    使う。これは推定を有利にしすぎているようだが、そうではない: 基底が合っていても
    係数は標本から推定するので推定誤差は残り、その大きさを測るのがこの経路の目的
    である。基底違いの頑健性は ``n_harmonics`` を振って確かめる。
    """
    cols: list[np.ndarray] = [np.ones_like(u)]
    if include_slope:
        cols.append(u - 0.5)
    for k in range(1, n_harmonics + 1):
        ang = 2.0 * math.pi * k * u
        cols.append(np.cos(ang))
        cols.append(np.sin(ang))
    return np.column_stack(cols)


def _normalize_sq_mean(g: np.ndarray) -> np.ndarray:
    """バー上の二乗平均を 1 に揃える (生成側の ``(1/T)∫phi^2 du = 1`` の離散版)。"""
    return g / math.sqrt(float((g**2).mean()))


def estimate_phi(
    r_2d: np.ndarray,
    n_harmonics: int = 3,
    method: str = "median_abs",
    include_slope: bool = True,
    fit_days: slice | None = None,
) -> dict[str, Any]:
    """リターンから φ_σ を推定する (φ を知らない立場)。

    手順は 3 段:

    1. バー位置ごとの散らばり (:func:`intraday_profile`) を測る
    2. その **対数**を Fourier 基底へ最小二乗回帰して平滑化する。
       対数空間で当てるのは (a) 正値が保証される (b) 季節性が乗法だから
       ``log disp_k = log phi_k + const + 誤差`` と加法モデルになる、の 2 点。
       生の disp に当てると誤差が水準に比例する不均一分散のまま扱うことになる。
    3. バー上の二乗平均が 1 になるよう正規化する (真値と同じ規約)

    Parameters
    ----------
    fit_days:
        推定に使う日の範囲。標本外評価をするときは訓練期間だけを渡す
        (既定の ``None`` は全標本 = 記述用)。
    """
    r = np.asarray(r_2d, dtype=np.float64)
    fit = r if fit_days is None else r[fit_days]
    prof = intraday_profile(fit, method=method)
    if prof["status"] != "ok":
        return prof

    disp = np.asarray(prof["value"], dtype=np.float64)
    n_bars = disp.shape[0]
    u = bin_centers(n_bars)
    n_par = 1 + int(include_slope) + 2 * n_harmonics
    if n_bars <= n_par:
        return na(f"バー数 {n_bars} が母数 {n_par} 以下です", n_bars=n_bars)

    X = _design(u, n_harmonics, include_slope)
    y = np.log(disp)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    phi_hat = _normalize_sq_mean(np.exp(fitted))

    resid = y - fitted
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    return ok(
        phi_hat,
        u=u,
        n_bars=int(n_bars),
        n_days_used=int(fit.shape[0]),
        method=method,
        n_harmonics=int(n_harmonics),
        include_slope=bool(include_slope),
        coefficients=beta,
        # 平滑化がどれだけ生プロファイルを説明したか。低いと基底が足りない。
        log_profile_r2=float(r_squared),
        raw_profile=_normalize_sq_mean(disp),
        max_min_ratio=float((phi_hat**2).max() / (phi_hat**2).min()),
        at_open=float(phi_hat[0]),
        at_close=float(phi_hat[-1]),
        argmin_u=float(u[int(phi_hat.argmin())]),
    )


def true_phi_bars(
    calendar: Any,
    n_bars: int,
    n_quad: int = 64,
    steps_per_day: int | None = None,
) -> dict[str, Any]:
    """カレンダーの真の φ_σ を**バー粒度**へ集約する。

    ★バーの φ は「バー中心での φ」ではなく **φ² のバー内平均の平方根**である:

    .. math:: \\varphi_k = \\sqrt{B \\int_{k/B}^{(k+1)/B} \\varphi_\\sigma^2(u)\\,du}

    バーリターンの分散は瞬間分散のバー内**積分**であり、加算されるのは φ² だから。
    中心値で代用すると、φ が急峻な寄付・引け近傍で系統誤差が出る (バーが粗いほど
    大きい)。この定義なら ``mean_k phi_k^2 = ∫phi^2 du = 1`` が自動的に成り立ち、
    推定側の正規化規約とそのまま一致する。

    ``steps_per_day`` を渡した場合 (推奨)
    -------------------------------------
    連続時間の積分ではなく、**生成側と同じ離散和**を使う。L2 は Euler-Maruyama で
    各ステップの**左端**の σ を使うので、バー k の実際の分散は
    ``sum_{i in bar k} phi(u_i)^2 sigma_i^2 dt`` であり、``u_i`` はステップ左端。
    連続積分で代用すると ``O(phi'/steps_per_day)`` の系統差が残り、「φ で割ると
    S3 が厳密に復元できる」という S4 最強のゲートが機械精度で通らなくなる。
    バー = 格子刻みのときは左端値そのものに退化する。
    """
    if not hasattr(calendar, "phi_sigma_of_u"):
        return na("カレンダーに季節性がありません (S0〜S3)")
    if not getattr(calendar, "has_seasonality", False):
        return na("enable_seasonality=False")

    if steps_per_day is not None:
        if steps_per_day % n_bars != 0:
            return na(
                f"steps_per_day={steps_per_day} がバー数 {n_bars} で割り切れません",
                steps_per_day=int(steps_per_day),
                n_bars=int(n_bars),
            )
        steps_per_bar = steps_per_day // n_bars
        u_steps = np.arange(steps_per_day, dtype=np.float64) / steps_per_day
        phi2 = np.asarray(calendar.phi_sigma_of_u(u_steps), dtype=np.float64) ** 2
        phi_bar = np.sqrt(phi2.reshape(n_bars, steps_per_bar).mean(axis=1))
        mode = "generator_discrete"
    else:
        edges = np.arange(n_bars + 1, dtype=np.float64) / n_bars
        # 各バーを n_quad 分割した中点則。φ は滑らかなのでこれで十分収束する。
        offs = (np.arange(n_quad, dtype=np.float64) + 0.5) / (n_quad * n_bars)
        uu = edges[:-1, None] + offs[None, :]
        phi2 = np.asarray(calendar.phi_sigma_of_u(uu.ravel()), dtype=np.float64) ** 2
        phi_bar = np.sqrt(phi2.reshape(n_bars, n_quad).mean(axis=1))
        mode = "continuous_quadrature"

    return ok(
        phi_bar,
        u=bin_centers(n_bars),
        n_bars=int(n_bars),
        n_quad=int(n_quad),
        mode=mode,
        # 離散版では厳密に 1 にならない (左 Riemann 和の O(1/n) 誤差)。全バー共通の
        # スケールなので平坦さや ACF には影響しないが、ずれの大きさは残しておく。
        sq_mean=float((phi_bar**2).mean()),
        max_min_ratio=float((phi_bar**2).max() / (phi_bar**2).min()),
        at_open=float(phi_bar[0]),
        at_close=float(phi_bar[-1]),
    )


# ---------------------------------------------------------------------------
# 除去
# ---------------------------------------------------------------------------
def deseasonalize(r_2d: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """バー位置ごとの φ で割って季節性を除く。

    ``phi`` は :func:`true_phi_bars` か :func:`estimate_phi` の出力 (長さ ``n_bars``)。
    二乗平均が 1 に正規化されているので、**除去しても系列全体の分散水準は変わらない**
    (帯域ごとの配分が平らになるだけ)。水準が動くと S3 との比較で「季節性を消したのか
    スケールを変えたのか」が分離できなくなる。
    """
    r = np.asarray(r_2d, dtype=np.float64)
    p = np.asarray(phi, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("deseasonalize は (n_days, n_bars) の 2 次元入力が必要です")
    if p.shape != (r.shape[1],):
        raise ValueError(f"phi の長さ {p.shape} がバー数 {r.shape[1]} と一致しません")
    if not np.all(p > 0):
        raise ValueError("phi に非正の値が含まれます")
    return r / p[None, :]


def phi_recovery(phi_hat: np.ndarray, phi_true: np.ndarray) -> dict[str, Any]:
    """推定 φ̂ が真の φ をどれだけ再現したか。"""
    a = np.asarray(phi_hat, dtype=np.float64)
    b = np.asarray(phi_true, dtype=np.float64)
    if a.shape != b.shape:
        return na(f"形状が違います: {a.shape} vs {b.shape}")
    rel = a / b - 1.0
    return ok(
        float(np.abs(rel).max()),
        max_abs_rel_error=float(np.abs(rel).max()),
        rmse_rel=float(np.sqrt((rel**2).mean())),
        bias_rel=float(rel.mean()),
        correlation=float(np.corrcoef(a, b)[0, 1]),
        max_min_ratio_hat=float((a**2).max() / (a**2).min()),
        max_min_ratio_true=float((b**2).max() / (b**2).min()),
    )


def phi_normalization_check(calendar: Any, n_grid: int = 20001) -> dict[str, Any]:
    """生成側 φ の正規化条件と形状を検証する。

    - ``phi_sigma``: ``(1/T)∫phi^2 du = 1`` (分散が加算されるので二乗)
    - ``phi_lambda``: ``(1/T)∫phi du = 1`` (強度なので一乗)

    正規化がずれていると、日次積分分散が目標 ``sigma_bar^2`` を外れる。片方の規約を
    もう片方に流用するのがこの層で最も起こりやすい誤りなので、両方を明示的に測る。
    """
    if not hasattr(calendar, "phi_sigma_of_u"):
        return na("カレンダーに季節性がありません (S0〜S3)")
    if not getattr(calendar, "has_seasonality", False):
        return na("enable_seasonality=False")

    u = np.linspace(0.0, 1.0, n_grid)
    ps = np.asarray(calendar.phi_sigma_of_u(u), dtype=np.float64)
    pl = np.asarray(calendar.phi_lambda_of_u(u), dtype=np.float64)
    sq_mean = float(np.trapezoid(ps**2, u))
    lam_mean = float(np.trapezoid(pl, u))

    return ok(
        None,
        phi_sigma_sq_mean=sq_mean,
        phi_lambda_mean=lam_mean,
        phi_sigma_sq_mean_error=abs(sq_mean - 1.0),
        phi_lambda_mean_error=abs(lam_mean - 1.0),
        phi_sigma_sq_max_min_ratio=float((ps**2).max() / (ps**2).min()),
        phi_lambda_max_min_ratio=float(pl.max() / pl.min()),
        phi_sigma_positive=bool(ps.min() > 0),
        phi_lambda_positive=bool(pl.min() > 0),
        # 形状の要件 (§4.3): ボラは寄付が最大で引けはそれより低い、
        # 出来高は引けが最大、どちらも最小は日中。
        phi_sigma_open=float(ps[0]),
        phi_sigma_close=float(ps[-1]),
        phi_sigma_argmin_u=float(u[int(ps.argmin())]),
        phi_sigma_open_gt_close=bool(ps[0] > ps[-1]),
        phi_lambda_open=float(pl[0]),
        phi_lambda_close=float(pl[-1]),
        phi_lambda_argmin_u=float(u[int(pl.argmin())]),
        phi_lambda_close_gt_open=bool(pl[-1] > pl[0]),
        phi_sigma_min_interior=bool(0.15 < u[int(ps.argmin())] < 0.85),
        phi_lambda_min_interior=bool(0.15 < u[int(pl.argmin())] < 0.85),
    )


# ---------------------------------------------------------------------------
# 除去の効き目
# ---------------------------------------------------------------------------
def profile_flatness(r_2d: np.ndarray, method: str = "median_abs") -> dict[str, Any]:
    """日内プロファイルの平坦さ。脱季節化の効き目を直接測る一番素直な指標。

    季節性があれば ``log`` プロファイルの標準偏差は大きく、除去できていれば
    標本誤差の水準まで落ちる。ACF の周期ピーク (:func:`acf_periodicity_test`) より
    解釈が容易で、バー粒度に依らず検出力があるので、ゲートの主判定はこちらを使う。

    標本誤差はデータから推定する (理論値を使わない)
    ----------------------------------------------
    「平坦になった」の基準は標本誤差なので、その見積りが甘いと判定が意味を失う。
    当初は正規分布を仮定した ``sqrt(pi/2)/sqrt(n)`` を使ったが、これは 2 つの点で
    正しくない:

    1. 定数が違う。``|x|`` の中央値の漸近 SE は ``sqrt(pi/2)`` ではなく
       ``1/(2 f(m) sqrt(n))`` で、標準正規なら 0.787、対数にすると 1.166
    2. そもそも iid 正規ではない。確率ボラで日ごとに σ が変わるぶん SE は膨らむ

    そこで**分割標本**で直接推定する: 日を前半・後半に分けて別々にプロファイルを
    作り、その差の散らばりから全標本の SE を逆算する
    (``Var(log p_A - log p_B) = 4 Var(log p_full)``)。日ごとのボラ水準の違いは
    プロファイルを平均で割る時点で消えるので、遅い成分による前半後半の相関は
    形の推定にはほとんど効かない。
    """
    prof = intraday_profile(r_2d, method=method)
    if prof["status"] != "ok":
        return prof
    r = np.asarray(r_2d, dtype=np.float64)
    disp = np.asarray(prof["value"], dtype=np.float64)
    log_disp = np.log(disp / disp.mean())
    sd = float(log_disp.std(ddof=1))
    n_days = int(prof["n_days"])

    half = n_days // 2
    se_split: float | None = None
    if half >= 20:
        pa = intraday_profile(r[:half], method=method, min_days=10)
        pb = intraday_profile(r[half : 2 * half], method=method, min_days=10)
        if pa["status"] == "ok" and pb["status"] == "ok":
            a = np.asarray(pa["value"], dtype=np.float64)
            b = np.asarray(pb["value"], dtype=np.float64)
            d = np.log(a / a.mean()) - np.log(b / b.mean())
            se_split = float(d.std(ddof=1) / 2.0)

    # 参考値 (iid 正規を仮定した理論 SE)。分割標本が取れないときの後退先でもある。
    se_analytic = (
        1.1664 / math.sqrt(max(n_days, 1))
        if method == "median_abs"
        else 1.0 / math.sqrt(max(2 * n_days, 1))
    )
    se = se_split if se_split and se_split > 0 else se_analytic

    return ok(
        sd,
        sd_log_profile=sd,
        max_min_ratio=float(disp.max() / disp.min()),
        sampling_se=float(se),
        sampling_se_source="split_half" if se is se_split else "analytic_iid_normal",
        sampling_se_split_half=se_split,
        sampling_se_analytic=float(se_analytic),
        excess_over_se=float(sd / se) if se > 0 else float("nan"),
        n_days=n_days,
        n_bars=int(prof["n_bars"]),
    )


def acf_periodicity_test(
    x: np.ndarray, period: int, n_multiples: int = 5, halfwidth: int = 2
) -> dict[str, Any]:
    """日次周期ラグでの ACF の盛り上がりを測る。

    季節性のある ``|r|`` は決定論的な周期成分を持つので、ACF はラグが周期の整数倍の
    ところで隣接ラグより高くなる。脱季節化できていればその段差が消える。

    統計量の作り方 (★近傍平均を引いてはならない)
    --------------------------------------------
    素朴には ``excess = rho(mB) - mean(rho(mB +- j))`` としたくなるが、これは
    **系統的に負へ偏る**。ボラの ACF はラグに対して減少**凸**なので、
    ``mean(rho(L-j), rho(L+j)) > rho(L)`` が季節性と無関係に成り立つからである
    (Jensen)。実測でも、季節性のある 1 分バーで excess = -0.0027 (z = -2.2) という
    「符号が逆の有意」が出た。

    そこで近傍の点に**ラグの 2 次式を当てはめて**ピーク位置に外挿し、その予測値との
    差をとる。凸性 (2 次の項) が明示的に吸収されるので、残るのは周期成分だけになる。

    検出力はバー粒度に強く依存する (実測)
    -------------------------------------
    隣接バー間の φ の差が小さいと ``rho(mB)`` と ``rho(mB±1)`` はほぼ同じ値になり、
    周期の段差が原理的に立たない。同一経路 (400 日) での実測は次のとおり。
    「近傍平均」は凸性バイアスを含む旧版、「2 次外挿」がこの実装:

    ====== ====== ============== ============ ============
    バー   本/日  隣接 φ 差/平均  近傍平均の z  2 次外挿の z
    ====== ====== ============== ============ ============
    1 分   390    0.008          -2.2 (偽陽性) -1.1
    15 分  26     0.111          +1.6          +0.6
    30 分  13     0.215          +4.1          +1.2
    78 分  5      0.469          +8.6          +1.6
    ====== ====== ============== ============ ============

    旧版は 1 分足で「符号が逆の有意」を出し (凸性バイアス)、新版はそれを消した
    代わりにシグナルも失った。**どちらの版でも実用的な検出力は無い。**
    細かいバーでの「周期性なし」は季節性が無い証拠にはならないので、主判定には
    :func:`spectral_periodicity_test` と :func:`profile_flatness` を使う
    (どちらも粒度に依らず検出力がある — 1 分足でスペクトル比 241)。

    ★この関数は診断用であって、ゲートの主判定に使ってはならない
    ------------------------------------------------------------
    凸性を 2 次式で吸収すると、**周期成分そのものも一緒に吸収されてしまう**。
    period=13、halfwidth=2 の近傍はラグにして周期の ±15% しかなく、そこでは周期
    成分自体が大きく曲がっているからである。実測でも、近傍平均から 2 次外挿に
    替えたことで 30 分足の raw の z が +4.1 → +1.2 に落ちた (バイアスと一緒に
    シグナルが消えた)。近傍を広げても周期の隣の山に届くだけで解決しない。

    ACF は「周期成分」と「滑らかな減衰」を分離するのに向いていない、というのが
    結論である。分離はスペクトルでやるべきで、それが
    :func:`spectral_periodicity_test` — 周期成分は離散的な高調波にしか力を持たず、
    長期記憶は連続スペクトルなので、周波数領域では原理的に分離できる。
    ゲートはそちらを使い、この関数は「実証で普通に描かれる ACF がどう見えるか」の
    記録として残す。

    ``z`` は ACF 推定量が独立で SE = 1/sqrt(n) という粗い近似に基づく目安値。
    ボラ系列では推定量同士が相関するので厳密な p 値として読んではならない。
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.shape[0]
    max_lag = n_multiples * period + halfwidth
    if n < 4 * max_lag:
        return na(f"標本 {n} がラグ {max_lag} に対して短すぎます", n=n, max_lag=max_lag)
    if halfwidth < 2:
        return na("halfwidth は 2 以上必要です (2 次式の当てはめに 4 点要る)")

    d = a - a.mean()
    denom = float(d @ d)
    if denom <= 0:
        return na("系列の分散が 0 です")
    lags = np.arange(1, max_lag + 1)
    rho = np.array([float(d[: -int(k)] @ d[int(k) :]) / denom for k in lags])

    offs = np.array([j for j in range(-halfwidth, halfwidth + 1) if j != 0], dtype=np.float64)
    basis = np.column_stack([np.ones_like(offs), offs, offs**2])
    peaks: list[float] = []
    baselines: list[float] = []
    for m in range(1, n_multiples + 1):
        L = m * period
        peaks.append(float(rho[L - 1]))
        nb = rho[(L + offs.astype(int)) - 1]
        coef, *_ = np.linalg.lstsq(basis, nb, rcond=None)
        baselines.append(float(coef[0]))  # offs = 0 での予測値

    peak_arr = np.asarray(peaks)
    base_arr = np.asarray(baselines)
    excess = float((peak_arr - base_arr).mean())
    # 2 次外挿は近傍平均より分散が大きい。基底の投影から実効的な重みを出す。
    w0 = np.linalg.pinv(basis)[0]
    se = math.sqrt((1.0 + float(w0 @ w0)) / (n_multiples * n))

    return ok(
        excess,
        excess=excess,
        peak_mean=float(peak_arr.mean()),
        baseline_mean=float(base_arr.mean()),
        z_approx=float(excess / se) if se > 0 else float("nan"),
        period=int(period),
        n_multiples=int(n_multiples),
        halfwidth=int(halfwidth),
        peak_acf=peak_arr,
        excess_by_multiple=peak_arr - base_arr,
        n=int(n),
    )


def spectral_periodicity_test(
    r_2d: np.ndarray,
    n_harmonics: int = 4,
    n_neighbours: int = 40,
    detrend: bool = True,
) -> dict[str, Any]:
    """日内周期成分をスペクトルの高調波で検出する (**ゲートの主判定**)。

    なぜ周波数領域なのか
    --------------------
    季節性は ``|r|`` に**厳密に周期 B の決定論的成分**を加える。周期成分の
    スペクトルは離散的で、力は基本周波数 ``2pi/B`` の整数倍 (高調波) **だけ**に
    載る。一方ボラの長期記憶は連続スペクトルで、高調波の位置に特別なことは
    起きない。したがって周波数領域では両者が原理的に分離でき、ACF でやろうとした
    ときの「周期成分と滑らかな減衰が混ざる」問題 (:func:`acf_periodicity_test`) が
    そもそも生じない。

    バー粒度に依らず検出力があるのも周波数領域の利点である。φ は滑らかなので、
    細かいバーにしても力は低次の高調波 (k=1,2,3) に集中したままで、薄まらない。

    統計量
    ------
    長さ ``n = D*B`` (D 日 x B バー) の系列のピリオドグラムを取ると、周期 B の
    高調波はちょうど周波数番号 ``j = k*D`` に落ちる (整数なので漏れが無い)。
    帰無仮説 (周期成分なし) のもとで各オーダーは ``I_j ~ Exp(f(lambda_j))`` に
    独立に従うので、近傍の**中央値**で基準化した

    .. math:: R_k = I_{kD} / (\\mathrm{median}(近傍) / \\log 2)

    は近似的に ``Exp(1)``。中央値を使うのは、近傍に他の高調波や外れ値が混じっても
    基準が動かないようにするため (平均だと汚染される)。K 個の高調波をまとめて
    ``2 sum_k R_k ~ chi^2(2K)`` で p 値にする。

    ``detrend`` は各日の平均を抜くかどうか。抜かないと日ごとのボラ水準の変動
    (S1〜S3 の長期記憶そのもの) が超低周波に巨大な力を作り、近傍中央値の推定を
    歪める。抜いても周期成分は日内の形なので失われない。
    """
    from scipy import stats as _st

    r = np.asarray(r_2d, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("(n_days, n_bars) の 2 次元入力が必要です")
    n_days, n_bars = r.shape
    if n_days < 30:
        return na(f"日数 {n_days} が少なすぎます", n_days=n_days)
    if n_bars < 2 * (n_harmonics + 1):
        return na(f"バー数 {n_bars} が高調波 {n_harmonics} に対して少なすぎます")

    a = np.abs(r)
    if detrend:
        a = a - a.mean(axis=1, keepdims=True)
    else:
        a = a - a.mean()

    x = a.ravel()
    n = x.size
    power = np.abs(np.fft.rfft(x)) ** 2 / n
    n_freq = power.size - 1  # 番号 1..n_freq (0 は平均、除く)

    harmonics = [k * n_days for k in range(1, n_harmonics + 1)]
    if harmonics[-1] + n_neighbours >= n_freq:
        return na("高調波が周波数範囲を超えます", n_freq=int(n_freq))
    harmonic_set = {k * n_days for k in range(1, n_bars // 2 + 1)}

    ratios: list[float] = []
    for j in harmonics:
        lo, hi = max(j - n_neighbours, 1), min(j + n_neighbours, n_freq)
        idx = [i for i in range(lo, hi + 1) if i not in harmonic_set]
        if len(idx) < 10:
            return na("近傍の周波数が足りません")
        med = float(np.median(power[idx]))
        if med <= 0:
            return na("近傍スペクトルの中央値が 0 です")
        ratios.append(float(power[j] / (med / math.log(2.0))))

    ratio_arr = np.asarray(ratios)
    stat = 2.0 * float(ratio_arr.sum())
    p_value = float(_st.chi2.sf(stat, df=2 * n_harmonics))

    return ok(
        float(ratio_arr.mean()),
        mean_ratio=float(ratio_arr.mean()),
        max_ratio=float(ratio_arr.max()),
        ratios=ratio_arr,
        chi2_stat=stat,
        p_value=p_value,
        # 帰無のもとでの期待値。1 から離れるほど周期成分がある。
        null_expected_ratio=1.0,
        n_harmonics=int(n_harmonics),
        n_neighbours=int(n_neighbours),
        n_days=int(n_days),
        n_bars=int(n_bars),
        detrended=bool(detrend),
    )


# ---------------------------------------------------------------------------
# オーバーナイト
# ---------------------------------------------------------------------------
def close_to_close_returns(r_intraday_daily: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """日中日次リターンとギャップからクローズ・トゥ・クローズ系列を組む。

    ``gaps[d] = open_{d+1} - close_d`` なので
    ``c2c_{d+1} = close_{d+1} - close_d = gaps[d] + intraday_{d+1}``。
    長さは ``n_days - 1``。日中側の系列だけを見ると ON の分散が丸ごと欠けるので、
    尖度やテール指数を実証と比べるときは必ずこちらを使う。
    """
    r = np.asarray(r_intraday_daily, dtype=np.float64).ravel()
    g = np.asarray(gaps, dtype=np.float64).ravel()
    if g.shape[0] != r.shape[0] - 1:
        raise ValueError(f"ギャップ数 {g.shape[0]} が日数 {r.shape[0]} - 1 と一致しません")
    return g + r[1:]


def overnight_stats(
    gaps: np.ndarray,
    r_intraday_daily: np.ndarray,
    sigma_close: np.ndarray | None = None,
) -> dict[str, Any]:
    """オーバーナイト・ギャップの統計。

    測るもの:

    - 分散シェア ``Var(gap) / (Var(gap) + Var(日中日次))`` — 設定値との整合
    - 尖度 — ON は「単一の情報ショックが溜まって一度に出る」ので日中日次より厚い
    - ``corr(|gap|, sigma_close)`` — 引けのボラ水準との連動 (構造上正のはず)
    - **帰無対照**: ``corr(gap_d, 日中_{d+1})``。設計上ギャップは翌日の日中方向を
      予測しないので 0 のはず。ここが 0 でなければ実装が未来を漏らしている。
    """
    g = np.asarray(gaps, dtype=np.float64).ravel()
    r = np.asarray(r_intraday_daily, dtype=np.float64).ravel()
    if g.size == 0:
        return na("ギャップがありません (enable_overnight=False)")
    if g.size < 30:
        return na(f"ギャップ数 {g.size} が少なすぎます", n_gaps=int(g.size))

    from scipy import stats as _st

    var_g = float(g.var(ddof=1))
    var_r = float(r.var(ddof=1))
    share = var_g / (var_g + var_r) if var_g + var_r > 0 else float("nan")

    out: dict[str, Any] = {
        "n_gaps": int(g.size),
        "var_gap": var_g,
        "var_intraday_daily": var_r,
        "variance_share": share,
        "sd_gap": float(math.sqrt(var_g)),
        "mean_gap": float(g.mean()),
        "kurtosis_gap": float(_st.kurtosis(g, fisher=False, bias=False)),
        "kurtosis_intraday_daily": float(_st.kurtosis(r, fisher=False, bias=False)),
        "skewness_gap": float(_st.skew(g, bias=False)),
        # ジャンプの痕跡: 3 標準偏差を超えるギャップの割合 (正規なら 0.27%)。
        "frac_beyond_3sd": float((np.abs(g - g.mean()) > 3.0 * math.sqrt(var_g)).mean()),
    }

    if sigma_close is not None:
        s = np.asarray(sigma_close, dtype=np.float64).ravel()
        if s.shape == g.shape and s.std() > 0:
            out["corr_abs_gap_sigma_close"] = float(np.corrcoef(np.abs(g), s)[0, 1])
            out["corr_abs_gap_sigma_close_se"] = float(1.0 / math.sqrt(g.size))
            out["sigma_close_cv"] = float(s.std(ddof=1) / s.mean())
            # ★この相関に「> 0.5」のような固定閾値を当ててはならない。
            # 構造上 sigma_ON = c_ON sigma_close なので**潜在**の相関は厳密に 1 だが、
            # 観測できるのは |gap| = |sigma_ON z + J| であり、|z| の揺らぎと ON
            # ジャンプの独立成分で必ず希薄化する。希薄化後の目安は
            #   sqrt(2/pi) v / sqrt(v^2 + 1 - 2/pi)   (v = sigma_close の変動係数)
            # にジャンプ分の希薄化 sqrt(1 - ジャンプ分散シェア) を掛けたもの。
            # 判定は「0 より有意に大きいか」で行い、この予測値は解釈の補助に使う。
            v = out["sigma_close_cv"]
            base = math.sqrt(2.0 / math.pi) * v / math.sqrt(v * v + 1.0 - 2.0 / math.pi)
            out["corr_abs_gap_sigma_close_predicted_no_jump"] = float(base)

    # 帰無対照: ギャップは翌日の日中リターンの向きを予測しないはず。
    if r.size == g.size + 1:
        nxt = r[1:]
        if g.std() > 0 and nxt.std() > 0:
            out["corr_gap_next_intraday"] = float(np.corrcoef(g, nxt)[0, 1])
            out["corr_gap_next_intraday_se"] = float(1.0 / math.sqrt(g.size))
        c2c = close_to_close_returns(r, g)
        out["var_close_to_close"] = float(c2c.var(ddof=1))
        out["kurtosis_close_to_close"] = float(_st.kurtosis(c2c, fisher=False, bias=False))

    return ok(share, **out)


# ---------------------------------------------------------------------------
# S7 への引き渡し
# ---------------------------------------------------------------------------
def time_change_by_phi_lambda(
    times: np.ndarray, calendar: Any, session_seconds: float
) -> np.ndarray:
    """イベント時刻を ``Lambda(t) = ∫_0^t phi_lambda(u) du`` で時間変更する。

    **S4 では消費されないが、S7 の Hawkes 推定はこれを必ず通す。** 季節性のある
    ポアソン過程は、この時間変更のあとで定常ポアソンになる (時間変更定理)。
    生時刻のまま Hawkes を当てると、活発な時間帯へのイベント集中が自己励起として
    説明され、分岐比 ``n`` が過大推定される (Filimonov-Sornette)。φ_λ の
    ``(1/T)∫phi du = 1`` 正規化は「変更後の時間の総量がセッション長と等しい」ことを
    意味するので、変更前後で単位が保たれる。

    S4 の時点で実装しておく理由は、φ_λ を作った層とそれを消費する規約を
    離してしまうと、S7 で規約 (一乗正規化か二乗か) を取り違える危険があるため。
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    if not hasattr(calendar, "phi_lambda_of_u"):
        raise ValueError("カレンダーに phi_lambda がありません")

    day = np.floor(t / session_seconds)
    u = t / session_seconds - day
    # Lambda(u) をセッション内で数値積分し、その累積を線形補間で引く。
    n_grid = 4001
    ug = np.linspace(0.0, 1.0, n_grid)
    lam = np.asarray(calendar.phi_lambda_of_u(ug), dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(ug))])
    return (day + np.interp(u, ug, cum)) * session_seconds


def coarsen(r_2d: np.ndarray, n_out: int) -> np.ndarray:
    """細かいバーのリターンを足し上げて粗いバーにする。

    リターンは対数価格の差なので、隣接するバーの**和**がそのまま粗いバーの
    リターンになる (再標本化と厳密に一致する)。周期 ACF 検定は粗いバーでないと
    検出力が出ないので、その前処理として使う。
    """
    r = np.asarray(r_2d, dtype=np.float64)
    n_bars = r.shape[1]
    if n_bars % n_out != 0:
        raise ValueError(f"バー数 {n_bars} が {n_out} で割り切れません")
    return r.reshape(r.shape[0], n_out, n_bars // n_out).sum(axis=2)


def deseasonalization_report(
    r_2d: np.ndarray,
    calendar: Any,
    n_harmonics: int = 3,
    method: str = "median_abs",
    steps_per_day: int | None = None,
    periodicity_bars_per_day: int = 13,
) -> dict[str, Any]:
    """真値経路と推定経路の両方で脱季節化し、効き目をまとめる。

    ゲートが参照する枝を 1 か所に集める:

    - ``raw`` / ``true_phi_removed`` / ``est_phi_removed`` の平坦さと周期 ACF
    - ``recovery``: φ̂ が真の φ をどれだけ当てたか

    平坦さは入力の粒度で、周期 ACF は ``periodicity_bars_per_day`` 本へ粗くしてから
    測る (:func:`acf_periodicity_test` の docstring にある検出力の理由)。
    """
    r = np.asarray(r_2d, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("(n_days, n_bars) の 2 次元入力が必要です")
    n_bars = r.shape[1]

    truth = true_phi_bars(calendar, n_bars, steps_per_day=steps_per_day)
    est = estimate_phi(r, n_harmonics=n_harmonics, method=method)

    coarse_ok = n_bars % periodicity_bars_per_day == 0

    def _branch(arr: np.ndarray) -> dict[str, Any]:
        out = {
            "flatness": profile_flatness(arr, method=method),
            # 主判定。バー粒度に依らず検出力がある。
            "spectral": spectral_periodicity_test(arr),
        }
        if coarse_ok:
            c = coarsen(arr, periodicity_bars_per_day)
            # 記録用 (実証で普通に描かれる ACF の見え方)。検出力は低い。
            out["acf_periodicity"] = acf_periodicity_test(
                np.abs(c).ravel(), periodicity_bars_per_day
            )
        else:
            out["acf_periodicity"] = na(
                f"バー数 {n_bars} が {periodicity_bars_per_day} で割り切れません"
            )
        return out

    out: dict[str, Any] = {
        "n_days": int(r.shape[0]),
        "n_bars": int(n_bars),
        "method": method,
        "periodicity_bars_per_day": int(periodicity_bars_per_day),
        "true_phi": truth,
        "est_phi": est,
        "raw": _branch(r),
    }

    if truth["status"] == "ok":
        out["true_phi_removed"] = _branch(deseasonalize(r, np.asarray(truth["value"])))
    if est["status"] == "ok":
        out["est_phi_removed"] = _branch(deseasonalize(r, np.asarray(est["value"])))
    if truth["status"] == "ok" and est["status"] == "ok":
        out["recovery"] = phi_recovery(
            np.asarray(est["value"]), np.asarray(truth["value"])
        )
    return out


__all__ += ["coarsen", "deseasonalization_report"]
