"""L2: 情報価格層。

S1 までの中身
-------------
S0: 定数ボラ・正規革新の幾何ブラウン運動。
S1: 対数ボラに MSM (Markov-Switching Multifractal) と緩慢 OU を加算。

    log sigma_t = log sigma_bar + 0.5 * sum_i log M_i(t) + X_t - Var(X)
                                                           ^^^^^^^^ 凸性補正

これで |r| の長期記憶 (べき則的 ACF)・ボラティリティ・クラスタリング・
集計正規性・マルチフラクタルスケーリングが初めて現れる。革新項 z は正規のまま。
テールはボラ過程 (と S3 のジャンプ) から内生的に出す。t 分布などを外生的に入れると
時間集計で尖度が下がる性質が永久に再現できなくなる。

時間スケール不変性 (最重要の設計制約)
------------------------------------
MSM の切替強度 gamma_i と OU の theta は物理時間 (1 日 = 1 セッション) で定義し、
グリッド刻みへの変換は実装内部で行う。1 ステップあたり切替確率で実装すると
steps_per_day を変えた瞬間に別のモデルになる。具体的には:

- MSM は切替時刻を連続時間で生成する (Poisson 個数 + 一様時刻)。グリッドへは
  searchsorted で写像するだけなので、**同一シードなら解像度を変えても切替過程が
  ビット単位で一致する**。
- OU は厳密離散化 (ガウス遷移の閉形式)。Euler-Maruyama は刻み依存なので使わない。

分散予算
--------
log sigma の分散は段階間で配分する (最終予算 0.25 のうち S1 は MSM 0.125 + OU 0.050)。
m0 は自由パラメータではなく、配分 vol_var_target_msm から :func:`solve_m0` で逆算する。

拡張の入り口 (S2 以降)
----------------------
- :meth:`GBMPriceLayer._log_vol_path` … S2 (ラフ), S5 (chi_2) をさらに加算
- :meth:`GBMPriceLayer._jump_component` … S3 (Hawkes ジャンプ)
- :meth:`GBMPriceLayer._leverage_innovation` … S3 (レバレッジ効果)
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from scipy import fft as sp_fft
from scipy import signal

from ..config import Config
from ..rng import RNGRegistry
from ..types import PriceProcess
from .l0_calendar import ConstantCalendar
from .l1_activity import ConstantActivity

__all__ = [
    "GBMPriceLayer",
    "build_price_layer",
    "solve_m0",
    "msm_theoretical_var_log_sigma",
    "solve_eta_rough",
    "rough_discrete_stationary_variance",
    "davies_harte_fgn",
    "compose_log_sigma",
    "prepare_chaos_component",
    "VOL_SUBSAMPLE_SECONDS",
]

#: 診断用サブサンプルの間隔 (秒)。全ステップの成分内訳を保持すると本番設定で
#: 数 GB になるため、分単位に間引いて保存する 。
VOL_SUBSAMPLE_SECONDS: float = 60.0


# ---------------------------------------------------------------------------
# 分散予算ユーティリティ
# ---------------------------------------------------------------------------
def solve_m0(k: int, target_var: float) -> float:
    """MSM の状態値 m0 を、log sigma への分散配分から逆算する。

    1 成分あたりの寄与は Var(0.5 * log M_i) = [ln(m0 / (2 - m0))]^2 / 16 なので、

        Var_MSM(log sigma) = k * [ln(m0 / (2 - m0))]^2 / 16

    を m0 について解く。m0 は (1, 2) に入る (target_var > 0 のとき)。
    """
    if k < 1:
        raise ValueError("k は 1 以上である必要があります")
    if target_var <= 0:
        raise ValueError("target_var は正である必要があります")
    L = 4.0 * (target_var / k) ** 0.5
    rho = math.exp(L)
    return 2.0 * rho / (1.0 + rho)


def msm_theoretical_var_log_sigma(k: int, m0: float) -> float:
    """(k, m0) から MSM の Var(log sigma) 理論値を返す (solve_m0 の逆写像)。"""
    return k * math.log(m0 / (2.0 - m0)) ** 2 / 16.0


def solve_eta_rough(hurst: float, theta_r: float, target_var: float) -> float:
    """fOU の拡散係数 eta_r を、log sigma への分散配分から逆算する。

    定常分散 (Cheridito-Kawaguchi-Maejima):

        Var(Y) = eta_r^2 * Gamma(2H + 1) / (2 * theta_r^{2H})

    を eta_r について解く。時間の単位は日 (theta_r は [1/日])。
    """
    if not (0.0 < hurst < 1.0):
        raise ValueError("hurst は (0, 1) の範囲である必要があります")
    if theta_r <= 0 or target_var <= 0:
        raise ValueError("theta_r と target_var は正である必要があります")
    return (2.0 * target_var * theta_r ** (2.0 * hurst) / math.gamma(2.0 * hurst + 1.0)) ** 0.5


def rough_discrete_stationary_variance(
    hurst: float, theta_per_day: float, dt_days: float, eta: float
) -> float:
    """離散 fOU (AR(1) フィルタ x fGn) の定常分散を厳密に数値計算する。

    Y_k = a Y_{k-1} + eta * g_k, a = e^{-theta dt}, g は per-step 分散 dt^{2H} の
    fGn。定常分散は

        Var(Y) = eta^2 dt^{2H} / (1 - a^2) * [1 + 2 sum_{h>=1} a^h rho(h)]

    (rho は単位 fGn の自己相関)。連続式 (:func:`solve_eta_rough` の分母) とは
    離散化の分だけずれるので、凸性補正とアンサンブル断面はこちらを使う —
    補正が実際に生成される過程の分散と一致していないと E[sigma^2] = sigma_bar^2
    が崩れるため。H < 1/2 では rho の総和が -1/2 に収束する (スペクトルが原点で
    消える) ため、和の打ち切りは a^h の減衰で決める。
    """
    a = math.exp(-theta_per_day * dt_days)
    n_terms = int(min(60.0 / (theta_per_day * dt_days), 5_000_000)) + 1
    h = np.arange(1, n_terms, dtype=np.float64)
    two_h = 2.0 * hurst
    rho = 0.5 * ((h + 1.0) ** two_h - 2.0 * h**two_h + (h - 1.0) ** two_h)
    s = 1.0 + 2.0 * float(np.sum(a**h * rho))
    return eta**2 * dt_days**two_h * s / (1.0 - a * a)


def davies_harte_fgn(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """単位分散の fGn (fractional Gaussian noise) を n 点、厳密に生成する。

    Davies-Harte (circulant embedding)。O(N log N) で、共分散は近似ではなく厳密
    (テストが標本自己共分散と理論値の一致を確認する)。H < 1/2 では埋め込みの
    非負定値性が保証される。Cholesky (O(N^2)) は使わない。

    乱数消費: ``rng.standard_normal(2m)`` を一度だけ (m は n 以上の FFT 高速長)。
    m は n にのみ依存するので、同じ n なら消費列は同一。
    """
    if n < 2:
        raise ValueError("n は 2 以上である必要があります")
    m = int(sp_fft.next_fast_len(n))
    k = np.arange(m + 1, dtype=np.float64)
    two_h = 2.0 * hurst
    gamma = 0.5 * (
        (k + 1.0) ** two_h - 2.0 * k**two_h + np.abs(k - 1.0) ** two_h
    )
    row = np.concatenate([gamma, gamma[m - 1 : 0 : -1]])
    eig = np.fft.fft(row).real
    eig_min = float(eig.min())
    if eig_min < -1e-8 * float(eig.max()):
        raise ValueError(
            f"circulant embedding が非負定値ではありません (最小固有値 {eig_min:.3e})。"
            f" H={hurst} — H < 1/2 では起きないはずなので実装を疑うこと。"
        )
    np.clip(eig, 0.0, None, out=eig)

    z = rng.standard_normal(2 * m)
    v = np.empty(2 * m, dtype=np.complex128)
    v[0] = math.sqrt(eig[0]) * z[0]
    v[m] = math.sqrt(eig[m]) * z[1]
    half = np.sqrt(eig[1:m] / 2.0)
    a_part = z[2 : m + 1]
    b_part = z[m + 1 : 2 * m]
    v[1:m] = half * (a_part + 1j * b_part)
    v[m + 1 :] = np.conj(v[1:m][::-1])

    x = np.fft.fft(v) / math.sqrt(2.0 * m)
    return np.ascontiguousarray(x.real[:n])


def prepare_chaos_component(
    config, n_days: float
) -> tuple[np.ndarray, np.ndarray, float, float, dict]:
    """chi_2 の窓を用意する (S5)。生成とアンサンブル検証の両方がこれを使う。

    Returns
    -------
    (chaos_t_days, chi_norm, a, c_chi, diagnostics)
        ``chaos_t_days`` は市場日単位の格子、``chi_norm`` は正規化済み系列。
        注入は ``a * interp(t_days, chaos_t_days, chi_norm) - c_chi``。

    正規化は使用する窓の上で行う (平均 0・分散 1)。これにより経路上の chi
    分散寄与が厳密に ``a^2 = vol_var_target_chaos`` になり、シード横断相関ゲート
     の目標値 0.20 = 0.05/0.25 が構成から導ける。

    凸性補正 c_chi は数値で求める 。S1 の -Var(X) は X がガウス
    だから成立した式で、chi はガウスでない (MG の周辺分布は有界・歪み・多峰)。
    ガウスの公式を流用すると E[sigma^2] の水準がずれ、なんとなくボラが高いと
    しか見えない壊れ方をする (S1・S4 と同型の事故)。
    c_chi = 0.5 log(mean(exp(2 a chi))) で時間平均の意味で
    E[sigma_t^2] = sigma_bar^2 が厳密に保たれる。
    """
    from ..chaos import chaos_generate

    s = config.chaos_days_per_unit
    length_units = n_days / s + 2.0 * config.chaos_dt  # 端の補間分の余白
    series = chaos_generate(
        system=config.chaos_system,
        params={
            "tau": config.chaos_tau_delay,
            "beta": config.chaos_beta,
            "gamma": config.chaos_gamma,
            "n_exponent": config.chaos_n_exponent,
        },
        length_units=length_units,
        dt=config.chaos_dt,
        ic=config.chaos_ic,
        burn_in_units=config.chaos_burn_in_units,
        cache_dir=config.chaos_cache_dir or None,
    )
    x = series.x
    if config.chaos_normalization == "ecdf_normal":
        # 案 B (§3.2): 経験 CDF で周辺分布を正規に写像する。時間順序 (順位の
        # 動力学) は保たれ、写像も決定論的なので再現性を損なわない。
        from scipy.stats import norm as _norm

        ranks = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        x = _norm.ppf((ranks + 0.5) / x.shape[0])
    mu = float(x.mean())
    sd = float(x.std())
    if sd <= 0:
        raise ValueError("chi_2 の分散が 0 です (窓が短すぎるか系が退化しています)")
    chi_norm = (x - mu) / sd
    a = math.sqrt(config.vol_var_target_chaos)
    c_chi = 0.5 * math.log(float(np.mean(np.exp(2.0 * a * chi_norm))))
    chaos_t_days = series.t * s

    # ジャンプ強度の Jensen 係数 (S4 の φ 補正と同じ理屈)。
    # λ(t) = λ0 (σ/σ̄)^ρ の σ に e^{aχ−c_χ} が掛かると平均強度が
    # time-mean(e^{ρ(aχ−c_χ)}) 倍になる。c_χ が補正するのは E[e^{2aχ}] であって
    # E[e^{ρaχ}] ではない (ρ=1 では Cauchy-Schwarz により必ず 1 未満 → JV シェアが
    # 黙って動く)。jump_intensity_scale がこの係数で割って打ち消す。
    jensen = (
        float(np.mean(np.exp(config.jump_vol_exponent * (a * chi_norm - c_chi))))
        if config.enable_jump
        else 1.0
    )

    diagnostics = {
        "system": config.chaos_system,
        "sha256": series.sha256,
        "cache_path": series.cache_path,
        "params": dict(series.params),
        "normalization": config.chaos_normalization,
        "days_per_unit": s,
        "grid_spacing_days": config.chaos_dt * s,
        "n_grid_points": int(chi_norm.shape[0]),
        "a": a,
        "c_chi_numerical": c_chi,
        # ガウス公式 (Var = a^2) との差 — 数値補正が効いていることの証拠。
        "c_chi_gaussian_formula": config.vol_var_target_chaos,
        "c_chi_difference": c_chi - config.vol_var_target_chaos,
        "window_mean": mu,
        "window_sd": sd,
        "var_contribution": a * a,
        "jensen_intensity_factor": jensen,
    }
    return chaos_t_days, chi_norm, a, c_chi, diagnostics


def compose_log_sigma(
    log_sigma_bar: float,
    half_log_msm: np.ndarray | float,
    x_slow: np.ndarray | float,
    var_slow: float,
    y_rough: np.ndarray | float = 0.0,
    var_rough: float = 0.0,
    inplace: bool = False,
) -> np.ndarray | float:
    """log sigma の合成式。生成とアンサンブル検証の両方がこの 1 つを使う。

    式を 2 か所に書くと、片方だけ直して乖離する事故が起きる。凸性補正
    -Var(X) - Var(Y) はガウス成分 (OU とラフ fOU) のためのもの:
    E[e^{2X}] = e^{2Var(X)} != 1 なので、引かないと実効ボラが e^{Var} 倍に膨らむ。
    MSM 側は E[prod M_i] = 1 なので補正不要。**var_rough には生成される離散過程の
    実分散 (:func:`rough_discrete_stationary_variance`) を渡すこと** — 連続式の
    目標値を渡すと離散化の分だけ E[sigma^2] がずれる。

    ``inplace=True`` のとき ``half_log_msm`` を書き換えて返す (呼び出し側が所有権を
    渡す)。本番設定では 1 配列 936MB なので、素直に書くと中間結果だけで数 GB を
    使ってしまうため。両経路の結果はビット単位で一致する: IEEE754 の加算は
    可換なので ``log_sigma_bar + a == a + log_sigma_bar`` であり、それ以外の演算
    順序は同一だからである (tests が一致を固定している)。
    """
    if inplace:
        if not isinstance(half_log_msm, np.ndarray):
            raise TypeError("inplace=True には half_log_msm が ndarray である必要があります")
        out = half_log_msm
        out += log_sigma_bar
        out += x_slow
        out -= var_slow
        if isinstance(y_rough, np.ndarray) or y_rough != 0.0:
            out += y_rough
        if var_rough != 0.0:
            out -= var_rough
        return out
    result = log_sigma_bar + half_log_msm + x_slow - var_slow
    # ラフ成分が無いときは加算そのものを行わない (+0.0 は -0.0 を +0.0 に変える
    # ため、S1 経路とのビット単位一致を「加算しない」ことで保証する)。
    if isinstance(y_rough, np.ndarray) or y_rough != 0.0:
        result = result + y_rough
    if var_rough != 0.0:
        result = result - var_rough
    return result


def simulate_msm_path(
    cfg, rng: np.random.Generator, t_days: np.ndarray, component_range=None
) -> tuple[np.ndarray, dict]:
    """MSM の ``0.5 * sum_i log M_i(t)`` を成分レンジ指定つきで生成する。

    :meth:`GBMPriceLayer._simulate_msm` の実体 (アルゴリズムの説明はそちら)。
    S13 の共通因子 (cross_factor) と資産固有側がこの 1 つの関数を使う —
    式を 2 か所に書くと片方だけ直して乖離する事故が起きるため。

    - ``component_range=None`` は全成分 [0, k)。S12 までの経路とビット単位同一
      (乱数消費列・浮動小数の演算順序とも変更なし)。
    - m0 は常に全体の配分 (vol_var_target_msm, k) から解く — 部分集合でも
      1 成分あたりの分散寄与は変わらない (共有分割は周辺分布を保存する §4.2)。
    """
    k = cfg.msm_k
    lo, hi = (0, k) if component_range is None else component_range
    if not (0 <= lo <= hi <= k):
        raise ValueError(f"component_range ({lo}, {hi}) が [0, {k}] の外です")
    m0 = solve_m0(k, cfg.vol_var_target_msm)
    log_hi = math.log(m0)
    log_lo = math.log(2.0 - m0)
    T = float(t_days[-1])
    n_points = int(t_days.shape[0])

    switch_hash = hashlib.sha256()
    n_switches: list[int] = []
    occupancy_hi: list[float] = []
    bounds_per_component: list[np.ndarray] = []
    values_per_component: list[np.ndarray] = []

    for i in range(lo, hi):
        gamma_i = cfg.msm_gamma1_per_day * cfg.msm_b**i
        n_switch = int(rng.poisson(gamma_i * T))
        switch_times = np.sort(rng.uniform(0.0, T, n_switch))
        # 区間は n_switch + 1 個。先頭が初期値で、定常分布 (等確率) から引く。
        states = rng.integers(0, 2, n_switch + 1)

        # 切替 m 以降の状態を使い始める最初の格子点。
        # bounds[m] <= j <=> switch_times[m] <= t_days[j] なので、
        # 素朴版の状態番号 (自分以下の切替の個数) と厳密に一致する。
        bounds = np.searchsorted(t_days, switch_times, side="left")
        np.clip(bounds, 0, n_points, out=bounds)
        log_values = np.where(states == 1, log_hi, log_lo)
        bounds_per_component.append(bounds)
        values_per_component.append(log_values)

        seg_len = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), bounds,
                                          np.full(1, n_points, dtype=np.int64))))
        occupancy_hi.append(float(seg_len[states == 1].sum() / n_points))

        switch_hash.update(np.int64(n_switch).tobytes())
        switch_hash.update(np.ascontiguousarray(switch_times).tobytes())
        switch_hash.update(np.ascontiguousarray(states).tobytes())
        n_switches.append(n_switch)

    # 全成分の切替点を統合。区間数は sum(n_switch) + 1 で、格子点数とは無関係。
    starts = np.unique(
        np.concatenate((np.zeros(1, dtype=np.int64), *bounds_per_component))
        if bounds_per_component
        else np.zeros(1, dtype=np.int64)
    )
    starts = starts[starts < n_points]
    ends = np.append(starts[1:], n_points)
    seg_value = np.zeros(starts.size, dtype=np.float64)
    for bounds, log_values in zip(bounds_per_component, values_per_component):
        seg_value += log_values[np.searchsorted(bounds, starts, side="right")]
    seg_value *= 0.5

    total = np.empty(n_points, dtype=np.float64)
    for a, b, value in zip(starts, ends, seg_value):
        total[a:b] = value
    diag = {
        "k": k,
        "component_range": [int(lo), int(hi)],
        "b": cfg.msm_b,
        "gamma_per_day": [cfg.msm_gamma1_per_day * cfg.msm_b**i for i in range(lo, hi)],
        "m0": m0,
        "target_var_log_sigma": cfg.vol_var_target_msm,
        "theoretical_var_log_sigma": msm_theoretical_var_log_sigma(k, m0),
        "n_switches": n_switches,
        "expected_switches": [
            cfg.msm_gamma1_per_day * cfg.msm_b**i * T for i in range(lo, hi)
        ],
        "occupancy_hi": occupancy_hi,
        "horizon_days": T,
        "n_merged_segments": int(starts.size),
        # 切替過程のダイジェスト。解像度を変えても一致することが
        # 物理時間定義の直接証拠になる (test_scale_invariance)。
        "switch_digest": switch_hash.hexdigest(),
    }
    return total, diag


def simulate_ou_path(
    rng: np.random.Generator,
    t_days: np.ndarray,
    theta_per_day: float,
    var_x: float,
    driver: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """平均 0 の OU を厳密離散化で生成する (:meth:`GBMPriceLayer._simulate_slow_ou` の実体)。

    乱数消費は (x0 -> z 列) の順で固定。``driver`` があれば z 列は消費しない
    (レバレッジ = 駆動の置き換え)。S13 の共通 OU (cross_factor) と資産固有側が
    この 1 つを使う。
    """
    dt = np.diff(t_days)
    if dt.size and abs(dt.min() - dt.max()) > 1e-12 * max(dt.max(), 1.0):
        raise NotImplementedError(
            "非一様グリッド上の OU は S4 (オーバーナイト導入) で実装します。"
            " 厳密離散化自体は刻みごとの係数で対応可能です。"
        )
    step_days = float(dt[0])
    a = math.exp(-theta_per_day * step_days)
    s = math.sqrt(var_x * (1.0 - a * a))

    x0 = float(rng.normal(0.0, math.sqrt(var_x)))
    if driver is None:
        z = rng.standard_normal(t_days.shape[0] - 1)
    else:
        if driver.shape[0] != t_days.shape[0] - 1:
            raise ValueError("OU 駆動列の長さがステップ数と一致しません")
        z = driver
    # X_j = a X_{j-1} + s z_j (j >= 1) を lfilter で。zi = [a * x0] により
    # y[0] = s z[0] + a x0 = X_1 となる。
    y, _ = signal.lfilter([s], [1.0, -a], z, zi=np.array([a * x0]))
    if driver is None:
        del z
    x = np.empty(t_days.shape[0], dtype=np.float64)
    x[0] = x0
    x[1:] = y
    del y
    diag = {
        "theta_per_day": theta_per_day,
        "target_var": var_x,
        "eta_per_sqrt_day": math.sqrt(2.0 * theta_per_day * var_x),
        "step_days": step_days,
        "ar_coeff": a,
        "x0": x0,
        "sample_var": float(x.var()),
        "sample_mean": float(x.mean()),
        "driver": "external" if driver is not None else "independent",
    }
    return x, diag


# ---------------------------------------------------------------------------
class GBMPriceLayer:
    """log p*(t) を生成する。S1 では確率ボラつき GBM。"""

    name = "l2.gbm"

    def __init__(
        self,
        config: Config,
        rng: RNGRegistry,
        calendar: ConstantCalendar,
        activity: ConstantActivity | None = None,
        factor=None,
    ) -> None:
        self._config = config
        self._rng = rng
        self._calendar = calendar
        self._activity = activity
        #: S13: (beta_i, CommonFactorState)。None なら単一資産 (S12 までと同一経路)。
        if factor is not None:
            beta_i, common_state = factor
            self._factor_beta = float(beta_i)
            self._common = common_state
        else:
            self._factor_beta = 0.0
            self._common = None
        #: 秒 -> 年 の換算。年 = ann_days 日 x 1 日の取引秒数 (equity 252 x 23400 /
        #: perp 365 x 86400 — 時間軸の単一情報源は config、S0-perp §4)。
        self._seconds_per_year = config.ann_days * calendar.session_seconds()
        #: 秒 -> 日 の換算。gamma_i と theta は1 日あたりの物理時間定義 (§7)。
        self._seconds_per_day = calendar.session_seconds()
        #: 直近の simulate() の診断。pipeline が StageResult.meta に回収する。
        self.last_diagnostics: dict[str, Any] = {}
        #: 直近に生成したラフ成分の離散定常分散 (凸性補正に使う)。
        self._rough_var_eff: float = 0.0
        #: 直近のラフ生成で使った単位分散 fGn (burnin 込み。レバレッジが参照する)。
        self._last_fgn_unit: np.ndarray | None = None
        self._last_fgn_burnin: int = 0

    @property
    def sigma_bar_diffusion(self) -> float:
        """拡散側の基準ボラ。

        総 QV (年率 sigma_bar^2) を保つため、後段が確保する分だけ縮小していく:

        - S3: ジャンプ分 ``sqrt(1 - jump_qv_share_target)``
        - S4: オーバーナイト分 ``sqrt(1 - overnight_variance_share)``

        Var(log sigma) の予算 (変動幅) とは独立な軸で、S1/S2 の配分は変わらない。
        phi の正規化 ((1/T)∫phi^2 du = 1) が正しければ、季節性の導入自体は
        日次積分分散を変えないので σ̄ の修正には効かない 。
        """
        cfg = self._config
        sigma = cfg.sigma_bar
        if cfg.enable_jump:
            sigma *= math.sqrt(1.0 - cfg.jump_qv_share_target)
        if cfg.enable_overnight:
            sigma *= math.sqrt(1.0 - cfg.overnight_variance_share)
        return sigma

    @property
    def jump_intensity_scale(self) -> float:
        """ジャンプ基準強度 ``lambda0`` に掛ける S4 の補正係数。

        S4 を入れたことで日中のジャンプ/拡散の配分が動いてしまうのを止める。
        補正しないと実測で JV シェアが 12.7% → 14.9% に動いた (S3 の QV 予算の破壊)。
        原因は 2 つあり、どちらも「季節性・ON は配分を変えるが総量は変えない」という
        S4 の設計原則からの逸脱なので、両方を打ち消す。

        1. ON の取り分: 拡散側だけが ``sqrt(1-ON_share)`` で縮み、ジャンプ側は
           そのままだったので、日中 QV に占めるジャンプの比率が上がった。
           日中 QV 全体を ``(1-ON_share)`` 倍にするには強度も同率で縮める。
        2. phi の Jensen 効果: 強度は ``lambda0 (sigma_t/sigma_bar)^rho`` で、
           sigma に phi が掛かると平均強度が ``(1/T)∫phi^rho du`` 倍になる。
           phi の正規化は ``∫phi^2 du = 1`` (二乗) なので、rho != 2 では
           ``∫phi^rho du != 1`` になり、既定の rho=1 では 0.969 倍に下がる。
           φ_σ の二乗正規化をそのまま強度に流用してはならない、ということ。

        補正後はジャンプの時刻が日内で偏るが、本数と分散寄与は S3 と同じ
        になる。これがジャンプ側の季節性の正しい入れ方である。
        """
        cfg = self._config
        if not cfg.enable_jump:
            return 1.0
        scale = 1.0
        if cfg.enable_overnight:
            scale *= 1.0 - cfg.overnight_variance_share
        if cfg.enable_seasonality and hasattr(self._calendar, "phi_sigma_of_u"):
            u = np.linspace(0.0, 1.0, 20001)
            phi = np.asarray(self._calendar.phi_sigma_of_u(u), dtype=np.float64)
            mean_phi_rho = float(np.trapezoid(phi**cfg.jump_vol_exponent, u))
            if mean_phi_rho <= 0:
                raise ValueError("phi^rho の平均が 0 以下です")
            scale /= mean_phi_rho
        if cfg.enable_chaos_vol:
            # S5: chi の Jensen 係数 (φ と同じ理屈、数値は _simulate_chaos が計算)。
            # ジャンプは log_vol の後に生成されるので diagnostics に値がある。
            jf = (self.last_diagnostics.get("chaos") or {}).get("jensen_intensity_factor")
            if jf is None:
                raise RuntimeError(
                    "chi_2 の Jensen 係数が未計算です (_log_vol_path が先に走る必要があります)"
                )
            scale /= jf
        return scale

    # ------------------------------------------------------------------
    # S1: MSM 成分
    # ------------------------------------------------------------------
    def _simulate_msm(self, t_days: np.ndarray, component_range=None) -> np.ndarray:
        """0.5 * sum_i log M_i(t) をグリッド上で生成する (切替時刻ベース)。

        per-step の Bernoulli ループは k*N 回の乱数生成になり 11.7M ステップでは
        非現実的。成分ごとに (1) 切替回数 ~ Poisson(gamma_i * T)、(2) 切替時刻 ~
        Uniform(0, T) ソート、(3) 各区間の値を等確率で引く、の順で生成する。
        この生成はグリッド解像度に一切依存しない (t_days の値でしか使わない)
        ので、同一シードなら steps_per_day を変えても切替過程がビット単位で一致する。

        乱数消費は ``l2.vol_msm`` ストリームから成分 i=1..k の順に固定
        (Poisson 数 -> 時刻 -> 値)。この順序を変えると同一シードの経路が変わる。

        グリッドへの写像 (性能)
        -----------------------
        素朴には ``searchsorted(switch_times, t_days)`` で成分ごとに全格子点を
        引けるが、これは N 回の二分探索を k 回繰り返すことになり、5000 日 x 23400
        では約 30 秒・一時配列 3.7GB を要した。向きを逆にして**切替時刻の側を
        グリッドへ引く** (問い合わせ数 = 切替回数、高々数千) と、あとは区間ごとの
        定数埋めで済む。さらに全成分の切替点を統合すると、区間ごとの合計値を
        小さい配列の上で計算してから N 要素を 1 回書くだけになる。

        この経路変更は出力をビット単位で変えない。 統合区間の値は
        ``(((0 + v_0) + v_1) + ... ) * 0.5`` を成分順に計算しており、素朴版が
        格子点ごとに行う浮動小数演算と順序も含めて同一だからである
        (tests/test_s1_vol.py がリファレンス実装との一致を固定している)。

        S13: ``component_range=(lo, hi)`` で成分の部分集合だけを生成する
        (共通 = 遅い側 [0, k_c)、固有 = 速い側 [k_c, k))。実体は
        :func:`simulate_msm_path` — 共通側 (cross_factor) と単一の情報源を共有する。
        既定 (None) は全成分 [0, k) で、S12 までとビット単位同一。
        """
        total, diag = simulate_msm_path(
            self._config, self._rng.get("l2.vol_msm"), t_days, component_range
        )
        self.last_diagnostics["msm"] = diag
        return total

    # ------------------------------------------------------------------
    # S1: 緩慢 OU 成分
    # ------------------------------------------------------------------
    def _simulate_slow_ou(
        self,
        t_days: np.ndarray,
        driver: np.ndarray | None = None,
        var_override: float | None = None,
    ) -> np.ndarray:
        """X_t (平均 0 の OU) をグリッド上で生成する (厳密離散化)。

        遷移はガウスで閉形式なので、Euler-Maruyama ではなく

            X_{t+D} = X_t e^{-theta D} + sqrt(Var(X) (1 - e^{-2 theta D})) z

        を使う。これはどんな刻み D でも分布が厳密で、グリッド解像度に依存しない。
        初期値 X_0 は定常分布 N(0, Var(X)) から引く (バーンイン不要)。
        逐次再帰は AR(1) なので scipy.signal.lfilter で O(N)。

        乱数消費は ``l2.vol_slow`` から (X_0 -> z 列) の順で固定。

        Parameters
        ----------
        driver:
            S3 のレバレッジ長期チャンネル用の外部駆動列 (N(0,1)、ステップ数本)。
            与えられた場合、z 列は消費せず driver で駆動する — レバレッジとは
            OU の駆動が価格革新と相関を持つことそのものなので、駆動の
            置き換えが実装である。x0 は常に ``l2.vol_slow`` から引く ので、
            前段階照合の証人 (x0 の厳密一致) はレバレッジ有効時も機能する。
        """
        cfg = self._config
        theta = math.log(2.0) / cfg.ou_half_life_days  # [1/日]
        # S3 の中速成分は slow の予算 (0.05) の内数として再配分される。
        var_x = var_override if var_override is not None else cfg.vol_var_target_slow
        x, diag = simulate_ou_path(
            self._rng.get("l2.vol_slow"), t_days, theta, var_x, driver
        )
        diag["half_life_days"] = cfg.ou_half_life_days
        diag["driver"] = "leverage" if driver is not None else "independent"
        self.last_diagnostics["slow_ou"] = diag
        return x

    # ------------------------------------------------------------------
    # S2: ラフ成分 (fractional OU)
    # ------------------------------------------------------------------
    def _simulate_rough(self, t: np.ndarray) -> np.ndarray:
        """定常 fOU (H ~ 0.1) を専用の物理グリッドで生成し、価格グリッドへ展開する。

        なぜ fOU か 
        -----------------------
        rough Bergomi の Volterra 過程 W^H_t は非定常 (分散が t^{2H} で増大) で、
        5000 日のシミュレーションではそのドリフトが低周波の見かけの長期記憶として
        GPH に混入する。定常な fOU なら長スケールでは指数減衰し、MSM/OU の帯域を
        汚染しない。

        なぜ専用グリッドか 
        ------------------------------
        価格グリッド (117M 点) で Davies-Harte を回すと FFT が数 GB になる。
        ラフ成分は ``rough_grid_seconds`` (既定 60 秒) の物理グリッドで生成し、
        価格グリッドへは区分定数で展開する。「ボラの粗さは 1 分まで解像され、
        それ以下では一定」というモデル化であり、steps_per_day と独立なので
        同一シードならラフ経路は解像度に依らずビット単位で一致する
        (時間スケール不変性が構造的に成立する)。

        構成
        ----
        1. fGn を Davies-Harte で厳密生成 (per-step sd = dt^H)
        2. AR(1) 指数フィルタ Y_k = a Y_{k-1} + eta g_k (a = e^{-theta dt})
        3. バーンイン 40 半減期 (e^{-40 ln 2} ~ 1e-12) を捨てて定常化。
           Y_0 を定常分布から独立に引かないのは、fOU の現在値は駆動 fGn の過去と
           相関しており、独立初期化では結合分布が壊れるため

        乱数消費は ``l2.vol_rough`` から standard_normal(2m) を一度だけ。
        """
        cfg = self._config
        rng = self._rng.get("l2.vol_rough")
        hurst = cfg.rough_hurst
        theta = math.log(2.0) / cfg.rough_half_life_days  # [1/日]
        dt_days = cfg.rough_grid_seconds / self._seconds_per_day
        eta = solve_eta_rough(hurst, theta, cfg.vol_var_target_rough)
        var_eff = rough_discrete_stationary_variance(hurst, theta, dt_days, eta)

        total_seconds = float(t[-1] - t[0])
        n_steps_rough = int(round(total_seconds / cfg.rough_grid_seconds))
        burnin = int(math.ceil(40.0 * cfg.rough_half_life_days / dt_days))

        fgn = davies_harte_fgn(burnin + n_steps_rough, hurst, rng)
        # レバレッジ (S3) 用に、スケール前の単位分散 fGn を burnin 込みで 保持
        # (whitening フィルタの先頭が文脈を必要とするため。~16MB)。
        self._last_fgn_unit = fgn.copy()
        self._last_fgn_burnin = burnin
        fgn *= dt_days**hurst  # 物理時間スケール (per-step sd = dt^H)
        a = math.exp(-theta * dt_days)
        filtered, _ = signal.lfilter([eta], [1.0, -a], fgn, zi=np.zeros(1))
        del fgn
        # グリッド点 p (p = 0..K) は Y_{burnin+p}。filtered[j] = Y_{j+1} (Y_0 = 0)。
        y = np.empty(n_steps_rough + 1, dtype=np.float64)
        y[:] = filtered[burnin - 1 : burnin + n_steps_rough]
        del filtered

        self.last_diagnostics["rough"] = {
            "hurst": hurst,
            "half_life_days": cfg.rough_half_life_days,
            "theta_per_day": theta,
            "eta": eta,
            "grid_seconds": cfg.rough_grid_seconds,
            "dt_days": dt_days,
            "target_var": cfg.vol_var_target_rough,
            "var_discrete": var_eff,
            "sample_var": float(y.var()),
            "sample_mean": float(y.mean()),
            "n_rough_points": int(y.shape[0]),
            "burnin_steps": burnin,
            "ar_coeff": a,
            # 解像度を変えても一致することが物理グリッド定義の直接証拠
            # (scale_invariance が照合する)。
            "y_digest": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
        }
        self._rough_var_eff = var_eff
        return self._expand_rough_to_grid(y, t)

    def _expand_rough_to_grid(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """ラフグリッドの値を価格グリッドへ区分定数で展開する。

        価格グリッド点 t_i にはラフ区間 ``floor(t_i / dt_r)`` の値を割り当てる。
        整数比のときは repeat / スライスで済ませ (117M 点で数百 MB の添字配列を
        作らないため)、それ以外は floor 添字で引く。
        """
        cfg = self._config
        step_sec = float(t[1] - t[0])
        n_points = int(t.shape[0])
        ratio = cfg.rough_grid_seconds / step_sec
        if abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1:
            k = int(round(ratio))  # 1 ラフ区間 = k 価格ステップ
            expanded = np.repeat(y, k)[:n_points]
        elif abs(1.0 / ratio - round(1.0 / ratio)) < 1e-9:
            m = int(round(1.0 / ratio))  # 価格ステップがラフ区間の m 倍粗い
            expanded = y[::m][:n_points].copy()
        else:
            idx = np.floor(t / cfg.rough_grid_seconds + 1e-9).astype(np.int64)
            np.clip(idx, 0, y.shape[0] - 1, out=idx)
            expanded = y[idx]
            del idx
        if expanded.shape[0] != n_points:
            raise ValueError(
                f"ラフ成分の展開点数が合いません: {expanded.shape[0]} != {n_points}"
            )
        return expanded

    # ------------------------------------------------------------------
    # S3: レバレッジ (Brownian bridge 分解) とジャンプ
    # ------------------------------------------------------------------
    @staticmethod
    def fgn_whitening_innovations(
        fgn_unit: np.ndarray, hurst: float, order: int = 96
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """fGn を AR(order) で白色化し、innovation 列 (ほぼ iid N(0,1)) を返す。

        なぜ fGn そのものと相関させないか (設計要件 からの意図的な逸脱)
        --------------------------------------------------------------------
        字義どおりセル集計を fGn 増分 G_j と相関させると、
        fGn の反持続 (lag-1 自己相関 2^{2H-1}-1 ~ -0.43) がセル集計 A_j の系列に
        rho^2 倍で乗り移り、60 秒バーのリターン自己相関が -0.21 になる
        (実測)。これは設計要件 が最重要とする (2) の不変ゲート
        (acf_r / ljung_box の S0 基準維持) と両立しない — §6 の検証表はセル内
        しか確認しておらず、セル間の fGn 継承を見落としている。

        rough Bergomi が価格の dW を W^H そのものではなく背後の駆動 BM dZ と
        相関させるのと同じ構造で、fGn の時間領域 innovation epsilon (iid) を相関
        相手にすれば、(a) z は全ラグで厳密に無相関 ((2)維持)、(b) epsilon_j は
        g_j, g_{j+1}, ... へ因果的に伝播して将来のボラだけを動かす (レバレッジ)、
        の両方が成立する。

        実装: fGn は正則ガウス過程なので whitening は因果 AR(∞)。位数 order で
        打ち切り、係数は自己共分散から solve_toeplitz (Levinson 型) で厳密に解く。
        残差自己相関は order=96 で ~1e-3 未満 (テストで固定)。

        Returns
        -------
        (epsilon, ar_coeffs, prediction_sd)
            epsilon は入力と同じ長さ (先頭 order 個は文脈不足なので、呼び出し側は
            burnin 内で捨てること)。
        """
        two_h = 2.0 * hurst
        k = np.arange(order + 1, dtype=np.float64)
        gamma = 0.5 * ((k + 1.0) ** two_h - 2.0 * k**two_h + np.abs(k - 1.0) ** two_h)
        from scipy.linalg import solve_toeplitz

        phi = solve_toeplitz(gamma[:order], gamma[1 : order + 1])
        pred_var = float(gamma[0] - np.dot(phi, gamma[1 : order + 1]))
        # epsilon_j = g_j - sum_m phi_m g_{j-m} = FIR フィルタ [1, -phi]
        eps = signal.lfilter(np.concatenate(([1.0], -phi)), [1.0], fgn_unit)
        eps /= math.sqrt(pred_var)
        return eps, phi, math.sqrt(pred_var)

    def _bridge_innovations(self, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """価格革新 z を Brownian bridge 分解で構成する (S3 設計要件 — 最重要)。

        ラフセル j に n 個の価格ステップがあるとき:

            S = sum(b), A = rho sqrt(n) eps_j + sqrt(1-rho^2) sqrt(n) w_j
            z = b - S/n + A/n

        この構成は厳密に Var(z_i)=1, Cov(z_i,z_k)=0 (i != k), sum z = A,
        corr(sum z / sqrt(n), eps_j) = rho を満たす。**共通ショックを足す実装は
        セル内に正の自己相関 (+rho^2/n) を作り (2) を壊す** — bridge 項の -1/n が
        集計項の +1/n をちょうど打ち消すのがこの分解の要点。

        相関相手 eps_j は fGn の whitening innovation
        (:meth:`fgn_whitening_innovations` — fGn 直結が (2) を壊す理由もそこに記載)。

        b は ``l2.diffusion`` から引いた列そのもの (in-place で z に変換する) なので、
        l2.diffusion の消費列は S0 以来ビット単位で不変のまま (使い方が変わる
        だけ)。セル直交成分 w は ``l2.leverage`` から。
        """
        cfg = self._config
        rho = cfg.leverage_rho_rough
        if self._last_fgn_unit is None:
            raise RuntimeError("ラフ成分が未生成です (enable_leverage には enable_rough が必要)")
        eps_full, _phi, _sd = self.fgn_whitening_innovations(
            self._last_fgn_unit, cfg.rough_hurst
        )
        burnin = self._last_fgn_burnin
        g_unit = eps_full[burnin:]
        del eps_full
        # 残差自己相関 (whitening の打ち切り誤差) を記録する。
        gc = g_unit - g_unit.mean()
        denom = float(np.dot(gc, gc))
        eps_resid_acf1 = float(np.dot(gc[:-1], gc[1:]) / denom)
        del gc
        step_sec = float(t[1] - t[0])
        ratio = cfg.rough_grid_seconds / step_sec
        if not (abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1):
            raise ValueError(
                f"enable_leverage には価格ステップ ({step_sec}s) がラフグリッド"
                f" ({cfg.rough_grid_seconds}s) を整数分割することが必要です。"
                f" 1 ステップが複数セルにまたがると bridge 分解が定義できません。"
            )
        k = int(round(ratio))
        n_steps = b.shape[0]
        if n_steps % k != 0:
            raise ValueError(f"ステップ数 {n_steps} がセル幅 {k} で割り切れません")
        n_cells = n_steps // k
        if g_unit.shape[0] != n_cells:
            raise ValueError(
                f"fGn のセル数 ({g_unit.shape[0]}) が価格側のセル数 ({n_cells}) と一致しません"
            )

        w = self._rng.get("l2.leverage").standard_normal(n_cells)
        sqrt_k = math.sqrt(k)
        a_cells = rho * sqrt_k * g_unit + math.sqrt(1.0 - rho * rho) * sqrt_k * w
        del w

        b2 = b.reshape(n_cells, k)
        cell_sums = b2.sum(axis=1)
        # z = b - S/n + A/n を in-place で (本番 117M 点、余分な 1GB を作らない)。
        adjust = (a_cells - cell_sums) / k
        b2 += adjust[:, None]
        del cell_sums

        # 実測診断 (§6.4 のゲートが参照)。z のセル集計 = A なので相関は A と eps で測る。
        corr_rough_realized = float(np.corrcoef(a_cells / sqrt_k, g_unit)[0, 1])
        self.last_diagnostics["leverage"] = {
            "rho_rough": rho,
            "rho_slow": cfg.leverage_rho_slow,
            "steps_per_cell": k,
            "n_cells": int(n_cells),
            "corr_rough_realized": corr_rough_realized,
            "correlation_target": "fgn_whitening_innovation",
            "eps_residual_acf1": eps_resid_acf1,
        }
        del a_cells, adjust
        return b

    def _simulate_mid_leverage(self, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        """中速レバレッジ成分 X_mid (日次グリッド OU、2026-08-19 設計判断)。

        なぜ必要か: 緩慢 OU (HL 30 日) は 1 日の駆動が定常 sd の
        sqrt(1-e^{-2 theta}) ~ 21% しか動かせず、ラフ fOU は反持続でショックの
        翌日への伝達が相殺されるため、per-step 相関だけでは corr(r_t, RV_{t+1})
        が理論上限 ~-0.06 でゲート帯 [-0.28, -0.16] に届かない (実測済み)。
        HL ~5 日なら 1 日の駆動が sd の sqrt(1-e^{-2 ln2/5}) ~ 49% を動かせる。

        構成 (rough の 60 秒グリッドと同型の専用物理グリッド— 日次):

            u_d = (日 d の z の和) / sqrt(n_steps_day) … 厳密 N(0,1)
            X[d+1] = a X[d] + s (rho_mid u_d + sqrt(1-rho^2) w_d)

        因果: 日 d の sigma に入るのは X[d] = 日 d-1 までの u のみ。同日の
        z と sigma の同時相関は作らない (ルックアヘッドなし、増分の条件付き
        正規性も保たれる)。分散は vol_var_target_slow の内数
        (leverage_mid_var) — 総予算は不変。

        乱数消費は ``l2.leverage_mid`` から (x0 -> w 列) の順で固定。
        """
        cfg = self._config
        rng = self._rng.get("l2.leverage_mid")
        var_mid = cfg.leverage_mid_var
        theta = math.log(2.0) / cfg.leverage_mid_half_life_days  # [1/日]
        a = math.exp(-theta)  # 日次刻み
        s = math.sqrt(var_mid * (1.0 - a * a))
        rho = cfg.leverage_rho_mid

        n_steps = z.shape[0]
        n_days = int(round((t[-1] - t[0]) / self._seconds_per_day))
        if n_steps % n_days != 0:
            raise ValueError("ステップ数が日数で割り切れません")
        spd = n_steps // n_days
        u = z.reshape(n_days, spd).sum(axis=1) / math.sqrt(spd)

        x0 = float(rng.normal(0.0, math.sqrt(var_mid)))
        w = rng.standard_normal(n_days)
        driver = rho * u + math.sqrt(1.0 - rho * rho) * w
        del w
        y, _ = signal.lfilter([s], [1.0, -a], driver, zi=np.array([a * x0]))
        x_daily = np.empty(n_days + 1, dtype=np.float64)
        x_daily[0] = x0
        x_daily[1:] = y
        del y

        corr_mid = float(np.corrcoef(u, driver)[0, 1])
        self.last_diagnostics["leverage_mid"] = {
            "half_life_days": cfg.leverage_mid_half_life_days,
            "var_mid": var_mid,
            "var_slow_remaining": cfg.vol_var_target_slow - var_mid,
            "rho_mid": rho,
            "corr_mid_realized": corr_mid,
            "ar_coeff_daily": a,
            "x0": x0,
            "sample_var": float(x_daily.var()),
            "sample_mean": float(x_daily.mean()),
            "n_days": n_days,
        }
        # 価格グリッドへ日内階段で展開 (点 i の日 = i // spd、最終点は X[n_days])。
        expanded = np.empty(n_steps + 1, dtype=np.float64)
        expanded[:-1] = np.repeat(x_daily[:n_days], spd)
        expanded[-1] = x_daily[n_days]
        return expanded

    def _z_autocorrelation(self, z: np.ndarray, n_days: int, max_lag: int = 60) -> dict:
        """価格革新 z の ACF (ラグ 1..max_lag) — bridge 実装の直接テスト (§6.4)。

        セッション行に reshape して FFT でまとめて計算する (117M 点で数秒)。
        """
        from ..validation.memory import acf as acf_fn

        steps_per_day = z.shape[0] // n_days
        usable = n_days * steps_per_day
        result = acf_fn(z[:usable].reshape(n_days, steps_per_day), max_lag=max_lag)
        values = result.get("values") or []
        conf = 2.0 / math.sqrt(z.shape[0])
        abs_vals = [abs(v) for v in values[1:] if v is not None]
        return {
            "max_abs_acf": max(abs_vals) if abs_vals else None,
            "threshold_2_over_sqrt_n": conf,
            "all_within": bool(abs_vals and max(abs_vals) < conf),
            "lag1": values[1] if len(values) > 1 else None,
            "n": int(z.shape[0]),
            "max_lag": max_lag,
        }

    def _simulate_jumps(
        self, t: np.ndarray, sigma_left: np.ndarray, dt_years: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kou 二重指数ジャンプを生成する (S3 設計要件)。

        強度はボラ変調 ``lambda(t) = lambda0 min((sigma_t/sigma_bar_diff)^rho_J, cap)``
        で、クラスタリングは既存ボラ状態から得る (自己励起 Hawkes は S11 の担当 —
        S3 では入れない §3.3)。発生は per-step Bernoulli(lambda dt)。

        Returns
        -------
        (jump_times, jump_add, compensation_per_step)
            jump_add はステップ増分に加算する配列 (ジャンプが無いステップは 0)、
            compensation_per_step は時変の補償 ``-lambda(t) k dt`` (全ステップ)。
        """
        cfg = self._config
        n_steps = sigma_left.shape[0]
        sig_bar = self.sigma_bar_diffusion

        # 強度 (in-place で構成し一時配列を減らす: ratio -> lambda(t))
        lam = sigma_left / sig_bar
        if cfg.jump_vol_exponent != 1.0:
            np.power(lam, cfg.jump_vol_exponent, out=lam)
        # 上限はボラ比の増幅に掛ける (S4 の phi もこの比に入る)。既定 cap=10 は
        # phi の最大 1.48 では滅多に binding しないが、黙って効いていると強度の
        # 設計値がずれるので、実際に binding した割合を診断に残す。
        cap_binding = float((lam > cfg.jump_intensity_cap).mean())
        np.minimum(lam, cfg.jump_intensity_cap, out=lam)
        # S4 補正 (ON 取り分 + phi の Jensen 効果) — 詳細は jump_intensity_scale。
        intensity_scale = self.jump_intensity_scale
        # S13: 共通 (システマティック) ジャンプがあるとき、資産固有側は (1−s_J)。
        # 総強度 λ0 は共通 + 固有で S12 と同じ (資産あたりの QV 予算保存 §4.3)。
        idio_share = (
            1.0 - cfg.jump_common_share if self._common is not None else 1.0
        )
        lam *= cfg.jump_lambda_per_year * intensity_scale * idio_share  # [1/年]

        u = self._rng.get("l2.jump_time").uniform(size=n_steps)
        prob = lam * dt_years
        mask = u < prob
        del u
        n_jumps = int(mask.sum())

        rng_size = self._rng.get("l2.jump_size")
        u_sign = rng_size.uniform(size=n_jumps)
        e_mag = rng_size.standard_exponential(size=n_jumps)
        up = u_sign < cfg.jump_p_up
        sizes = np.where(up, e_mag / cfg.jump_eta_up, -e_mag / cfg.jump_eta_down)

        # マルチンゲール補償 k = E[e^J] - 1 (eta_u > 1 は config が保証)。
        k_comp = (
            cfg.jump_p_up * cfg.jump_eta_up / (cfg.jump_eta_up - 1.0)
            + (1.0 - cfg.jump_p_up) * cfg.jump_eta_down / (cfg.jump_eta_down + 1.0)
            - 1.0
        )
        compensation = lam
        compensation *= -k_comp * dt_years  # in-place: lam を潰して補償列に転用

        jump_add = np.zeros(n_steps, dtype=np.float64)
        jump_add[mask] = sizes
        jump_times = t[1:][mask]  # 増分 i は区間 (t_i, t_{i+1}] — 右端に記録

        e_j2 = (
            cfg.jump_p_up * 2.0 / cfg.jump_eta_up**2
            + (1.0 - cfg.jump_p_up) * 2.0 / cfg.jump_eta_down**2
        )
        lam_eff = -float(compensation.mean()) / (k_comp * dt_years)  # 実効平均強度
        diffusion_qv = sig_bar**2
        # S13: jv_share_theory は共通ジャンプ込みの総量で報告する (固有だけだと
        # (1−s_J) 倍に見え、QV 予算の照合が誤る)。共通側の実効強度は
        # cross_factor が同じ規約 (補償平均 / (k dt)) で計算した値。
        lam_eff_common = 0.0
        if self._common is not None and self._common.jump_lam_eff > 0.0:
            lam_eff_common = self._common.jump_lam_eff
        jump_qv = (lam_eff + lam_eff_common) * e_j2
        self.last_diagnostics["jump"] = {
            "p_up": cfg.jump_p_up,
            "eta_up": cfg.jump_eta_up,
            "eta_down": cfg.jump_eta_down,
            "lambda0_per_year": cfg.jump_lambda_per_year,
            "vol_exponent": cfg.jump_vol_exponent,
            "intensity_cap": cfg.jump_intensity_cap,
            "cap_binding_fraction": cap_binding,
            "k_compensation": k_comp,
            "compensation_applied": True,
            # S4 の強度補正。1.0 なら補正なし (S0〜S3)。
            "intensity_scale_s4": intensity_scale,
            "lambda_effective_per_year": lam_eff,
            # S13: 共通ジャンプの内訳 (単一資産では 1.0 / 0.0)。
            "idio_intensity_share": idio_share,
            "lambda_effective_common_per_year": lam_eff_common,
            "n_jumps": n_jumps,
            "n_up": int(up.sum()),
            "mean_jump": float(sizes.mean()) if n_jumps else None,
            "min_jump": float(sizes.min()) if n_jumps else None,
            "max_jump": float(sizes.max()) if n_jumps else None,
            "e_j2": e_j2,
            "e_j": cfg.jump_p_up / cfg.jump_eta_up - (1.0 - cfg.jump_p_up) / cfg.jump_eta_down,
            "sigma_bar_diffusion": sig_bar,
            "jv_share_theory": jump_qv / (jump_qv + diffusion_qv),
            "jv_share_target": cfg.jump_qv_share_target,
        }
        return jump_times, jump_add, compensation

    # ------------------------------------------------------------------
    # S4: オーバーナイト・ギャップ
    # ------------------------------------------------------------------
    def _simulate_overnight(
        self, log_vol: np.ndarray, n_days: int, steps_per_day: int
    ) -> np.ndarray:
        """日 d の引けと日 d+1 の寄付の間のギャップ・リターンを生成する。

        物理時間比例 (17.5h/6.5h) では作らない 。取引の無い時間帯は
        情報時計がほとんど進まないので、別レジームとして扱う:

            r_ON(d) = sigma_ON(d) z + J_ON(d), sigma_ON(d) = c_ON sigma_close(d)

        ``c_ON`` はクローズ・トゥ・クローズ分散に占める ON の寄与が設定値に
        なるように逆算する。日中の 1 日分の分散は sigma_bar_diff^2/252 なので、

            share = var_ON / (var_ON + var_intraday)
            => var_ON = var_intraday * share/(1-share)

        ``sigma_close(d)`` に連動させることで ON にもボラ・クラスタリングが伝わり、
        ``corr(sigma_ON, sigma_close) > 0.5`` のゲートが成立する。ジャンプは
        S3 の Kou を ON 用パラメータで入れる (ギャップは実質ジャンプ的なので
        発生確率は日中より高い) — これで ON リターンの尖度が日中より高くなる。

        乱数消費は ``l0.overnight`` から (z -> ジャンプ判定 -> 符号 -> 大きさ) の順。
        """
        cfg = self._config
        rng = self._rng.get("l0.overnight")
        n_gaps = n_days - 1
        if n_gaps < 1:
            return np.zeros(0, dtype=np.float64)

        # 各日の引け時点 (その日の最終ステップ) の瞬間ボラ。
        close_idx = np.arange(1, n_days) * steps_per_day - 1
        sigma_close = np.exp(log_vol[close_idx])

        # ON の分散水準。設計上「クローズ・トゥ・クローズ分散 sigma_bar^2/252 の
        # うち share を ON が取る」なので、目標は sigma_bar^2 share / 252。
        # sigma_bar_diffusion (拡散のみ) から (1-share) 経由で逆算してはならない —
        # 日中の分散にはジャンプ分も含まれるため share がずれる。
        share = cfg.overnight_variance_share
        var_on_target = cfg.sigma_bar**2 * share / cfg.ann_days

        # ジャンプは分散シェアで指定し、そこから Kou の eta スケールを逆算する。
        # サイズ倍率で指定すると E[J^2] が ON の分散予算と噛み合わず、実測シェアが
        # 目標の 3 倍になる (実際にそうなった)。形状 (p_up, eta 比) は S3 のまま。
        jshare = cfg.overnight_jump_variance_share
        var_on_jump = var_on_target * jshare
        var_on_diffusion = var_on_target * (1.0 - jshare)
        eta_u = eta_d = k_on = 0.0
        if cfg.overnight_jump_prob > 0 and jshare > 0:
            shape_e_j2 = (
                cfg.jump_p_up * 2.0 / cfg.jump_eta_up**2
                + (1.0 - cfg.jump_p_up) * 2.0 / cfg.jump_eta_down**2
            )
            # eta' = eta / s とすると E[J^2] = s^2 * shape_e_j2。
            s = math.sqrt(var_on_jump / (cfg.overnight_jump_prob * shape_e_j2))
            eta_u = cfg.jump_eta_up / s
            eta_d = cfg.jump_eta_down / s
            if eta_u <= 1.0:
                raise ValueError(
                    f"ON ジャンプの eta_u ({eta_u:.3f}) が 1 以下です。"
                    f" E[e^J] が発散するので overnight_jump_variance_share を下げるか"
                    f" overnight_jump_prob を上げてください。"
                )
        # sigma_close は引け時点の値なので、季節性があると phi(引け) 倍に
        # 膨らんでいる (既定で 1.38、二乗で 1.90 倍)。E[sigma_close^2] =
        # phi_close^2 sigma_bar_diff^2 なので、その分を割って基準を揃える。
        # 割らないと ON シェアが目標の 1.9 倍になる (実際にそうなった)。
        phi_close = 1.0
        if cfg.enable_seasonality and hasattr(self._calendar, "phi_sigma_of_u"):
            u_close = (steps_per_day - 1) / steps_per_day
            phi_close = float(self._calendar.phi_sigma_of_u(u_close))
        c_on = math.sqrt(var_on_diffusion * cfg.ann_days) / (
            self.sigma_bar_diffusion * phi_close
        )

        z = rng.standard_normal(n_gaps)
        sigma_on = c_on * sigma_close / math.sqrt(cfg.ann_days)
        gaps = sigma_on * z

        n_on_jumps = 0
        if eta_u > 0:
            u = rng.uniform(size=n_gaps)
            mask = u < cfg.overnight_jump_prob
            n_on_jumps = int(mask.sum())
            u_sign = rng.uniform(size=n_on_jumps)
            e_mag = rng.standard_exponential(size=n_on_jumps)
            up = u_sign < cfg.jump_p_up
            gaps[mask] += np.where(up, e_mag / eta_u, -e_mag / eta_d)
            # マルチンゲール補償 (ON ジャンプ分)。忘れると価格にドリフトが乗る。
            k_on = (
                cfg.jump_p_up * eta_u / (eta_u - 1.0)
                + (1.0 - cfg.jump_p_up) * eta_d / (eta_d + 1.0)
                - 1.0
            )
            gaps -= cfg.overnight_jump_prob * k_on
            del u, u_sign, e_mag, up
        # 拡散側の凸性補正 (E[e^{r_ON}] = 1 を保つ)。
        gaps -= 0.5 * sigma_on**2

        # sigma_ON ∝ sigma_close なので構成上の相関は 1。ゲートが見るべきは
        # 観測可能な連動 (|gap| と sigma_close の相関) — こちらは z のノイズで
        # 1 より小さくなる。両方を記録して取り違えを防ぐ。
        corr_obs = float(np.corrcoef(np.abs(gaps), sigma_close)[0, 1])
        self.last_diagnostics["overnight"] = {
            "n_gaps": int(n_gaps),
            "c_on": c_on,
            "variance_share_target": share,
            "var_on_target": var_on_target,
            "var_on_diffusion": var_on_diffusion,
            "var_on_jump": var_on_jump,
            "jump_variance_share": jshare,
            "jump_prob": cfg.overnight_jump_prob,
            "jump_eta_up": eta_u,
            "jump_eta_down": eta_d,
            "k_compensation": k_on,
            "n_on_jumps": n_on_jumps,
            "sample_var": float(gaps.var()),
            "sample_mean": float(gaps.mean()),
            # 分散設計の直接の証人。検証側の variance_share は分母 (日中日次分散)
            # が右に歪んだ推定量なので系統的に上振れするが、こちらは分子だけを
            # 設計値と比べるので偏らない (6 シード実測 0.93〜1.19、平均 1.05)。
            "sample_var_over_target": float(gaps.var() / var_on_target),
            "phi_at_close": phi_close,
            "corr_sigma_on_close_by_construction": 1.0,
            "corr_abs_gap_sigma_close": corr_obs,
            "sigma_close_mean": float(sigma_close.mean()),
        }
        return gaps

    # ------------------------------------------------------------------
    # S5: 決定論的カオス成分 chi_2
    # ------------------------------------------------------------------
    def _simulate_chaos(self, n_days: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        """chi_2 を系固有グリッドで用意し、診断を残す (乱数を一切消費しない)。

        実体は :func:`prepare_chaos_component` — **生成とアンサンブル検証の両方が
        同じ関数を使う** (compose_log_sigma と同じ理由: 式を 2 か所に書くと片方だけ
        直して乖離する事故が起きる)。
        """
        chaos_t_days, chi_norm, a, c_chi, diag = prepare_chaos_component(
            self._config, n_days
        )
        self.last_diagnostics["chaos"] = diag
        return chaos_t_days, chi_norm, a, c_chi

    # ------------------------------------------------------------------
    # 拡張フック
    # ------------------------------------------------------------------
    def _log_vol_path(
        self,
        t: np.ndarray,
        ou_driver: np.ndarray | None = None,
        y_rough_pre: np.ndarray | None = None,
        x_mid: np.ndarray | None = None,
    ) -> np.ndarray:
        """log (瞬間ボラ) の経路。

        すべて対数ボラの加法成分として設計する (各成分の寄与を分散分解で
        切り分けられるようにするため。乗法で混ぜると成分の効果が分離できない)。

        - S1: MSM ``+ 0.5 sum log M_i`` と緩慢 OU ``+ X_t - Var(X)`` (実装済み)
        - S2: ラフ成分 ``+ Y_t - Var(Y)`` (実装済み)
        - S3: 基準は ``log sigma_diff`` (ジャンプ有効時は sqrt(1-JV) 縮小)
        - S5: カオス成分 chi_2 ``+ c * g(chi_2(t))``

        Parameters
        ----------
        ou_driver:
            レバレッジ有効時の OU 駆動列 (価格革新と相関済み)。
        y_rough_pre:
            レバレッジ有効時、bridge が fGn を必要とするためラフ成分は先に生成
            される。その展開済み配列を受け取り再生成しない (ストリーム消費を
            二重にしないため)。
        """
        cfg = self._config
        n = t.shape[0]
        log_sigma_bar = math.log(self.sigma_bar_diffusion)

        # 早期リターンは「log σ に何も足さない」場合のみ。カオス (S5) と季節性
        # (S4) は確率成分が無くても log σ を変えるので、ここを通ってはならない
        # (通すと有効フラグが暗黙 no-op になる — このプロジェクトの禁止事項)。
        if not (
            cfg.enable_msm
            or cfg.enable_slow_ou
            or cfg.enable_rough
            or cfg.enable_chaos_vol
            or (cfg.enable_seasonality and hasattr(self._calendar, "phi_sigma_of_u"))
        ):
            return np.full(n, log_sigma_bar, dtype=np.float64)

        t_days = t / self._seconds_per_day
        half_log_msm: np.ndarray | float = 0.0
        x_slow: np.ndarray | float = 0.0
        y_rough: np.ndarray | float = 0.0
        var_slow = 0.0
        var_rough = 0.0
        # 生成順は OU -> MSM。ストリームは名前ごとに独立なので消費列は順序に
        # 依存せず (rng.py の設計、test_stream_order_does_not_matter が固定)、
        # 出力はビット単位で不変。OU の一時配列 (z/y/x) と MSM の出力配列が同時に
        # 生きる時間を無くすことで、本番設定 (1 配列 936MB) のピークが 5.6 -> 4.7GB
        # に下がる。
        if cfg.enable_slow_ou:
            # 中速成分 (S3 レバレッジ) がある場合、slow の予算はその残り。
            slow_var = (
                cfg.vol_var_target_slow - cfg.leverage_mid_var
                if x_mid is not None
                else None
            )
            # S13: 共通 OU があるときの固有側は残り (1−f_c) 分。
            if self._common is not None and self._common.ou_common_var > 0.0:
                slow_var = cfg.vol_var_target_slow - self._common.ou_common_var
            x_slow = self._simulate_slow_ou(t_days, driver=ou_driver, var_override=slow_var)
            # 凸性補正は OU 族の合計分散 (共通 + 固有 = 総予算) に対して行う。
            # x_mid の場合のみ、後段の加算ブロックが mid の分を足し戻す。
            var_slow = cfg.vol_var_target_slow if x_mid is None else slow_var
            # S13: 共通 OU を加算 (固有配列は自分の所有なので in-place 可)。
            if self._common is not None and isinstance(self._common.x_slow, np.ndarray):
                x_slow += self._common.x_slow
        if cfg.enable_msm:
            # S13: 共有分割 — 固有は速い側 [k_c, k)、共通 (遅い側 [0, k_c)) を加算。
            if self._common is not None and self._common.msm_k_common > 0:
                k_c = self._common.msm_k_common
                if k_c < cfg.msm_k:
                    half_log_msm = self._simulate_msm(
                        t_days, component_range=(k_c, cfg.msm_k)
                    )
                    half_log_msm += self._common.half_log_msm
                else:
                    half_log_msm = self._common.half_log_msm.copy()
            else:
                half_log_msm = self._simulate_msm(t_days)
        if x_mid is not None:
            # 合成上は slow チャンネル (OU 族) に合算する。凸性補正も加算。
            if isinstance(x_slow, np.ndarray):
                x_slow = x_slow + x_mid
            else:
                x_slow = x_mid
            var_slow += cfg.leverage_mid_var
        if cfg.enable_rough:
            y_rough = y_rough_pre if y_rough_pre is not None else self._simulate_rough(t)
            var_rough = self._rough_var_eff

        # 診断用サブサンプル (分単位)。成分内訳を全ステップ保持すると本番設定で
        # 数 GB になるため間引く。検証スイートの path 診断がこれを使う。
        # 合成の前に採る: 下の合成は half_log_msm の配列を書き換えるので、
        # あとから採ると成分内訳ではなく log sigma そのものになってしまう。
        step_seconds = float(t[1] - t[0])
        stride = max(int(round(VOL_SUBSAMPLE_SECONDS / step_seconds)), 1)
        n_sub = t_days[::stride].shape[0]
        subsample = {
            "stride": stride,
            "step_seconds": step_seconds,
            "t_days": t_days[::stride].copy(),
            "half_log_msm": (
                half_log_msm[::stride].copy()
                if isinstance(half_log_msm, np.ndarray)
                else np.zeros(n_sub)
            ),
            # x_slow は OU 族チャンネルの合算 (S3 レバレッジ有効時は HL30 + 中速)。
            "x_slow": (
                x_slow[::stride].copy() if isinstance(x_slow, np.ndarray) else np.zeros(n_sub)
            ),
            "x_mid": (
                x_mid[::stride].copy() if isinstance(x_mid, np.ndarray) else np.zeros(n_sub)
            ),
            "y_rough": (
                y_rough[::stride].copy() if isinstance(y_rough, np.ndarray) else np.zeros(n_sub)
            ),
        }

        # MSM の配列は自分で確保したものなので、そのまま合成先として使い回す。
        log_vol = np.asarray(
            compose_log_sigma(
                log_sigma_bar, half_log_msm, x_slow, var_slow,
                y_rough, var_rough,
                inplace=isinstance(half_log_msm, np.ndarray),
            ),
            dtype=np.float64,
        )
        if log_vol.ndim == 0:
            # 確率成分が全て無効 (カオス/季節性のみ) の場合、compose はスカラーを
            # 返すので配列に展開する。
            log_vol = np.full(n, float(log_vol), dtype=np.float64)
        # 展開済みラフ配列 (本番で 936MB) はもう不要。
        if isinstance(y_rough, np.ndarray):
            del y_rough

        # S5: 決定論的カオス成分 chi_2。確率成分と同じ log 加算だが、凸性補正は
        # ガウス公式ではなく数値 (c_chi) — 詳細は _simulate_chaos。φ の前に足す
        # (式: chi は log σ_stoch の一部で、φ はその全体に掛かる)。
        # 本番では 1 配列 936MB なので、補間はチャンクで in-place 加算する。
        if cfg.enable_chaos_vol:
            chaos_t_days, chi_norm, a_chi, c_chi = self._simulate_chaos(float(t_days[-1]))
            chunk = 8_000_000
            for i0 in range(0, n, chunk):
                i1 = min(i0 + chunk, n)
                log_vol[i0:i1] += a_chi * np.interp(
                    t_days[i0:i1], chaos_t_days, chi_norm
                )
            log_vol -= c_chi
            subsample["chi_term"] = a_chi * np.interp(
                t_days[::stride], chaos_t_days, chi_norm
            )
            subsample["c_chi"] = c_chi
            del chaos_t_days, chi_norm
        else:
            subsample["chi_term"] = np.zeros(n_sub)
            subsample["c_chi"] = 0.0
        # S4: 日内季節性を観測ボラへの乗法変調として掛ける。
        # log sigma_obs = log phi_sigma(u) + log sigma_stoch。
        # 確率ボラ成分そのものには掛けない (§3) — phi で割れば S3 の系列が
        # 完全に復元でき、それが S4 のゲートの検定力の源になる。
        if cfg.enable_seasonality and hasattr(self._calendar, "phi_sigma_of_u"):
            u = self._calendar.intraday_position(t)
            log_phi = np.log(self._calendar.phi_sigma_of_u(u))
            del u
            log_vol += log_phi
            subsample["log_phi_sigma"] = log_phi[::stride].copy()
            mean_log_phi = float(log_phi.mean())
            del log_phi
        else:
            subsample["log_phi_sigma"] = np.zeros(n_sub)
            mean_log_phi = 0.0

        subsample["log_vol"] = log_vol[::stride].copy()
        self.last_diagnostics["vol_subsample"] = subsample
        # S10c: E[log σ_obs] の決定論的部分。MSM は E[ΠM]=1 だが E[log M]<0
        # (Jensen の逆側) なので、log 平均には固有の負の定数が乗る。
        # V_t の中心化定数 m_V はこれを使う (全標本平均は使わない — 因果性)。
        if cfg.enable_msm:
            m0_msm = solve_m0(cfg.msm_k, cfg.vol_var_target_msm)
            msm_log_mean = 0.5 * cfg.msm_k * 0.5 * (
                math.log(m0_msm) + math.log(2.0 - m0_msm)
            )
        else:
            msm_log_mean = 0.0
        self.last_diagnostics["composition"] = {
            "log_sigma_bar": log_sigma_bar,
            "mean_log_vol_deterministic": (
                log_sigma_bar + msm_log_mean
                - var_slow - var_rough - float(subsample["c_chi"])
                + mean_log_phi
            ),
            "msm_log_mean": msm_log_mean,
            "mean_log_phi_sigma": mean_log_phi,
            "convexity_correction": -var_slow - var_rough - float(subsample["c_chi"]),
            "c_chi": float(subsample["c_chi"]),
            "enable_msm": cfg.enable_msm,
            "enable_slow_ou": cfg.enable_slow_ou,
            "enable_rough": cfg.enable_rough,
            "enable_seasonality": cfg.enable_seasonality,
            "enable_chaos_vol": cfg.enable_chaos_vol,
        }
        return log_vol

    # ------------------------------------------------------------------
    # S13: 因子合成
    # ------------------------------------------------------------------
    def _compose_factor_innovation(self, z: np.ndarray) -> np.ndarray:
        """z_i = β_i z_F + √(1−β_i²) z_i^idio 。

        z (固有チャネル — bridge 済み) を in-place で合成後の革新に変換する。
        両項とも線形なので、固有側の bridge が保証するセル内無相関・単位分散は
        合成後も厳密に保たれる (z_F は iid、bridge 項と独立)。資産間の per-step
        相関は全スケールで β_i β_j。

        β=0 は完全スキップ (乗算すら行わない) — 退化テスト (§8.3) が
        因子経路を通しても資産 0 は S12 とビット単位一致を固定する。
        """
        if self._common is None or self._factor_beta == 0.0:
            return z
        beta = self._factor_beta
        s_id = math.sqrt(1.0 - beta * beta)
        z_f = self._common.z_f
        if z_f.shape[0] != z.shape[0]:
            raise ValueError(
                f"共通因子の長さ ({z_f.shape[0]}) がステップ数 ({z.shape[0]}) と一致しません"
            )
        # 本番では 1 配列 187MB (1000日)。一時配列をチャンクに抑えて合成する。
        chunk = 8_000_000
        for i0 in range(0, z.shape[0], chunk):
            i1 = min(i0 + chunk, z.shape[0])
            seg = z[i0:i1]
            seg *= s_id
            seg += beta * z_f[i0:i1]
        return z

    # ------------------------------------------------------------------
    def simulate(self, t: np.ndarray) -> PriceProcess:
        """時刻グリッド ``t`` (秒) 上で log p* を生成する。

        構成は対数価格の増分:
        ``log_p[i+1] = log_p[i] + (mu - 0.5 sigma_i^2 - lambda_i k) dt
        + sigma_i sqrt(dt) z_i + J_i``

        ボラは区間の左端の値を使う (Euler-Maruyama)。右端や区間平均を使うと
        レバレッジで未来のボラ情報が当該区間のリターンへ漏れる (ルックアヘッド)。

        拡散乱数は ``l2.diffusion`` から最初に n-1 個を一括で引く。レバレッジ
        有効時はその列 b を bridge 分解 (§6) で z に変換するが、**消費列そのものは
        S0 以来不変** (rng_diffusion ゲートがこれを検証する)。無効時は z = b で
        S2 までの経路とビット単位同一。
        """
        if t.ndim != 1 or t.shape[0] < 2:
            raise ValueError("時刻グリッドは 1 次元で 2 点以上必要です")
        self.last_diagnostics = {}
        cfg = self._config

        n = int(t.shape[0])
        z = self._rng.get("l2.diffusion").standard_normal(n - 1)
        self.last_diagnostics["diffusion_digest"] = hashlib.sha256(
            np.ascontiguousarray(z).tobytes()
        ).hexdigest()

        if cfg.enable_leverage:
            # ラフ成分を先に生成 (bridge が fGn を必要とするため)。順序を変えても
            # ストリームは名前ごとに独立なので、各ストリームの消費列は不変。
            y_rough_pre = self._simulate_rough(t)
            z = self._bridge_innovations(z, t)

            # 長期チャンネル: OU の駆動 xi = rho_slow z + sqrt(1-rho^2) w2。
            # z を先に構成してから xi を導出する (順序を逆にしない — §6.3)。
            # S13: xi は固有チャネルの z (合成前) から作る — 固有 OU の駆動
            # (固有チャネルに同じ ρ)。共通 OU の駆動は
            # cross_factor が z_F から同じ ρ_slow で作る (共通チャネル側)。
            rho_s = cfg.leverage_rho_slow
            xi = self._rng.get("l2.leverage_slow").standard_normal(n - 1)
            xi *= math.sqrt(1.0 - rho_s * rho_s)
            scaled = z * rho_s
            xi += scaled
            del scaled
            corr_slow = float(
                (np.dot(z, xi) / z.shape[0] - z.mean() * xi.mean())
                / (z.std() * xi.std())
            )
            # 中速レバレッジ成分 (日次グリッド)。既定は無効 (var=0 — (3) 保全の
            # 裁定 2026-08-20)。有効時のみ z から前日集計 u_d を作る。
            x_mid = (
                self._simulate_mid_leverage(z, t)
                if cfg.leverage_mid_var > 0.0
                else None
            )
            # S13: 因子合成 z_i = β z_F + √(1−β²) z_id (§4.1)。xi (固有 OU 駆動) の
            # 構成後・z_acf の測定前に行う — z_acf は合成後のセル内無相関
            # (各資産で個別に再検証すべき箇所 §4.1) を測る。
            z = self._compose_factor_innovation(z)
            log_vol = self._log_vol_path(
                t, ou_driver=xi, y_rough_pre=y_rough_pre, x_mid=x_mid
            )
            del xi, y_rough_pre, x_mid

            lev_diag = self.last_diagnostics["leverage"]
            lev_diag["corr_slow_realized"] = corr_slow
            if self._common is not None:
                # 合成後の周辺レバレッジの理論係数 (記録)。ラフは固有チャネルに
                # しか存在しない (§4.3 共有禁止) ため √(1−β²) 希釈が構造的に残る。
                b_ = self._factor_beta
                lev_diag["rho_rough_marginal_theory"] = (
                    cfg.leverage_rho_rough * math.sqrt(1.0 - b_ * b_)
                )
            # z のセル内無相関の直接テスト (§6.4)。増分構築で z を潰す前に測る。
            n_days_grid = int(round((t[-1] - t[0]) / self._seconds_per_day))
            lev_diag["z_acf"] = self._z_autocorrelation(z, max(n_days_grid, 1))
        else:
            z = self._compose_factor_innovation(z)
            log_vol = self._log_vol_path(t)

        sigma_left = np.exp(log_vol[:-1])

        dt_sec = np.diff(t)
        if dt_sec.min() <= 0:
            raise ValueError("時刻グリッドが単調増加ではありません")
        uniform = dt_sec.min() == dt_sec.max()
        mu = float(self._config.mu_drift)

        # ジャンプは sigma_left に依存するので増分構築の前に生成する
        # (増分構築は sigma_left を in-place で潰すため)。
        jump_times = np.empty(0, dtype=np.float64)
        jump_add: np.ndarray | None = None
        jump_compensation: np.ndarray | None = None
        if cfg.enable_jump:
            if not uniform:
                raise NotImplementedError("非一様グリッドのジャンプは S4 で対応します")
            dt_y_scalar = float(dt_sec[0]) / self._seconds_per_year
            jump_times, jump_add, jump_compensation = self._simulate_jumps(
                t, sigma_left, dt_y_scalar
            )

        if uniform and np.all(sigma_left == sigma_left[0]):
            # S0 経路: 全部スカラーで済むので中間配列を作らない。
            dt_y = float(dt_sec[0]) / self._seconds_per_year
            sigma = float(sigma_left[0])
            increments = z  # 以降は in-place で書き換える
            increments *= sigma * math.sqrt(dt_y)
            increments += (mu - 0.5 * sigma * sigma) * dt_y
        elif uniform:
            # S1 経路: sigma が時間変動。一時配列を増やさないよう in-place で組む。
            dt_y = float(dt_sec[0]) / self._seconds_per_year
            increments = z
            increments *= sigma_left  # z * sigma
            increments *= math.sqrt(dt_y)  # 拡散項完成
            drift = sigma_left  # sigma_left を潰してドリフト項に転用する
            np.square(drift, out=drift)
            drift *= -0.5 * dt_y
            drift += mu * dt_y
            increments += drift
            del drift, sigma_left
        else:
            dt_y = dt_sec / self._seconds_per_year
            increments = (mu - 0.5 * sigma_left**2) * dt_y + sigma_left * np.sqrt(dt_y) * z

        if jump_add is not None:
            # 補償 -lambda(t) k dt を先に、次にジャンプ本体。忘れると価格に系統
            # ドリフトが乗る (§4.3) — テストが補償 on/off の終端差 = sum(lambda k dt)
            # を厳密に検証する。
            increments += jump_compensation
            increments += jump_add
            del jump_add, jump_compensation

        # S13: 共通 (システマティック) ジャンプ — 全資産に同一の対数サイズで
        # 入る (§4.3市場全体のニュース)。補償も共通側で計算済み (同一配列を
        # 全資産が読む — increments += は共有配列を変異させない)。
        if (
            self._common is not None
            and self._common.jump_idx is not None
            and cfg.enable_jump
        ):
            increments += self._common.jump_comp
            if self._common.jump_idx.size:
                increments[self._common.jump_idx] += self._common.jump_sizes
                jump_times = np.sort(
                    np.concatenate([jump_times, t[1:][self._common.jump_idx]])
                )

        log_p = np.empty(n, dtype=np.float64)
        log_p[0] = math.log(self._config.p0)
        np.cumsum(increments, out=log_p[1:])
        log_p[1:] += log_p[0]

        # S4: オーバーナイト・ギャップ。log_p には足さず別配列として返す
        # (PriceProcess.overnight_gaps の docstring に理由を記載)。
        gaps = np.empty(0, dtype=np.float64)
        if cfg.enable_overnight:
            n_days_grid = int(round((t[-1] - t[0]) / self._seconds_per_day))
            steps_per_day = (n - 1) // n_days_grid
            gaps = self._simulate_overnight(log_vol, n_days_grid, steps_per_day)

        comp = self.last_diagnostics.get("composition") or {}
        return PriceProcess(
            t=t,
            log_p_star=log_p,
            log_vol=log_vol,
            jump_times=jump_times,
            overnight_gaps=gaps,
            interpolation="linear",
            mean_log_vol_deterministic=comp.get(
                "mean_log_vol_deterministic", math.log(self.sigma_bar_diffusion)
            ),
        )


def build_price_layer(
    config: Config,
    rng: RNGRegistry,
    calendar: ConstantCalendar,
    activity: ConstantActivity,
    factor=None,
) -> GBMPriceLayer:
    if config.enable_overnight and not config.enable_jump:
        raise ValueError(
            "enable_overnight=True には enable_jump=True が必要です"
            " (ON ジャンプは S3 の Kou パラメータを ON 用倍率で使うため)"
        )
    return GBMPriceLayer(config, rng, calendar, activity, factor=factor)
