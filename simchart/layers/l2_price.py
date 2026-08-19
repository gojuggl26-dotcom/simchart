"""L2: 情報価格層。

S1 までの中身
-------------
S0: 定数ボラ・正規革新の幾何ブラウン運動。
S1: 対数ボラに MSM (Markov-Switching Multifractal) と緩慢 OU を加算。

    log sigma_t = log sigma_bar + 0.5 * sum_i log M_i(t) + X_t - Var(X)
                                                           ^^^^^^^^ 凸性補正

これで |r| の長期記憶 (べき則的 ACF)・ボラティリティ・クラスタリング・
集計正規性・マルチフラクタルスケーリングが初めて現れる。**革新項 z は正規のまま。**
テールはボラ過程 (と S3 のジャンプ) から内生的に出す。t 分布などを外生的に入れると
時間集計で尖度が下がる性質が永久に再現できなくなる。

時間スケール不変性 (最重要の設計制約)
------------------------------------
MSM の切替強度 gamma_i と OU の theta は**物理時間 (1 日 = 1 セッション) で定義**し、
グリッド刻みへの変換は実装内部で行う。「1 ステップあたり切替確率」で実装すると
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

from ..config import TRADING_DAYS_PER_YEAR, Config
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
    "VOL_SUBSAMPLE_SECONDS",
]

#: 診断用サブサンプルの間隔 (秒)。全ステップの成分内訳を保持すると本番設定で
#: 数 GB になるため、分単位に間引いて保存する (指示書 §8)。
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
    離散化の分だけずれるので、**凸性補正とアンサンブル断面はこちらを使う** —
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

    乱数消費: ``rng.standard_normal(2m)`` を**一度だけ** (m は n 以上の FFT 高速長)。
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


def compose_log_sigma(
    log_sigma_bar: float,
    half_log_msm: np.ndarray | float,
    x_slow: np.ndarray | float,
    var_slow: float,
    y_rough: np.ndarray | float = 0.0,
    var_rough: float = 0.0,
    inplace: bool = False,
) -> np.ndarray | float:
    """log sigma の合成式。**生成とアンサンブル検証の両方がこの 1 つを使う。**

    式を 2 か所に書くと、片方だけ直して乖離する事故が起きる。凸性補正
    -Var(X) - Var(Y) はガウス成分 (OU とラフ fOU) のためのもの:
    E[e^{2X}] = e^{2Var(X)} != 1 なので、引かないと実効ボラが e^{Var} 倍に膨らむ。
    MSM 側は E[prod M_i] = 1 なので補正不要。**var_rough には生成される離散過程の
    実分散** (:func:`rough_discrete_stationary_variance`) **を渡すこと** — 連続式の
    目標値を渡すと離散化の分だけ E[sigma^2] がずれる。

    ``inplace=True`` のとき ``half_log_msm`` を書き換えて返す (呼び出し側が所有権を
    渡す)。本番設定では 1 配列 936MB なので、素直に書くと中間結果だけで数 GB を
    使ってしまうため。**両経路の結果はビット単位で一致する**: IEEE754 の加算は
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
    ) -> None:
        self._config = config
        self._rng = rng
        self._calendar = calendar
        self._activity = activity
        #: 秒 -> 年 の換算。年 = 252 立会日 x 1 セッションの秒数。
        self._seconds_per_year = TRADING_DAYS_PER_YEAR * calendar.session_seconds()
        #: 秒 -> 日 の換算。gamma_i と theta は「1 日あたり」の物理時間定義 (§7)。
        self._seconds_per_day = calendar.session_seconds()
        #: 直近の simulate() の診断。pipeline が StageResult.meta に回収する。
        self.last_diagnostics: dict[str, Any] = {}
        #: 直近に生成したラフ成分の離散定常分散 (凸性補正に使う)。
        self._rough_var_eff: float = 0.0

    # ------------------------------------------------------------------
    # S1: MSM 成分
    # ------------------------------------------------------------------
    def _simulate_msm(self, t_days: np.ndarray) -> np.ndarray:
        """0.5 * sum_i log M_i(t) をグリッド上で生成する (切替時刻ベース)。

        per-step の Bernoulli ループは k*N 回の乱数生成になり 11.7M ステップでは
        非現実的。成分ごとに (1) 切替回数 ~ Poisson(gamma_i * T)、(2) 切替時刻 ~
        Uniform(0, T) ソート、(3) 各区間の値を等確率で引く、の順で生成する。
        **この生成はグリッド解像度に一切依存しない** (t_days の値でしか使わない)
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

        **この経路変更は出力をビット単位で変えない。** 統合区間の値は
        ``(((0 + v_0) + v_1) + ... ) * 0.5`` を成分順に計算しており、素朴版が
        格子点ごとに行う浮動小数演算と順序も含めて同一だからである
        (tests/test_s1_vol.py がリファレンス実装との一致を固定している)。
        """
        cfg = self._config
        rng = self._rng.get("l2.vol_msm")
        k = cfg.msm_k
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

        for i in range(k):
            gamma_i = cfg.msm_gamma1_per_day * cfg.msm_b**i
            n_switch = int(rng.poisson(gamma_i * T))
            switch_times = np.sort(rng.uniform(0.0, T, n_switch))
            # 区間は n_switch + 1 個。先頭が初期値で、定常分布 (等確率) から引く。
            states = rng.integers(0, 2, n_switch + 1)

            # 切替 m 以降の状態を使い始める最初の格子点。
            # bounds[m] <= j  <=>  switch_times[m] <= t_days[j] なので、
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
        self.last_diagnostics["msm"] = {
            "k": k,
            "b": cfg.msm_b,
            "gamma_per_day": [cfg.msm_gamma1_per_day * cfg.msm_b**i for i in range(k)],
            "m0": m0,
            "target_var_log_sigma": cfg.vol_var_target_msm,
            "theoretical_var_log_sigma": msm_theoretical_var_log_sigma(k, m0),
            "n_switches": n_switches,
            "expected_switches": [
                cfg.msm_gamma1_per_day * cfg.msm_b**i * T for i in range(k)
            ],
            "occupancy_hi": occupancy_hi,
            "horizon_days": T,
            "n_merged_segments": int(starts.size),
            # 切替過程のダイジェスト。解像度を変えても一致することが
            # 「物理時間定義」の直接証拠になる (test_scale_invariance)。
            "switch_digest": switch_hash.hexdigest(),
        }
        return total

    # ------------------------------------------------------------------
    # S1: 緩慢 OU 成分
    # ------------------------------------------------------------------
    def _simulate_slow_ou(self, t_days: np.ndarray) -> np.ndarray:
        """X_t (平均 0 の OU) をグリッド上で生成する (厳密離散化)。

        遷移はガウスで閉形式なので、Euler-Maruyama ではなく

            X_{t+D} = X_t e^{-theta D} + sqrt(Var(X) (1 - e^{-2 theta D})) z

        を使う。これはどんな刻み D でも分布が厳密で、グリッド解像度に依存しない。
        初期値 X_0 は定常分布 N(0, Var(X)) から引く (バーンイン不要)。
        逐次再帰は AR(1) なので scipy.signal.lfilter で O(N)。

        乱数消費は ``l2.vol_slow`` から (X_0 -> z 列) の順で固定。
        """
        cfg = self._config
        rng = self._rng.get("l2.vol_slow")
        theta = math.log(2.0) / cfg.ou_half_life_days  # [1/日]
        var_x = cfg.vol_var_target_slow

        dt = np.diff(t_days)
        if dt.size and abs(dt.min() - dt.max()) > 1e-12 * max(dt.max(), 1.0):
            raise NotImplementedError(
                "非一様グリッド上の OU は S4 (オーバーナイト導入) で実装します。"
                " 厳密離散化自体は刻みごとの係数で対応可能です。"
            )
        step_days = float(dt[0])
        a = math.exp(-theta * step_days)
        s = math.sqrt(var_x * (1.0 - a * a))

        x0 = float(rng.normal(0.0, math.sqrt(var_x)))
        z = rng.standard_normal(t_days.shape[0] - 1)
        # X_j = a X_{j-1} + s z_j  (j >= 1) を lfilter で。zi = [a * x0] により
        # y[0] = s z[0] + a x0 = X_1 となる。
        y, _ = signal.lfilter([s], [1.0, -a], z, zi=np.array([a * x0]))
        # z はもう不要。本番では 1 配列 936MB なので、x を確保する前に解放して
        # 同時に生きる大配列を 3 本から 2 本に減らす (値には影響しない)。
        del z
        x = np.empty(t_days.shape[0], dtype=np.float64)
        x[0] = x0
        x[1:] = y
        del y

        self.last_diagnostics["slow_ou"] = {
            "theta_per_day": theta,
            "half_life_days": cfg.ou_half_life_days,
            "target_var": var_x,
            "eta_per_sqrt_day": math.sqrt(2.0 * theta * var_x),
            "step_days": step_days,
            "ar_coeff": a,
            "x0": x0,
            "sample_var": float(x.var()),
            "sample_mean": float(x.mean()),
        }
        return x

    # ------------------------------------------------------------------
    # S2: ラフ成分 (fractional OU)
    # ------------------------------------------------------------------
    def _simulate_rough(self, t: np.ndarray) -> np.ndarray:
        """定常 fOU (H ~ 0.1) を専用の物理グリッドで生成し、価格グリッドへ展開する。

        なぜ fOU か (指示書 §4)
        -----------------------
        rough Bergomi の Volterra 過程 W^H_t は非定常 (分散が t^{2H} で増大) で、
        5000 日のシミュレーションではそのドリフトが低周波の見かけの長期記憶として
        GPH に混入する。定常な fOU なら長スケールでは指数減衰し、MSM/OU の帯域を
        汚染しない。

        なぜ専用グリッドか (指示書 §6)
        ------------------------------
        価格グリッド (117M 点) で Davies-Harte を回すと FFT が数 GB になる。
        ラフ成分は ``rough_grid_seconds`` (既定 60 秒) の**物理グリッド**で生成し、
        価格グリッドへは区分定数で展開する。「ボラの粗さは 1 分まで解像され、
        それ以下では一定」というモデル化であり、**steps_per_day と独立**なので
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
            # 解像度を変えても一致することが「物理グリッド定義」の直接証拠
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
    # 拡張フック
    # ------------------------------------------------------------------
    def _log_vol_path(self, t: np.ndarray) -> np.ndarray:
        """log (瞬間ボラ) の経路。

        すべて対数ボラの**加法**成分として設計する (各成分の寄与を分散分解で
        切り分けられるようにするため。乗法で混ぜると成分の効果が分離できない)。

        - S1: MSM ``+ 0.5 sum log M_i`` と緩慢 OU ``+ X_t - Var(X)`` (実装済み)
        - S2: ラフ成分 ``+ nu * W^H_t`` (H ~ 0.1 の分数ブラウン運動)
        - S5: カオス成分 chi_2 ``+ c * g(chi_2(t))``
        """
        cfg = self._config
        n = t.shape[0]
        log_sigma_bar = math.log(cfg.sigma_bar)

        if not (cfg.enable_msm or cfg.enable_slow_ou or cfg.enable_rough):
            return np.full(n, log_sigma_bar, dtype=np.float64)

        t_days = t / self._seconds_per_day
        half_log_msm: np.ndarray | float = 0.0
        x_slow: np.ndarray | float = 0.0
        y_rough: np.ndarray | float = 0.0
        var_slow = 0.0
        var_rough = 0.0
        if cfg.enable_msm:
            half_log_msm = self._simulate_msm(t_days)
        if cfg.enable_slow_ou:
            x_slow = self._simulate_slow_ou(t_days)
            var_slow = cfg.vol_var_target_slow
        if cfg.enable_rough:
            y_rough = self._simulate_rough(t)
            var_rough = self._rough_var_eff

        # 診断用サブサンプル (分単位)。成分内訳を全ステップ保持すると本番設定で
        # 数 GB になるため間引く。検証スイートの path 診断がこれを使う。
        # **合成の前に採る**: 下の合成は half_log_msm の配列を書き換えるので、
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
            "x_slow": (
                x_slow[::stride].copy() if isinstance(x_slow, np.ndarray) else np.zeros(n_sub)
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
        # 展開済みラフ配列 (本番で 936MB) はもう不要。
        if isinstance(y_rough, np.ndarray):
            del y_rough
        subsample["log_vol"] = log_vol[::stride].copy()
        self.last_diagnostics["vol_subsample"] = subsample
        self.last_diagnostics["composition"] = {
            "log_sigma_bar": log_sigma_bar,
            "convexity_correction": -var_slow - var_rough,
            "enable_msm": cfg.enable_msm,
            "enable_slow_ou": cfg.enable_slow_ou,
            "enable_rough": cfg.enable_rough,
        }
        return log_vol

    def _jump_component(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ジャンプの (時刻, 各グリッド区間に加える対数ジャンプ量)。

        S3 で ``l2.jump_time`` / ``l2.jump_size`` ストリームを使って実装する。
        線形補間がジャンプをなますため、**ジャンプ時刻は必ずグリッド点に載せる**
        こと (:class:`~simchart.types.PriceProcess` の docstring を参照)。
        """
        return np.empty(0, dtype=np.float64), np.zeros(t.shape[0] - 1, dtype=np.float64)

    def _leverage_innovation(self, z: np.ndarray) -> np.ndarray:
        """レバレッジ効果のためにボラ革新と価格革新を相関させる。

        S3 で実装する。``l2.leverage`` ストリームで直交成分を引き、
        ``z_vol = rho * z_price + sqrt(1 - rho^2) * z_orth`` の形にする。
        価格側の系列 ``z`` を書き換えてはならない (書き換えると S2 との比較で
        拡散経路が変わってしまい、段階間比較が壊れる)。
        """
        return z

    # ------------------------------------------------------------------
    def simulate(self, t: np.ndarray) -> PriceProcess:
        """時刻グリッド ``t`` (秒) 上で log p* を生成する。

        構成は対数価格の増分:
        ``log_p[i+1] = log_p[i] + (mu - 0.5 sigma_i^2) dt + sigma_i sqrt(dt) z_i``

        ボラは区間の**左端**の値を使う (Euler-Maruyama)。右端や区間平均を使うと
        S3 でレバレッジを入れたときに未来のボラ情報が当該区間のリターンへ漏れる
        (ルックアヘッド)。S1 では左端規約が実際に効いている。

        拡散乱数 ``z`` は ``l2.diffusion`` から**最初に n-1 個を一括で**引く。
        MSM / OU は別ストリームなので、S1 のフラグを立てても z の系列は S0 と
        ビット単位で同一になる (rng_diffusion ゲートがこれを検証する)。
        """
        if t.ndim != 1 or t.shape[0] < 2:
            raise ValueError("時刻グリッドは 1 次元で 2 点以上必要です")
        self.last_diagnostics = {}

        n = int(t.shape[0])
        z = self._rng.get("l2.diffusion").standard_normal(n - 1)
        self.last_diagnostics["diffusion_digest"] = hashlib.sha256(
            np.ascontiguousarray(z).tobytes()
        ).hexdigest()

        log_vol = self._log_vol_path(t)
        sigma_left = np.exp(log_vol[:-1])

        dt_sec = np.diff(t)
        if dt_sec.min() <= 0:
            raise ValueError("時刻グリッドが単調増加ではありません")
        uniform = dt_sec.min() == dt_sec.max()
        mu = float(self._config.mu_drift)

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

        jump_times, jump_increments = self._jump_component(t)
        if jump_increments.any():
            increments = increments + jump_increments

        log_p = np.empty(n, dtype=np.float64)
        log_p[0] = math.log(self._config.p0)
        np.cumsum(increments, out=log_p[1:])
        log_p[1:] += log_p[0]

        return PriceProcess(
            t=t,
            log_p_star=log_p,
            log_vol=log_vol,
            jump_times=jump_times,
            interpolation="linear",
        )


def build_price_layer(
    config: Config,
    rng: RNGRegistry,
    calendar: ConstantCalendar,
    activity: ConstantActivity,
) -> GBMPriceLayer:
    if config.enable_jump or config.enable_leverage:
        raise NotImplementedError("ジャンプ / レバレッジは S3 で simchart/layers/l2_price.py に実装します。")
    if config.enable_chaos_vol:
        raise NotImplementedError("カオス的ボラ成分 chi_2 は S5 で実装します。")
    return GBMPriceLayer(config, rng, calendar, activity)
