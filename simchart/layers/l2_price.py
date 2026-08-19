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


def compose_log_sigma(
    log_sigma_bar: float,
    half_log_msm: np.ndarray | float,
    x_slow: np.ndarray | float,
    var_slow: float,
) -> np.ndarray | float:
    """log sigma の合成式。**生成とアンサンブル検証の両方がこの 1 つを使う。**

    式を 2 か所に書くと、片方だけ直して乖離する事故が起きる。凸性補正 -Var(X) は
    OU 側のためのもの: E[e^{2X}] = e^{2Var(X)} != 1 なので、引かないと実効ボラが
    e^{Var(X)} 倍に膨らむ。MSM 側は E[prod M_i] = 1 なので補正不要。
    """
    return log_sigma_bar + half_log_msm + x_slow - var_slow


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

    # ------------------------------------------------------------------
    # S1: MSM 成分
    # ------------------------------------------------------------------
    def _simulate_msm(self, t_days: np.ndarray) -> np.ndarray:
        """0.5 * sum_i log M_i(t) をグリッド上で生成する (切替時刻ベース)。

        per-step の Bernoulli ループは k*N 回の乱数生成になり 11.7M ステップでは
        非現実的。成分ごとに (1) 切替回数 ~ Poisson(gamma_i * T)、(2) 切替時刻 ~
        Uniform(0, T) ソート、(3) 各区間の値を等確率で引く、の順で生成し、
        グリッドへは searchsorted で写像する。**この生成はグリッド解像度に一切
        依存しない** (t_days の値でしか使わない) ので、同一シードなら
        steps_per_day を変えても切替過程がビット単位で一致する。

        乱数消費は ``l2.vol_msm`` ストリームから成分 i=1..k の順に固定
        (Poisson 数 -> 時刻 -> 値)。この順序を変えると同一シードの経路が変わる。
        """
        cfg = self._config
        rng = self._rng.get("l2.vol_msm")
        k = cfg.msm_k
        m0 = solve_m0(k, cfg.vol_var_target_msm)
        log_hi = math.log(m0)
        log_lo = math.log(2.0 - m0)
        T = float(t_days[-1])

        total = np.zeros(t_days.shape[0], dtype=np.float64)
        switch_hash = hashlib.sha256()
        n_switches: list[int] = []
        occupancy_hi: list[float] = []

        for i in range(k):
            gamma_i = cfg.msm_gamma1_per_day * cfg.msm_b**i
            n_switch = int(rng.poisson(gamma_i * T))
            switch_times = np.sort(rng.uniform(0.0, T, n_switch))
            # 区間は n_switch + 1 個。先頭が初期値で、定常分布 (等確率) から引く。
            states = rng.integers(0, 2, n_switch + 1)

            idx = np.searchsorted(switch_times, t_days, side="right")
            grid_states = states[idx]
            total += np.where(grid_states == 1, log_hi, log_lo)

            switch_hash.update(np.int64(n_switch).tobytes())
            switch_hash.update(np.ascontiguousarray(switch_times).tobytes())
            switch_hash.update(np.ascontiguousarray(states).tobytes())
            n_switches.append(n_switch)
            occupancy_hi.append(float(grid_states.mean()))

        total *= 0.5
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
        x = np.empty(t_days.shape[0], dtype=np.float64)
        x[0] = x0
        x[1:] = y

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

        if not (cfg.enable_msm or cfg.enable_slow_ou):
            return np.full(n, log_sigma_bar, dtype=np.float64)

        t_days = t / self._seconds_per_day
        half_log_msm: np.ndarray | float = 0.0
        x_slow: np.ndarray | float = 0.0
        var_slow = 0.0
        if cfg.enable_msm:
            half_log_msm = self._simulate_msm(t_days)
        if cfg.enable_slow_ou:
            x_slow = self._simulate_slow_ou(t_days)
            var_slow = cfg.vol_var_target_slow

        log_vol = np.asarray(
            compose_log_sigma(log_sigma_bar, half_log_msm, x_slow, var_slow),
            dtype=np.float64,
        )

        # 診断用サブサンプル (分単位)。成分内訳を全ステップ保持すると本番設定で
        # 数 GB になるため間引く。検証スイートの path 診断がこれを使う。
        step_seconds = float(t[1] - t[0])
        stride = max(int(round(VOL_SUBSAMPLE_SECONDS / step_seconds)), 1)
        self.last_diagnostics["vol_subsample"] = {
            "stride": stride,
            "step_seconds": step_seconds,
            "t_days": t_days[::stride].copy(),
            "log_vol": log_vol[::stride].copy(),
            "half_log_msm": (
                half_log_msm[::stride].copy()
                if isinstance(half_log_msm, np.ndarray)
                else np.zeros(t_days[::stride].shape[0])
            ),
            "x_slow": (
                x_slow[::stride].copy()
                if isinstance(x_slow, np.ndarray)
                else np.zeros(t_days[::stride].shape[0])
            ),
        }
        self.last_diagnostics["composition"] = {
            "log_sigma_bar": log_sigma_bar,
            "convexity_correction": -var_slow,
            "enable_msm": cfg.enable_msm,
            "enable_slow_ou": cfg.enable_slow_ou,
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
    if config.enable_rough:
        raise NotImplementedError("ラフ・ボラティリティは S2 で simchart/layers/l2_price.py に実装します。")
    if config.enable_jump or config.enable_leverage:
        raise NotImplementedError("ジャンプ / レバレッジは S3 で simchart/layers/l2_price.py に実装します。")
    if config.enable_chaos_vol:
        raise NotImplementedError("カオス的ボラ成分 chi_2 は S5 で実装します。")
    return GBMPriceLayer(config, rng, calendar, activity)
