"""L2: 情報価格層。S0 で唯一実体を持つ層。

S0 の中身
---------
定数ボラ・正規革新の幾何ブラウン運動だけ。**これが非現実的なのが正しい。**
尖度 3、|r| の自己相関ゼロ、単一フラクタル、分散比 1 が S0 の正解であり、
「それらしく」見せるためのチューニング (t 分布革新、MA(1) の付加など) をしては
ならない。ファットテールは S1〜S3 でボラ過程とジャンプから内生的に、短期の負の
自己相関は S9 の uncertainty zones から出す。外生的に入れると、時間集計で尖度が
下がるという実データの性質 (集計正規性) が永久に再現できなくなる。

拡張の入り口
------------
- :meth:`GBMPriceLayer._log_vol_path` … S1 (MSM / 緩慢 OU), S2 (ラフ), S5 (chi_2)
- :meth:`GBMPriceLayer._jump_component` … S3 (Hawkes ジャンプ)
- :meth:`GBMPriceLayer._leverage_innovation` … S3 (レバレッジ効果)
"""

from __future__ import annotations

import math

import numpy as np

from ..config import TRADING_DAYS_PER_YEAR, Config
from ..rng import RNGRegistry
from ..types import PriceProcess
from .l0_calendar import ConstantCalendar
from .l1_activity import ConstantActivity

__all__ = ["GBMPriceLayer", "build_price_layer"]


class GBMPriceLayer:
    """定数ボラの幾何ブラウン運動で log p*(t) を生成する。"""

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

    # ------------------------------------------------------------------
    # 拡張フック
    # ------------------------------------------------------------------
    def _log_vol_path(self, t: np.ndarray) -> np.ndarray:
        """log (瞬間ボラ) の経路。S0 では定数 ``log(sigma_bar)``。

        後段でここに足していく (すべて対数ボラの**加法**成分として設計する):

        - S1: MSM のカスケード ``+ log(M_1 M_2 ... M_k)`` と緩慢 OU ``+ X_t``
        - S2: ラフ成分 ``+ nu * W^H_t`` (H ~ 0.1 の分数ブラウン運動)
        - S5: カオス成分 chi_2 ``+ c * g(chi_2(t))``

        加法で設計する理由は、各成分の寄与を対数ボラの分散分解で切り分けられる
        ようにするため。乗法で混ぜると S1 と S2 の効果が分離できなくなる。
        """
        return np.full(t.shape[0], math.log(self._config.sigma_bar), dtype=np.float64)

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

        価格の非定常性 (単位根) とリターンの定常性はこの構成から自動的に出る。
        わざわざ足したり調整したりしない。

        ボラは区間の**左端**の値を使う (Euler-Maruyama)。右端や区間平均を使うと
        S3 でレバレッジを入れたときに未来のボラ情報が当該区間のリターンへ漏れる
        (ルックアヘッド)。S0 では定数なので値は変わらないが、規約はここで固定する。
        """
        if t.ndim != 1 or t.shape[0] < 2:
            raise ValueError("時刻グリッドは 1 次元で 2 点以上必要です")

        n = int(t.shape[0])
        log_vol = self._log_vol_path(t)
        sigma_left = np.exp(log_vol[:-1])

        dt_sec = np.diff(t)
        if dt_sec.min() <= 0:
            raise ValueError("時刻グリッドが単調増加ではありません")
        uniform = dt_sec.min() == dt_sec.max()

        z = self._rng.get("l2.diffusion").standard_normal(n - 1)
        mu = float(self._config.mu_drift)

        if uniform and np.all(sigma_left == sigma_left[0]):
            # S0 の経路: 全部スカラーで済むので中間配列を作らない。
            dt_y = float(dt_sec[0]) / self._seconds_per_year
            sigma = float(sigma_left[0])
            increments = z  # 以降は in-place で書き換える
            increments *= sigma * math.sqrt(dt_y)
            increments += (mu - 0.5 * sigma * sigma) * dt_y
        else:
            del dt_sec
            dt_y = np.diff(t) / self._seconds_per_year
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
    if config.enable_msm or config.enable_slow_ou:
        raise NotImplementedError("MSM / 緩慢 OU は S1 で simchart/layers/l2_price.py に実装します。")
    if config.enable_rough:
        raise NotImplementedError("ラフ・ボラティリティは S2 で simchart/layers/l2_price.py に実装します。")
    if config.enable_jump or config.enable_leverage:
        raise NotImplementedError("ジャンプ / レバレッジは S3 で simchart/layers/l2_price.py に実装します。")
    if config.enable_chaos_vol:
        raise NotImplementedError("カオス的ボラ成分 chi_2 は S5 で実装します。")
    return GBMPriceLayer(config, rng, calendar, activity)
