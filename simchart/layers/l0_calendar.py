"""L0: カレンダー層 (S0 ではスタブ)。

最終系での役割
--------------
日内 U 字の活動度季節性 phi(t)、寄付・引けの特異点、オーバーナイト・ギャップ。
これらは「見かけの統計性質」を大量に作る (日内ボラの U 字は、それを除去しないと
長期記憶や多重フラクタルに見えてしまう)。したがって L0 は最下層に置き、
上の層はすべて phi(t) で伸縮された時間の上で動く。

S0 での実装
-----------
phi(t) は恒等的に 1、セッションは等長・等間隔、オーバーナイトなし。
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..rng import RNGRegistry

__all__ = ["ConstantCalendar", "build_calendar", "SESSION_SECONDS"]

#: 1 立会日の長さ (秒)。6.5 時間 = 23400 秒。
#: ``Config.steps_per_day`` はこの長さを何分割するかを表す。
SESSION_SECONDS: float = 6.5 * 3600.0


class ConstantCalendar:
    """時間構造を持たないカレンダー。

    セッションを等間隔に分割した通し時刻グリッドを供給するだけで、季節性も
    ギャップも入れない。
    """

    name = "l0.constant"

    def __init__(self, config: Config) -> None:
        self._config = config
        self._session_seconds = SESSION_SECONDS
        self._step_seconds = SESSION_SECONDS / config.steps_per_day

    # ------------------------------------------------------------------
    def session_seconds(self) -> float:
        return self._session_seconds

    def step_seconds(self) -> float:
        """シミュレーション格子の刻み (秒)。"""
        return self._step_seconds

    def n_days(self) -> int:
        return self._config.n_days

    def simulation_grid(self) -> np.ndarray:
        """L2 が価格を生成する時刻グリッド。

        ``t_i = i * step_seconds`` で ``i = 0 .. n_days * steps_per_day``。
        セッション境界の時刻は前日の最終点と翌日の始点が同一点として共有される
        (S0 にはオーバーナイトが無いため)。S4 でギャップを入れるときは、ここで
        境界に 2 点を置き、その間に不連続を作ることになる。
        """
        n_points = self._config.total_steps + 1
        return np.arange(n_points, dtype=np.float64) * self._step_seconds

    def phi(self, t: float | np.ndarray) -> float | np.ndarray:
        """活動度の季節係数。S0 では恒等的に 1。"""
        if np.isscalar(t):
            return 1.0
        return np.ones_like(np.asarray(t, dtype=np.float64))

    def day_index(self, t: float | np.ndarray) -> np.ndarray:
        """時刻が属するセッション番号。"""
        arr = np.asarray(t, dtype=np.float64)
        idx = np.floor(arr / self._session_seconds + 1e-9).astype(np.int64)
        return np.clip(idx, 0, self._config.n_days - 1)

    def overnight_gaps(self) -> np.ndarray:
        """各セッション境界のギャップ (対数)。S0 では常にゼロ。"""
        return np.zeros(max(self._config.n_days - 1, 0), dtype=np.float64)


def build_calendar(config: Config, rng: RNGRegistry) -> ConstantCalendar:
    """設定に応じた L0 を組み立てる。

    ``enable_seasonality`` / ``enable_overnight`` は S4 で実装する。Config 側で
    既に弾いているが、層の側でも二重に止めておく (Config を経由しない直接構築で
    静かにスタブが使われるのを防ぐため)。
    """
    if config.enable_seasonality or config.enable_overnight:
        raise NotImplementedError(
            "日内季節性 phi(t) とオーバーナイト・ギャップは S4 で "
            "simchart/layers/l0_calendar.py に実装します。"
        )
    del rng  # S0 の L0 は乱数を使わない (S4 で l0.calendar / l0.overnight を使う)
    return ConstantCalendar(config)
