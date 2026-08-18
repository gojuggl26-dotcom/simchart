"""L1: 潜在活動度層 (S0 ではスタブ)。

最終系での役割
--------------
``lambda(t) = phi_lambda(t) * mu * Z_t + 多変量 Hawkes``。取引の到来速度そのものを
生成し、L3 の注文流を駆動する。カオス成分 chi_1 (強度変調) と chi_3 (分岐比変調)
の注入点でもある。

S0 での実装
-----------
定数強度。イベント生成は行わない (S0 には板もイベントも無いため)。
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..rng import RNGRegistry
from .l0_calendar import ConstantCalendar

__all__ = ["ConstantActivity", "build_activity"]

#: S0 の名目強度 (イベント/秒)。値そのものは S0 では一切使われない。
#: S6 で L3 がイベント駆動になった時点で意味を持つ。
DEFAULT_INTENSITY: float = 1.0


class ConstantActivity:
    """自己励起のない定数強度。"""

    name = "l1.constant"

    def __init__(self, config: Config, calendar: ConstantCalendar) -> None:
        self._config = config
        self._calendar = calendar
        self._mu = DEFAULT_INTENSITY

    def intensity(self, t: float | np.ndarray) -> float | np.ndarray:
        """時刻 t での強度 ``phi(t) * mu``。S0 では phi = 1 なので定数。"""
        phi = self._calendar.phi(t)
        return self._mu * phi

    def branching_ratio(self) -> float | None:
        """Hawkes の分岐比。S0 には自己励起が無いので ``None``。

        ``0.0`` ではなく ``None`` を返すのは、「自己励起が無効」と「分岐比を
        推定したらゼロだった」を取り違えないため。検証側はこれを見て
        ``not_applicable`` を返す。
        """
        return None

    def event_times(self, t_start: float, t_end: float) -> np.ndarray:
        """区間内のイベント時刻。

        S0 の L3 はイベント駆動ではないため、この経路は使われない。呼ばれた場合は
        黙って空配列を返さず停止する。空を返すと「イベントが 0 件だった」という
        測定結果と区別がつかなくなるため。
        """
        raise NotImplementedError(
            f"L1 のイベント生成は S6 (板層の導入) で使い始め、S7 で Hawkes 化します。"
            f" 追加先: simchart/layers/l1_activity.py"
            f" (要求区間: [{t_start}, {t_end}])"
        )


def build_activity(
    config: Config, rng: RNGRegistry, calendar: ConstantCalendar
) -> ConstantActivity:
    if config.enable_hawkes:
        raise NotImplementedError("多変量 Hawkes は S7 で simchart/layers/l1_activity.py に実装します。")
    if config.enable_chaos_lambda or config.enable_chaos_branching:
        raise NotImplementedError("カオス成分 chi_1 / chi_3 は S12 で実装します。")
    del rng  # S0 の L1 は乱数を使わない (S7 で l1.hawkes を使う)
    return ConstantActivity(config, calendar)
