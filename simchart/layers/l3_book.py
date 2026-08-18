"""L3: 板層 (S0 ではスタブ)。

最終系での役割
--------------
メタオーダー分割 -> 6 次元 Hawkes 注文流 -> queue-reactive 板 -> uncertainty zones
による離散化。**観測価格は板のミッド**であり、p*(t) は外生生成された潜在情報価格に
すぎない。注文流の一部が p* 方向にバイアスを持つ (ハイブリッド方式) ことで、
情報が価格に染み込む速度そのものが内生的に決まる。結合強度が ``kappa`` (S10)。

S0 での実装
-----------
恒等写像。observed price = p*(t) をそのまま返し、イベントも板も空。
"""

from __future__ import annotations

from ..config import Config
from ..rng import RNGRegistry
from ..types import BookSnapshot, EventLog, Observation, PriceProcess
from .l0_calendar import ConstantCalendar
from .l1_activity import ConstantActivity

__all__ = ["PassThroughBook", "build_book_layer"]


class PassThroughBook:
    """板を持たず、潜在情報価格をそのまま観測価格として返す。"""

    name = "l3.passthrough"

    def __init__(self, config: Config, calendar: ConstantCalendar) -> None:
        self._config = config
        self._calendar = calendar

    def observe(
        self,
        price: PriceProcess,
        calendar: ConstantCalendar | None = None,
        activity: ConstantActivity | None = None,
    ) -> tuple[Observation, EventLog, BookSnapshot]:
        """観測系列・イベント列・板スナップショットを返す。

        S0 では観測時刻は L2 のグリッドと同一で、値は log p* そのもの。配列は
        コピーせず共有する (両方とも不変オブジェクトなので安全)。

        S6 以降、観測時刻は L2 のグリッドから切り離され、板イベントの時刻になる。
        そのとき L2 の値は ``price.at(event_times)`` で引く。**呼び出し側が
        「観測 = グリッド」を前提にしないよう、S0 の時点から
        :class:`~simchart.types.Observation` に時刻を明示的に持たせている。**
        """
        del activity
        cal = calendar or self._calendar
        observation = Observation(
            t=price.t,
            log_price=price.log_p_star,
            session_seconds=cal.session_seconds(),
            step_seconds=cal.step_seconds(),
            source="l3.passthrough(p_star)",
        )
        events = EventLog.empty(reason="S0 では注文流を生成しない (S6 で板層を導入)")
        book = BookSnapshot.empty(reason="S0 では板を持たない (S6 で板層を導入)")
        return observation, events, book


def build_book_layer(
    config: Config,
    rng: RNGRegistry,
    calendar: ConstantCalendar,
    activity: ConstantActivity,
) -> PassThroughBook:
    if config.enable_book:
        raise NotImplementedError("板層は S6 で simchart/layers/l3_book.py に実装します。")
    if config.enable_metaorder:
        raise NotImplementedError("メタオーダー分割は S8 で実装します。")
    if config.enable_queue_reactive or config.enable_uncertainty_zones:
        raise NotImplementedError("queue-reactive 板 / uncertainty zones は S9 で実装します。")
    if config.kappa != 0.0:
        raise NotImplementedError("p* との結合 (kappa) は S10 で実装します。")
    del rng, activity  # S0 の L3 は乱数も活動度も使わない
    return PassThroughBook(config, calendar)
