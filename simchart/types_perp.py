"""perp フォークの型宣言 (S0-perp §7.2)。

S0 の全段階分を先に宣言する方針の perp 版。ここに宣言することで、後段
(S10-perp / S11-perp) の実装が型を後付けして株式側の ``types.py`` を触らずに
済む。S0-perp では宣言のみ — 生成側は一切これらを作らない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PositionBook", "FundingState", "LiquidationEvent"]


@dataclass
class PositionBook:
    """建玉の density 表現: 清算価格ヒストグラム (long / short 別)。

    ``position_repr="density"`` (既定) の実体。個別エージェントを持たず、
    清算価格 p に何枚の建玉が積まれているかの密度だけを追跡する —
    清算カスケードの計算に必要なのはこの周辺分布であり、個別の建玉主体は
    S11-perp の検証 (agent 表現との相互検証) まで要らない。

    Attributes
    ----------
    price_edges:
        清算価格ビンの境界 (対数価格、単調増加)。
    long_density / short_density:
        各ビンの建玉量 (契約数)。long は価格下落で、short は上昇で清算される。
    """

    price_edges: np.ndarray
    long_density: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    short_density: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


@dataclass
class FundingState:
    """資金調達率の状態 (S10-perp)。

    Attributes
    ----------
    twap_basis:
        直近区間の TWAP 基差 (perp − index) / index。
    current_rate:
        確定済みの直近 funding レート (funding_cap でクリップ済み)。
    next_funding_time_sec:
        次回 funding 確定時刻 (シミュレーション秒)。
    history_times / history_rates:
        確定履歴 (validation/perp.funding_stats が読む)。
    """

    twap_basis: float = 0.0
    current_rate: float = 0.0
    next_funding_time_sec: float = 0.0
    history_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    history_rates: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


@dataclass(frozen=True)
class LiquidationEvent:
    """清算イベント 1 件 (S11-perp)。

    Attributes
    ----------
    t_sec:
        発生時刻 (シミュレーション秒)。
    side:
        +1 = long の清算 (売り成行が板に出る)、-1 = short の清算 (買い成行)。
    size:
        清算数量 (契約)。partial_liquidation_frac の分割後の 1 回分。
    trigger_price:
        トリガとなった対数価格。
    """

    t_sec: float
    side: int
    size: float
    trigger_price: float
