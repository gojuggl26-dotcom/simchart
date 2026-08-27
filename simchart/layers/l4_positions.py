"""L4: 建玉・レバレッジ分布・清算 (S0-perp ではスタブ §7)。

役割 (実装は S11-perp 以降)
---------------------------
open interest、レバレッジ分布 (Pareto α = leverage_pareto_alpha)、清算価格
密度 (:class:`~simchart.types_perp.PositionBook`) を追跡し、価格が清算帯に
触れたときに清算成行を L3 へ供給する — 清算カスケードは perp の裾の主要な
内生機構になる。

★実装順の制約: **S10-perp で基差 (perp − index) が定常になってから** L3⇄L4 の
循環を閉じる。基差が漂流したまま清算を配線すると、清算閾値との距離が
非定常になりカスケード統計が意味を失う (S10 の κ 結合を伝達率で固定してから
S11 のフィードバックを閉じたのと同じ段取り)。

S0-perp での状態
----------------
スタブ。config が ``enable_positions`` / ``enable_liquidation`` を
NotImplementedError で止めるため、このクラスは**構築すらされない**。
メソッド群はインターフェース宣言であり、呼べば NotImplementedError を送出する
(暗黙 no-op の構造的防止 — 本プロジェクトの禁止事項)。
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..types_perp import LiquidationEvent, PositionBook

__all__ = ["PositionLayer", "build_position_layer"]

_NOT_YET = (
    "L4 (建玉・清算) は S11-perp で実装されます (S10-perp で基差が定常に"
    "なってから L3⇄L4 の循環を閉じる — l4_positions.py の docstring)。"
)


class PositionLayer:
    """L4: open interest, leverage distribution, liquidation price density.

    S0-perp: stub。全メソッドが :class:`NotImplementedError` を送出する。
    """

    name = "l4.positions_stub"

    def __init__(self, config: Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    def liquidation_density(self) -> np.ndarray:
        raise NotImplementedError(_NOT_YET)

    def on_fill(self, side: int, size: float, price: float) -> None:
        raise NotImplementedError(_NOT_YET)

    def scan_liquidations(self, price: float) -> list[LiquidationEvent]:
        raise NotImplementedError(_NOT_YET)

    def apply_funding(self, funding_rate: float) -> None:
        raise NotImplementedError(_NOT_YET)

    def open_interest(self) -> float:
        raise NotImplementedError(_NOT_YET)

    def position_book(self) -> PositionBook:
        raise NotImplementedError(_NOT_YET)


def build_position_layer(config: Config) -> PositionLayer | None:
    """L4 を組み立てる。S0-perp では常に ``None`` (フラグは config が弾く)。"""
    if not (config.enable_positions or config.enable_liquidation):
        return None
    # ここには到達しない (config の未実装フラグ検証が先に止める) が、
    # 万一素通りしたときに暗黙 no-op にならないよう二重に守る。
    raise NotImplementedError(_NOT_YET)
