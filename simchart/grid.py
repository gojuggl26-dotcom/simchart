"""時間軸の単一情報源 (S0-perp §4)。

24/7 化 (perp_clob) で最も事故が起きやすいのは時間換算である: 年率換算・
半減期・強度 [1/日] が「1 日 = 6.5 時間 / 年 = 252 日」を暗黙に仮定した
箇所に散らばっていると、**すべてが静かにずれる**。

このモジュールが市場タイプ別の時間定義の**唯一の定義点**になる:

| | equity | perp_clob |
|---|---|---|
| ann_days (年率換算の日数) | 252 | 365 |
| seconds_per_day (1 日の取引秒数) | 23,400 (6.5h) | 86,400 (24h) |

運用規約:

- 生成系 (layers/) は :class:`~simchart.layers.l0_calendar.ConstantCalendar`
  経由で ``session_seconds()`` を、年率換算は ``config.ann_days`` を取る。
  **TRADING_DAYS_PER_YEAR / SESSION_SECONDS を config.py の外から直接参照
  しない** (定数は equity の定義値として config.py に残るが、参照点は
  config のプロパティと本モジュールに限る)。
- 検証系は観測の ``session_seconds`` (Observation が運ぶ) を使う。
- 遵守はコード検査テスト (tests_perp_fork の time_grid_single_source) が
  監視する。

★equity のビット単位不変性: equity では ``config.ann_days == 252``・
``config.seconds_per_day == 23400.0`` で、置換前の定数と**同一の float** を
返すため、この集約は経路を 1 bit も変えない (fixture ゲートが検証する)。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TimeGrid"]


@dataclass(frozen=True)
class TimeGrid:
    """時間換算の唯一の権威。

    すべての換算をメソッド経由にする (§4.1)。値はプロパティで導出し、
    フィールドの二重管理をしない。
    """

    n_days: int
    steps_per_day: int
    ann_days: int
    seconds_per_day: float

    def __post_init__(self) -> None:
        if self.n_days <= 0 or self.steps_per_day <= 0:
            raise ValueError("n_days / steps_per_day は正整数である必要があります")
        if self.ann_days <= 0 or self.seconds_per_day <= 0:
            raise ValueError("ann_days / seconds_per_day は正である必要があります")

    @classmethod
    def from_config(cls, config) -> "TimeGrid":
        return cls(
            n_days=config.n_days,
            steps_per_day=config.steps_per_day,
            ann_days=config.ann_days,
            seconds_per_day=config.seconds_per_day,
        )

    # ------------------------------------------------------------------
    @property
    def dt_days(self) -> float:
        """1 ステップの長さ [日]。"""
        return 1.0 / self.steps_per_day

    @property
    def dt_seconds(self) -> float:
        """1 ステップの長さ [秒]。"""
        return self.seconds_per_day / self.steps_per_day

    @property
    def dt_years(self) -> float:
        """1 ステップの長さ [年]。年率ボラを 1 ステップに落とすときに使う。"""
        return 1.0 / (self.ann_days * self.steps_per_day)

    @property
    def steps_per_year(self) -> float:
        return float(self.ann_days * self.steps_per_day)

    @property
    def seconds_per_year(self) -> float:
        return self.ann_days * self.seconds_per_day

    @property
    def total_steps(self) -> int:
        return self.n_days * self.steps_per_day

    # ------------------------------------------------------------------
    def days_to_steps(self, d: float) -> int:
        """日数 → ステップ数 (四捨五入)。"""
        return int(round(d * self.steps_per_day))

    def seconds_to_steps(self, s: float) -> int:
        return int(round(s / self.dt_seconds))

    def per_day_to_per_step(self, rate: float) -> float:
        """強度 [1/日] → [1/ステップ]。"""
        return rate / self.steps_per_day

    def per_year_to_per_step(self, rate: float) -> float:
        """強度 [1/年] → [1/ステップ]。"""
        return rate / self.steps_per_year

    def annualize_daily_sd(self, sd_daily: float) -> float:
        """日次リターン SD → 年率ボラ。"""
        return sd_daily * self.ann_days**0.5
