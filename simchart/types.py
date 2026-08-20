"""層のあいだを流れるデータ型と、層のインターフェース (Protocol)。

ここで決めた型が S1〜S13 全部を規定する。特に :class:`PriceProcess` は
**配列ではなく補間可能なオブジェクト**として定義してある。理由は S10 で L3 が
不規則なイベント時刻に p* を参照するようになるからで、ここを配列直返しにすると
その時点で L2 を書き直すことになる (指示書 §6)。
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import Config

__all__ = [
    "PriceProcess",
    "EventType",
    "EventLog",
    "BookSnapshot",
    "Observation",
    "BarSeries",
    "StageResult",
    "CalendarLayer",
    "ActivityLayer",
    "PriceLayer",
    "BookLayer",
]


# ---------------------------------------------------------------------------
# L2 の出力
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PriceProcess:
    """潜在情報価格 p*(t) の経路。

    Attributes
    ----------
    t:
        時刻グリッド (秒、セッション時間の通し)。狭義単調増加。
    log_p_star:
        各グリッド点での log p*。
    log_vol:
        各グリッド点での log (瞬間ボラ)。S0 では定数。
    jump_times:
        ジャンプが発生した時刻 (S0 では空)。
    interpolation:
        グリッド間の補間方式。S0 は ``"linear"`` のみ。

    補間方式についての設計判断
    --------------------------
    既定は対数価格の線形補間である。理由は決定論的で冪等 (同じ時刻を何度問い合わせ
    ても同じ値) だから。L3 のイベント時刻は S6 以降で注文流に依存して変わるので、
    問い合わせのたびに乱数を引く方式 (ブラウン橋) にすると **問い合わせ順序が価格
    経路を変えてしまい**、S10 で「L3 を変えても L2 は不変」という保証が崩れる。

    その代償として、グリッド間隔より短い時間スケールでは分散が過小になる (線形補間
    は橋の条件付き期待値であり、条件付き分散を捨てている)。S0 のグリッドは 1 秒で
    あり、板イベントがそれより高頻度になる S6 以降で問題になりうる。対処は
    「L2 のグリッドを細かくする」が第一で、どうしてもブラウン橋が要るなら
    ``interpolation="brownian_bridge"`` を追加し、専用ストリーム ``l2.bridge``
    (rng.py に予約済み) を使い、**問い合わせ時刻を事前に確定させてから一括で**
    橋を張ること。逐次問い合わせで乱数を引く実装にしてはならない。

    ジャンプについての注意 (S3)
    ---------------------------
    線形補間はジャンプをなまして (smear) しまう。S3 でジャンプを入れるときは
    ジャンプ時刻を必ずグリッド点に載せるか、``interpolation="jump_aware"`` を
    実装してジャンプ区間だけ左連続にすること。
    """

    t: np.ndarray
    log_p_star: np.ndarray
    log_vol: np.ndarray
    jump_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    #: S4: オーバーナイト・ギャップ (日境界ごとに 1 つ、長さ n_days-1)。
    #: ★``log_p_star`` は**日中のみ**の連続経路で、ギャップはそこに含まれない。
    #: グリッドを S3 と同一に保つことで ``compare S3 S4`` が直接成立し、開値〜引値
    #: の統計 (Hill α / JV share) が S3 と同じ定義で測れる。クローズ・トゥ・
    #: クローズ系列は検証側でギャップと合成して作る (S4 指示書 §10 の分離)。
    overnight_gaps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    interpolation: str = "linear"

    _SUPPORTED_INTERPOLATION = ("linear",)
    _PLANNED_INTERPOLATION = {
        "jump_aware": "S3 (ジャンプ区間を左連続にする)",
        "brownian_bridge": "S10 (グリッド間の分散を保つ)",
    }

    def __post_init__(self) -> None:
        n = self.t.shape[0]
        if self.t.ndim != 1:
            raise ValueError("t は 1 次元配列である必要があります")
        for name in ("log_p_star", "log_vol"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} の形状 {arr.shape} が t の形状 ({n},) と一致しません")
        if n < 2:
            raise ValueError("グリッド点が 2 点未満です")
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("t は狭義単調増加である必要があります")
        if self.interpolation not in self._SUPPORTED_INTERPOLATION:
            planned = self._PLANNED_INTERPOLATION.get(self.interpolation)
            if planned:
                raise NotImplementedError(
                    f"interpolation={self.interpolation!r} は {planned} で実装される予定です。"
                )
            raise ValueError(f"未知の補間方式 {self.interpolation!r} です")

    # ------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        return int(self.t.shape[0])

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def _interp(self, q: np.ndarray, values: np.ndarray) -> np.ndarray:
        """線形補間。グリッド点上では**丸め誤差なしに**格納値そのものを返す。

        ``np.interp`` は格子点でも ``slope * 0 + y_i`` を計算するため通常は厳密だが、
        依存しているのが実装詳細なので、格子点一致を明示的に上書きして保証する。
        """
        out = np.interp(q, self.t, values)
        idx = np.searchsorted(self.t, q, side="left")
        in_range = idx < self.t.shape[0]
        exact = np.zeros(q.shape, dtype=bool)
        if np.any(in_range):
            exact[in_range] = self.t[idx[in_range]] == q[in_range]
        if np.any(exact):
            out[exact] = values[idx[exact]]
        return out

    def _query(self, t_query: float | np.ndarray, values: np.ndarray) -> float | np.ndarray:
        q = np.asarray(t_query, dtype=np.float64)
        scalar = q.ndim == 0
        q1 = np.atleast_1d(q)
        if np.any(q1 < self.t[0]) or np.any(q1 > self.t[-1]):
            raise ValueError(
                f"問い合わせ時刻がグリッドの範囲 [{self.t[0]}, {self.t[-1]}] の外です。"
                f" 外挿は許可していません (静かに端の値を返すと board 側の不具合が隠れるため)。"
            )
        out = self._interp(q1, values)
        return float(out[0]) if scalar else out

    def at(self, t_query: float | np.ndarray) -> float | np.ndarray:
        """任意時刻での log p* を補間して返す。"""
        return self._query(t_query, self.log_p_star)

    def vol_at(self, t_query: float | np.ndarray) -> float | np.ndarray:
        """任意時刻での log (瞬間ボラ) を補間して返す。"""
        return self._query(t_query, self.log_vol)

    def digest(self) -> str:
        """経路の SHA-256。決定性ゲートの照合に使う。"""
        h = hashlib.sha256()
        # overnight_gaps は S0〜S3 では空配列なので tobytes() が b"" となり、
        # ハッシュに寄与しない — 既存段階のダイジェストは変わらない。
        for arr in (self.t, self.log_p_star, self.log_vol, self.jump_times,
                    self.overnight_gaps):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# L3 の出力
# ---------------------------------------------------------------------------
class EventType(IntEnum):
    """注文流イベントの種別。

    S7 の 6 次元 Hawkes は (LIMIT_ADD, MARKET, CANCEL) x (買い, 売り) に対応する。
    向きは :attr:`EventLog.side` が持つ。
    """

    LIMIT_ADD = 0
    CANCEL = 1
    MARKET = 2
    MODIFY = 3
    TRADE = 4
    TRIGGER = 5


@dataclass(frozen=True)
class EventLog:
    """注文流イベント列。S0 では空。

    ``side`` は +1 が買い、-1 が売り、0 が向きなし。``agent_id`` は S8 の
    メタオーダー帰属に使い、-1 は帰属なしを表す。
    """

    t: np.ndarray
    event_type: np.ndarray
    side: np.ndarray
    price: np.ndarray
    size: np.ndarray
    order_id: np.ndarray
    agent_id: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    _ARRAY_FIELDS = ("t", "event_type", "side", "price", "size", "order_id", "agent_id")

    def __post_init__(self) -> None:
        n = self.t.shape[0]
        for name in self._ARRAY_FIELDS:
            arr = getattr(self, name)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(f"EventLog.{name} の形状が t と一致しません")

    @classmethod
    def empty(cls, **meta: Any) -> "EventLog":
        return cls(
            t=np.empty(0, dtype=np.float64),
            event_type=np.empty(0, dtype=np.int8),
            side=np.empty(0, dtype=np.int8),
            price=np.empty(0, dtype=np.float64),
            size=np.empty(0, dtype=np.float64),
            order_id=np.empty(0, dtype=np.int64),
            agent_id=np.empty(0, dtype=np.int64),
            meta=dict(meta),
        )

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def select(self, mask: np.ndarray) -> "EventLog":
        return EventLog(
            **{name: getattr(self, name)[mask] for name in self._ARRAY_FIELDS},
            meta=dict(self.meta),
        )

    def trades(self) -> "EventLog":
        """約定イベントのみを抜き出す (符号 ACF / propagator の入力)。"""
        if self.is_empty:
            return self
        return self.select(self.event_type == int(EventType.TRADE))


@dataclass(frozen=True)
class BookSnapshot:
    """一定間隔の板スナップショット。S0 では空。

    価格・数量は ``(n_snapshots, n_levels)``。レベル 0 が最良気配。
    """

    t: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    _MATRIX_FIELDS = ("bid_px", "bid_sz", "ask_px", "ask_sz")

    def __post_init__(self) -> None:
        n = self.t.shape[0]
        for name in self._MATRIX_FIELDS:
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(f"BookSnapshot.{name} の形状 {arr.shape} が不正です")

    @classmethod
    def empty(cls, n_levels: int = 0, **meta: Any) -> "BookSnapshot":
        return cls(
            t=np.empty(0, dtype=np.float64),
            bid_px=np.empty((0, n_levels), dtype=np.float64),
            bid_sz=np.empty((0, n_levels), dtype=np.float64),
            ask_px=np.empty((0, n_levels), dtype=np.float64),
            ask_sz=np.empty((0, n_levels), dtype=np.float64),
            meta=dict(meta),
        )

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0 or self.bid_px.shape[1] == 0

    @property
    def n_levels(self) -> int:
        return int(self.bid_px.shape[1])

    def mid(self) -> np.ndarray:
        return 0.5 * (self.bid_px[:, 0] + self.ask_px[:, 0])

    def spread(self) -> np.ndarray:
        return self.ask_px[:, 0] - self.bid_px[:, 0]

    def microprice(self) -> np.ndarray:
        b, a = self.bid_sz[:, 0], self.ask_sz[:, 0]
        total = b + a
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, (self.ask_px[:, 0] * b + self.bid_px[:, 0] * a) / total, self.mid())


@dataclass(frozen=True)
class Observation:
    """観測される価格系列。S6 以降は板のミッド、S0 では p* そのもの。

    Attributes
    ----------
    t:
        観測時刻 (秒)。狭義単調増加。
    log_price:
        観測対数価格。
    session_seconds:
        1 セッションの長さ (秒)。日をまたぐリターンを除外するために使う。
    step_seconds:
        等間隔格子ならその刻み。不規則観測なら ``None``。
    source:
        観測の出どころ (診断用)。
    """

    t: np.ndarray
    log_price: np.ndarray
    session_seconds: float
    step_seconds: float | None
    source: str

    def __post_init__(self) -> None:
        if self.t.shape != self.log_price.shape:
            raise ValueError("Observation の t と log_price の形状が一致しません")
        if self.t.ndim != 1 or self.t.shape[0] < 2:
            raise ValueError("Observation は 1 次元で 2 点以上必要です")
        if self.session_seconds <= 0:
            raise ValueError("session_seconds は正である必要があります")

    @property
    def n_points(self) -> int:
        return int(self.t.shape[0])

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.t, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.log_price, dtype=np.float64).tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------
    def to_bars(self, bar_seconds: float) -> "BarSeries":
        """指定粒度のバー系列へ再標本化する。

        各セッション内で ``[開始, 開始+bar, 開始+2*bar, ...]`` の時刻における
        「その時刻以前の最後の観測値」を取る。セッション長が ``bar_seconds`` で
        割り切れない場合、末尾の端数は捨てる。

        **セッション境界をまたぐ差分はリターンに含めない。** S0 ではオーバーナイト
        が無いので実質的な違いはないが、S4 でギャップを入れた瞬間にこの区別が
        効いてくるため、最初からこの構造で測る。
        """
        if bar_seconds <= 0:
            raise ValueError("bar_seconds は正である必要があります")
        n_bars = int(self.session_seconds // bar_seconds)
        if n_bars < 1:
            raise ValueError(
                f"bar_seconds={bar_seconds} がセッション長 {self.session_seconds} より長いです"
            )
        n_days = int(round((self.t[-1] - self.t[0]) / self.session_seconds))
        if n_days < 1:
            raise ValueError("観測が 1 セッションに満たないため再標本化できません")

        offsets = np.arange(n_bars + 1, dtype=np.float64) * bar_seconds
        day_starts = self.t[0] + np.arange(n_days, dtype=np.float64) * self.session_seconds
        query = day_starts[:, None] + offsets[None, :]

        idx = self._index_at_or_before(query.ravel())
        log_price = self.log_price[idx].reshape(query.shape)
        return BarSeries(
            bar_seconds=float(bar_seconds),
            session_seconds=float(self.session_seconds),
            t0=float(self.t[0]),
            offsets=offsets,
            log_price=log_price,
        )

    def _index_at_or_before(self, query: np.ndarray) -> np.ndarray:
        """``query`` 以下で最大の観測インデックス。等間隔格子なら整数演算で求める。"""
        if self.step_seconds is not None:
            pos = (query - self.t[0]) / self.step_seconds
            idx = np.floor(pos + 1e-9).astype(np.int64)
        else:
            idx = np.searchsorted(self.t, query, side="right") - 1
        np.clip(idx, 0, self.t.shape[0] - 1, out=idx)
        return idx


@dataclass(frozen=True)
class BarSeries:
    """セッション x バーの 2 次元に整形した対数価格系列。

    ``log_price`` は ``(n_days, n_bars + 1)``。列 0 は各セッションの始値時点、
    列 k (k>=1) は k 本目のバーの終値時点。したがって 1 セッションあたりの
    リターンはちょうど ``n_bars`` 本で、セッションをまたぐ差分は構造的に発生しない。
    """

    bar_seconds: float
    session_seconds: float
    t0: float
    offsets: np.ndarray
    log_price: np.ndarray

    def __post_init__(self) -> None:
        if self.log_price.ndim != 2:
            raise ValueError("BarSeries.log_price は 2 次元である必要があります")
        if self.log_price.shape[1] != self.offsets.shape[0]:
            raise ValueError("BarSeries.log_price の列数が offsets と一致しません")

    @property
    def n_days(self) -> int:
        return int(self.log_price.shape[0])

    @property
    def n_bars_per_day(self) -> int:
        return int(self.log_price.shape[1] - 1)

    @property
    def n_returns(self) -> int:
        return self.n_days * self.n_bars_per_day

    @property
    def t(self) -> np.ndarray:
        """各点の時刻 ``(n_days, n_bars + 1)``。"""
        day_starts = self.t0 + np.arange(self.n_days, dtype=np.float64) * self.session_seconds
        return day_starts[:, None] + self.offsets[None, :]

    def returns_2d(self) -> np.ndarray:
        """セッション内リターン ``(n_days, n_bars)``。"""
        return np.diff(self.log_price, axis=1)

    def returns(self) -> np.ndarray:
        """全セッションを連結したリターン (1 次元)。"""
        return self.returns_2d().ravel()

    def segments(self) -> list[np.ndarray]:
        """セッションごとの対数価格 (行のリスト)。日内で完結する推定に使う。"""
        return [row for row in self.log_price]

    def log_price_flat(self) -> np.ndarray:
        """連結した対数価格。セッション境界の重複点 (各日の列 0) は落とす。"""
        first = self.log_price[0, :1]
        rest = self.log_price[:, 1:].ravel()
        return np.concatenate([first, rest])


# ---------------------------------------------------------------------------
# パイプラインの出力
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageResult:
    """1 回の実行の全出力。検証スイートとレポートはこれだけを入力とする。"""

    stage: str
    config: Config
    price: PriceProcess
    events: EventLog
    book: BookSnapshot
    observation: Observation
    runtime_sec: float
    rng_fingerprint: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        """実行結果全体の SHA-256。決定性ゲートで 2 回の実行を照合する。"""
        h = hashlib.sha256()
        h.update(self.price.digest().encode())
        h.update(self.observation.digest().encode())
        for name in EventLog._ARRAY_FIELDS:
            h.update(np.ascontiguousarray(getattr(self.events, name)).tobytes())
        for name in ("t",) + BookSnapshot._MATRIX_FIELDS:
            h.update(np.ascontiguousarray(getattr(self.book, name)).tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# 層のインターフェース
# ---------------------------------------------------------------------------
@runtime_checkable
class CalendarLayer(Protocol):
    """L0: 時間の構造。日内 U 字・寄引・オーバーナイトを司る。"""

    name: str

    def session_seconds(self) -> float:
        """1 セッションの長さ (秒)。"""

    def simulation_grid(self) -> np.ndarray:
        """L2 が価格を生成する時刻グリッド (秒、通し)。"""

    def phi(self, t: float | np.ndarray) -> float | np.ndarray:
        """活動度の季節係数 phi(t)。S0 では恒等的に 1。"""


@runtime_checkable
class ActivityLayer(Protocol):
    """L1: 潜在活動度 lambda(t)。"""

    name: str

    def intensity(self, t: float | np.ndarray) -> float | np.ndarray:
        """時刻 t での強度。"""

    def branching_ratio(self) -> float | None:
        """Hawkes の分岐比。自己励起が無い段階では ``None``。"""

    def event_times(self, t_start: float, t_end: float) -> np.ndarray:
        """区間内のイベント時刻。S6 以降で L3 の駆動に使う。"""


@runtime_checkable
class PriceLayer(Protocol):
    """L2: 潜在情報価格 p*(t) とボラ過程。"""

    name: str

    def simulate(self, t: np.ndarray) -> PriceProcess:
        """時刻グリッド上で経路を生成する。"""


@runtime_checkable
class BookLayer(Protocol):
    """L3: 板と観測価格。"""

    name: str

    def observe(
        self,
        price: PriceProcess,
        calendar: CalendarLayer,
        activity: ActivityLayer,
    ) -> tuple[Observation, EventLog, BookSnapshot]:
        """観測価格・イベント列・板スナップショットを返す。"""


def as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """入力を float64 の 1 次元 C 連続配列に揃える。"""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """dataclass を JSON 化しやすい辞書へ (配列は形状のみ)。"""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, np.ndarray):
            out[f.name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif dataclasses.is_dataclass(value):
            out[f.name] = dataclass_to_dict(value)
        else:
            out[f.name] = value
    return out
