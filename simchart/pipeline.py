"""層の組み立てと実行。

駆動方式について
----------------
S0 は「L0 が張った時間グリッド上で L2 を一括生成し、L3 がそれをそのまま観測する」
という**グリッド駆動**である。しかし S6 で L3 がイベント駆動になると、主役は
L1 が生成するイベント時刻に移り、L2 は ``price.at(event_times)`` で問い合わされる
側になる。この転換をコメントではなく構造で表しておくために、駆動ロジックを
:class:`GridDriver` として切り出し、:func:`select_driver` で選ぶ形にしてある。
S6 では ``EventDriver`` を足して ``select_driver`` の分岐を 1 行増やすだけで済む。
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Config
from .layers import (
    build_activity,
    build_book_layer,
    build_calendar,
    build_price_layer,
)
from .rng import STREAM_NAMES, RNGRegistry
from .types import BookSnapshot, EventLog, Observation, PriceProcess, StageResult

__all__ = ["run", "run_twice", "determinism_check", "rng_stability_check", "GridDriver"]


@dataclass
class _Layers:
    calendar: Any
    activity: Any
    price: Any
    book: Any


class GridDriver:
    """L0 の時間グリッドで L2 を一括生成し、L3 に観測させる。

    S0〜S5 (板を導入するまで) の駆動方式。
    """

    name = "grid"

    def __call__(
        self, layers: _Layers
    ) -> tuple[PriceProcess, Observation, EventLog, BookSnapshot]:
        grid = layers.calendar.simulation_grid()
        price = layers.price.simulate(grid)
        observation, events, book = layers.book.observe(price, layers.calendar, layers.activity)
        return price, observation, events, book


def select_driver(config: Config) -> GridDriver:
    """設定に応じた駆動方式を選ぶ。

    S6 で板層を入れたら、ここに ``EventDriver`` (L1 のイベント時刻で L3 を回し、
    L2 へは ``price.at()`` で問い合わせる) を追加する。
    """
    if config.enable_book:
        raise NotImplementedError(
            "イベント駆動 (EventDriver) は S6 で simchart/pipeline.py に追加します。"
        )
    return GridDriver()


def _build_layers(config: Config, rng: RNGRegistry) -> _Layers:
    calendar = build_calendar(config, rng)
    activity = build_activity(config, rng, calendar)
    price = build_price_layer(config, rng, calendar, activity)
    book = build_book_layer(config, rng, calendar, activity)
    return _Layers(calendar=calendar, activity=activity, price=price, book=book)


def run(config: Config, *, rng: RNGRegistry | None = None) -> StageResult:
    """設定を 1 回実行して :class:`~simchart.types.StageResult` を返す。"""
    started = time.perf_counter()
    registry = rng if rng is not None else RNGRegistry(config.seed)

    layers = _build_layers(config, registry)
    driver = select_driver(config)
    price, observation, events, book = driver(layers)

    runtime = time.perf_counter() - started
    meta: dict[str, Any] = {
        "driver": driver.name,
        "layers": {
            "l0": layers.calendar.name,
            "l1": layers.activity.name,
            "l2": layers.price.name,
            "l3": layers.book.name,
        },
        "grid": {
            "n_points": price.n_points,
            "t_start_sec": price.t_start,
            "t_end_sec": price.t_end,
            "step_seconds": layers.calendar.step_seconds(),
            "session_seconds": layers.calendar.session_seconds(),
            "n_days": layers.calendar.n_days(),
        },
        "rng_streams_used": list(registry.used_streams()),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    return StageResult(
        stage=config.stage,
        config=config,
        price=price,
        events=events,
        book=book,
        observation=observation,
        runtime_sec=runtime,
        rng_fingerprint=registry.fingerprint(),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# ゲート用の検査
# ---------------------------------------------------------------------------
def run_twice(config: Config) -> tuple[StageResult, StageResult]:
    """同一設定で 2 回実行する (決定性ゲート用)。"""
    return run(config), run(config)


def determinism_check(config: Config, first: StageResult | None = None) -> dict[str, Any]:
    """同一シードでの 2 回実行がビット単位で一致するかを検査する。

    ダイジェストの一致だけでなく、主要配列の :func:`numpy.array_equal` も取る。
    ハッシュ一致は「同じバイト列」の十分条件としては強いが、どの配列が壊れたかを
    切り分けられないため、両方記録する。
    """
    a = first if first is not None else run(config)
    b = run(config)
    arrays = {
        "price.t": (a.price.t, b.price.t),
        "price.log_p_star": (a.price.log_p_star, b.price.log_p_star),
        "price.log_vol": (a.price.log_vol, b.price.log_vol),
        "price.jump_times": (a.price.jump_times, b.price.jump_times),
        "observation.log_price": (a.observation.log_price, b.observation.log_price),
    }
    per_array = {name: bool(np.array_equal(x, y)) for name, (x, y) in arrays.items()}
    digest_a, digest_b = a.digest(), b.digest()
    return {
        "bitwise_identical": bool(all(per_array.values()) and digest_a == digest_b),
        "digest_first": digest_a,
        "digest_second": digest_b,
        "digests_match": digest_a == digest_b,
        "per_array": per_array,
    }


def rng_stability_check(config: Config, n_draws: int | None = None) -> dict[str, Any]:
    """新しいストリームを足しても既存ストリームが不変であることを検査する。

    後段で新ストリームを追加したときに既存の系列が動くと、段階間の差分が
    「新機能の効果」なのか「乱数がずれただけ」なのか区別できなくなる。これは
    段階構築という方法そのものを無効にするので、critical ゲートとして扱う。

    併せて、宣言済みストリームどうしが偶然同じ系列になっていないか (名前ハッシュの
    衝突や実装ミスによる別名化) も確認する。
    """
    draws = n_draws if n_draws is not None else config.validation.rng_probe_draws

    baseline_registry = RNGRegistry(config.seed)
    baseline = {name: baseline_registry.get(name).standard_normal(draws) for name in STREAM_NAMES}

    # 新段階で足されるストリームを模して、先に別のストリームを大量に消費してから、
    # さらに逆順で既存ストリームを取得する。順序依存があればここで露見する。
    probe_names = ("s3.dummy_probe_a", "s3.dummy_probe_b")
    perturbed_registry = RNGRegistry(config.seed, extra_streams=probe_names)
    for probe in probe_names:
        perturbed_registry.get(probe).standard_normal(draws * 3)
    perturbed = {
        name: perturbed_registry.get(name).standard_normal(draws)
        for name in reversed(STREAM_NAMES)
    }

    per_stream = {
        name: bool(np.array_equal(baseline[name], perturbed[name])) for name in STREAM_NAMES
    }

    distinct = True
    names = list(STREAM_NAMES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.array_equal(baseline[names[i]], baseline[names[j]]):
                distinct = False
    return {
        "unchanged": bool(all(per_stream.values())),
        "streams_distinct": bool(distinct),
        "n_streams": len(STREAM_NAMES),
        "n_draws": draws,
        "probe_streams": list(probe_names),
        "per_stream": per_stream,
    }
