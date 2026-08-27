"""RNG ストリームの安定性。

要件は「新しいストリームを追加しても既存ストリームの系列が変わらない」。
``SeedSequence.spawn()`` は呼び出し順に子シードを配るのでこれを満たさない。
名前ハッシュ方式がその要件を満たしていることをここで固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, rng_stability_check
from simchart.rng import (
    KNOWN_STREAMS,
    STREAM_NAMES,
    RNGRegistry,
    UnknownStreamError,
    derive_seed,
)

SMALL = dict(n_days=3, steps_per_day=390)


def test_derive_seed_is_pure() -> None:
    assert derive_seed(42, "l2.diffusion") == derive_seed(42, "l2.diffusion")
    assert derive_seed(42, "l2.diffusion") != derive_seed(43, "l2.diffusion")
    assert derive_seed(42, "l2.diffusion") != derive_seed(42, "l3.order_size")


def test_new_stream_does_not_disturb_existing_streams() -> None:
    baseline = RNGRegistry(42)
    before = {name: baseline.get(name).standard_normal(64) for name in STREAM_NAMES}

    perturbed = RNGRegistry(42, extra_streams=("s3.brand_new_stream",))
    perturbed.get("s3.brand_new_stream").standard_normal(5_000)
    after = {name: perturbed.get(name).standard_normal(64) for name in STREAM_NAMES}

    for name in STREAM_NAMES:
        assert np.array_equal(before[name], after[name]), name


def test_stream_order_does_not_matter() -> None:
    forward = RNGRegistry(7)
    values_forward = {name: forward.get(name).standard_normal(32) for name in STREAM_NAMES}

    backward = RNGRegistry(7)
    values_backward = {
        name: backward.get(name).standard_normal(32) for name in reversed(STREAM_NAMES)
    }
    for name in STREAM_NAMES:
        assert np.array_equal(values_forward[name], values_backward[name]), name


def test_streams_are_mutually_distinct() -> None:
    registry = RNGRegistry(42)
    draws = {name: registry.get(name).standard_normal(128) for name in STREAM_NAMES}
    names = list(STREAM_NAMES)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            assert not np.array_equal(draws[first], draws[second]), (first, second)


def test_cached_generator_is_the_same_object() -> None:
    registry = RNGRegistry(42)
    assert registry.get("l2.diffusion") is registry.get("l2.diffusion")


def test_strict_mode_rejects_undeclared_streams() -> None:
    registry = RNGRegistry(42)
    with pytest.raises(UnknownStreamError) as excinfo:
        registry.get("l2.difusion")  # 打ち間違い
    assert "STREAM_NAMES" in str(excinfo.value)

    permissive = RNGRegistry(42, strict=False)
    assert permissive.get("anything.goes") is not None


def test_declared_streams_are_all_known() -> None:
    for name in STREAM_NAMES:
        assert name in KNOWN_STREAMS


def test_l2_and_l3_streams_are_separated() -> None:
    """L2 と L3 が同じストリームを共有していないこと。

    共有していると S10 で L3 を変えただけで L2 の価格経路が動き、結合前後の
    比較が成立しなくなる。
    """
    l2 = [n for n in STREAM_NAMES if n.startswith("l2.")]
    l3 = [n for n in STREAM_NAMES if n.startswith("l3.")]
    assert l2 and l3
    assert not set(l2) & set(l3)


def test_rng_stability_check_passes() -> None:
    report = rng_stability_check(Config(**SMALL))
    assert report["unchanged"] is True
    assert report["streams_distinct"] is True
    assert report["n_streams"] == len(STREAM_NAMES)
