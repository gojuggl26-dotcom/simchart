"""同一シードでの再実行がビット単位で一致することの確認。

これが崩れると段階間の比較が一切できなくなるので、最も基本的な保証として扱う。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, determinism_check, run

SMALL = dict(n_days=5, steps_per_day=390)


def test_two_runs_are_bitwise_identical() -> None:
    config = Config(**SMALL)
    first = run(config)
    second = run(config)

    assert np.array_equal(first.price.t, second.price.t)
    assert np.array_equal(first.price.log_p_star, second.price.log_p_star)
    assert np.array_equal(first.price.log_vol, second.price.log_vol)
    assert np.array_equal(first.observation.log_price, second.observation.log_price)
    # array_equal は NaN を等しくないとみなすので、バイト列でも照合する。
    assert first.price.log_p_star.tobytes() == second.price.log_p_star.tobytes()
    assert first.digest() == second.digest()


def test_determinism_check_reports_success() -> None:
    report = determinism_check(Config(**SMALL))
    assert report["bitwise_identical"] is True
    assert report["digests_match"] is True
    assert all(report["per_array"].values())


def test_different_seed_gives_different_path() -> None:
    a = run(Config(seed=42, **SMALL))
    b = run(Config(seed=43, **SMALL))
    assert a.digest() != b.digest()
    assert not np.array_equal(a.price.log_p_star, b.price.log_p_star)


def test_result_is_reproducible_after_unrelated_rng_use() -> None:
    """関係のない乱数を先に消費しても結果が変わらないこと。

    グローバルな乱数状態に依存していれば、ここで落ちる。
    """
    baseline = run(Config(**SMALL))
    np.random.default_rng(999).standard_normal(10_000)
    np.random.seed(7)
    np.random.random(10_000)
    assert run(Config(**SMALL)).digest() == baseline.digest()


@pytest.mark.parametrize("n_days,steps_per_day", [(2, 234), (3, 390), (4, 780)])
def test_determinism_across_shapes(n_days: int, steps_per_day: int) -> None:
    config = Config(n_days=n_days, steps_per_day=steps_per_day)
    assert run(config).digest() == run(config).digest()
