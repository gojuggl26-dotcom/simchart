"""``PriceProcess`` の補間契約。

S10 で L3 が不規則なイベント時刻に p* を問い合わせるようになるため、``at()`` の
振る舞い (格子点で厳密一致・格子間で単調・範囲外は拒否・冪等) をここで固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, run
from simchart.types import PriceProcess


@pytest.fixture
def process() -> PriceProcess:
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 1000.0)
    log_p = np.cumsum(rng.standard_normal(t.size)) * 0.001 + np.log(100.0)
    return PriceProcess(t=t, log_p_star=log_p, log_vol=np.full(t.size, np.log(0.2)))


def test_at_is_exact_on_grid_points(process: PriceProcess) -> None:
    values = process.at(process.t)
    assert np.array_equal(values, process.log_p_star)
    for index in (0, 1, 17, 499, len(process.t) - 1):
        assert process.at(float(process.t[index])) == process.log_p_star[index]


def test_at_is_monotone_between_grid_points(process: PriceProcess) -> None:
    """格子間の値が必ず両端の間に入ること (線形補間なので単調)。"""
    rng = np.random.default_rng(1)
    index = rng.integers(0, process.t.size - 1, size=500)
    frac = rng.uniform(0.0, 1.0, size=500)
    query = process.t[index] + frac * (process.t[index + 1] - process.t[index])
    values = process.at(query)
    lo = np.minimum(process.log_p_star[index], process.log_p_star[index + 1])
    hi = np.maximum(process.log_p_star[index], process.log_p_star[index + 1])
    assert np.all(values >= lo - 1e-15)
    assert np.all(values <= hi + 1e-15)


def test_at_is_idempotent(process: PriceProcess) -> None:
    """同じ時刻を何度問い合わせても同じ値。

    ブラウン橋のように問い合わせのたびに乱数を引く実装にすると、S10 で
    「L3 を変えても L2 は不変」という保証が壊れる。
    """
    query = np.array([0.5, 12.25, 800.125])
    assert np.array_equal(process.at(query), process.at(query))
    assert process.at(0.5) == process.at(0.5)


def test_at_accepts_scalars_and_arrays(process: PriceProcess) -> None:
    scalar = process.at(3.5)
    assert isinstance(scalar, float)
    array = process.at(np.array([3.5]))
    assert isinstance(array, np.ndarray) and array.shape == (1,)
    assert scalar == array[0]


def test_at_refuses_extrapolation(process: PriceProcess) -> None:
    with pytest.raises(ValueError):
        process.at(-1.0)
    with pytest.raises(ValueError):
        process.at(float(process.t[-1]) + 1.0)


def test_vol_at_matches_grid(process: PriceProcess) -> None:
    assert np.array_equal(process.vol_at(process.t), process.log_vol)
    assert process.vol_at(10.5) == pytest.approx(np.log(0.2))


def test_rejects_non_monotone_grid() -> None:
    with pytest.raises(ValueError):
        PriceProcess(
            t=np.array([0.0, 2.0, 1.0]),
            log_p_star=np.zeros(3),
            log_vol=np.zeros(3),
        )


def test_rejects_unplanned_interpolation() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        PriceProcess(
            t=np.arange(3.0),
            log_p_star=np.zeros(3),
            log_vol=np.zeros(3),
            interpolation="brownian_bridge",
        )
    assert "S10" in str(excinfo.value)


def test_pipeline_returns_a_price_process_not_an_array() -> None:
    """S10 で L3 が ``at()`` を使えるよう、L2 の戻り値が配列でないこと。"""
    result = run(Config(n_days=2, steps_per_day=390))
    assert isinstance(result.price, PriceProcess)
    assert callable(result.price.at)
    midpoint = 0.5 * (result.price.t[10] + result.price.t[11])
    value = result.price.at(midpoint)
    assert min(result.price.log_p_star[10], result.price.log_p_star[11]) <= value
    assert value <= max(result.price.log_p_star[10], result.price.log_p_star[11])


def test_bar_resampling_respects_session_boundaries() -> None:
    """バー再標本化がセッションをまたぐ差分を作らないこと。"""
    config = Config(n_days=4, steps_per_day=390)
    result = run(config)
    bars = result.observation.to_bars(600.0)  # 10 分バー
    assert bars.n_days == 4
    assert bars.n_bars_per_day == 39
    assert bars.returns_2d().shape == (4, 39)
    assert bars.n_returns == 4 * 39
    # 各セッションの列 0 は、そのセッション開始時刻の観測値と一致する。
    session = result.observation.session_seconds
    for day in range(4):
        expected = result.price.at(day * session)
        assert bars.log_price[day, 0] == pytest.approx(expected)
