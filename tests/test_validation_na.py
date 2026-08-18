"""該当層が無い検証関数が「例外ではなく構造化された N/A」を返すことの確認。

検証関数を「まだ使えないから」と省略しないこと、そして空入力で落ちないことが
S0 の設計の中心にある。micro / cross はスタブではなく本実装なので、空入力の
扱いだけをここで固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, run
from simchart.validation import cross, micro
from simchart.validation.base import STATUS_ERROR, STATUS_NA, STATUS_OK
from simchart.validation.suite import collect_errors, run_all

SMALL = Config(n_days=20, steps_per_day=2340)


@pytest.fixture(scope="module")
def metrics() -> dict:
    return run_all(run(SMALL), SMALL)


def _iter_leaves(node, prefix=""):
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            if "status" in value:
                yield path, value
            else:
                yield from _iter_leaves(value, prefix=f"{path}.")


def test_no_validation_function_raises(metrics: dict) -> None:
    assert collect_errors(metrics) == []


def test_every_leaf_has_a_valid_status(metrics: dict) -> None:
    leaves = dict(_iter_leaves({k: v for k, v in metrics.items() if k != "series"}))
    assert leaves, "検証結果が 1 つも見つかりません"
    for path, node in leaves.items():
        assert node["status"] in (STATUS_OK, STATUS_NA, STATUS_ERROR), path
        assert node["status"] != STATUS_ERROR, f"{path}: {node.get('error')}"


def test_micro_group_is_not_applicable(metrics: dict) -> None:
    for name, node in metrics["micro"].items():
        assert node["status"] == STATUS_NA, f"micro.{name} が N/A ではありません"
        assert node["value"] is None
        assert isinstance(node["reason"], str) and node["reason"]


def test_cross_group_is_not_applicable(metrics: dict) -> None:
    for name, node in metrics["cross"].items():
        assert node["status"] == STATUS_NA, f"cross.{name} が N/A ではありません"
        assert node["value"] is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: micro.sign_acf(None),
        lambda: micro.sign_acf(np.empty(0)),
        lambda: micro.response_function(None, None),
        lambda: micro.propagator_fit(None, None, None),
        lambda: micro.impact_consistency(None, None),
        lambda: micro.sqrt_law_check(None),
        lambda: micro.sqrt_law_check([]),
        lambda: micro.branching_ratio_reestimate(None),
        lambda: micro.branching_ratio_reestimate(np.empty(0)),
        lambda: cross.hayashi_yoshida(None, None),
        lambda: cross.hayashi_yoshida((np.arange(5.0), np.zeros(5)), None),
        lambda: cross.hayashi_yoshida_lead_lag(None, None, (-1.0, 0.0, 1.0)),
    ],
)
def test_empty_inputs_return_na_without_raising(call) -> None:
    result = call()
    assert result["status"] == STATUS_NA
    assert result["value"] is None
    assert result["reason"]


def test_micro_functions_work_on_synthetic_flow() -> None:
    """空でない入力なら本当に推定できること (スタブでないことの確認)。

    符号に人工的な正の相関を入れ、価格をその累積で作る。真の値を当てにいく
    テストではなく、「計算経路が生きている」ことの確認。
    """
    rng = np.random.default_rng(0)
    n = 20_000
    signs = np.empty(n)
    signs[0] = 1.0
    for i in range(1, n):
        signs[i] = signs[i - 1] if rng.random() < 0.75 else -signs[i - 1]
    log_price = np.cumsum(0.0001 * signs + 0.0002 * rng.standard_normal(n))

    sign_result = micro.sign_acf(signs, max_lag=100, fit_lag_range=(2, 50))
    assert sign_result["status"] == STATUS_OK
    assert sign_result["lag1"] > 0.3

    response = micro.response_function(signs, log_price, max_lag=50)
    assert response["status"] == STATUS_OK
    assert response["values"][0] > 0

    propagator = micro.propagator_fit(signs, None, log_price, max_lag=50, fit_lag_range=(2, 40))
    assert propagator["status"] in (STATUS_OK, STATUS_NA)

    consistency = micro.impact_consistency(0.5, 0.25)
    assert consistency["status"] == STATUS_OK
    assert consistency["beta_predicted"] == pytest.approx(0.25)
    assert consistency["difference"] == pytest.approx(0.0)


def test_sqrt_law_recovers_the_planted_exponent() -> None:
    rng = np.random.default_rng(1)
    n = 500
    ratio = np.exp(rng.uniform(-6, -1, n))
    sigma = np.full(n, 0.02)
    impact = sigma * (ratio**0.5) * np.exp(rng.normal(0, 0.05, n))
    records = [
        {"q": r * 1e6, "v": 1e6, "sigma": s, "impact": i}
        for r, s, i in zip(ratio, sigma, impact)
    ]
    result = micro.sqrt_law_check(records)
    assert result["status"] == STATUS_OK
    assert result["delta"] == pytest.approx(0.5, abs=0.02)


def test_hayashi_yoshida_recovers_correlation() -> None:
    """非同期に間引いた 2 系列から相関が戻ること。"""
    rng = np.random.default_rng(2)
    n = 40_000
    rho = 0.7
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    t = np.arange(n + 1, dtype=float)
    p1 = np.concatenate([[0.0], np.cumsum(z1)])
    p2 = np.concatenate([[0.0], np.cumsum(z2)])

    keep1 = np.sort(rng.choice(np.arange(1, n + 1), size=n // 3, replace=False))
    keep2 = np.sort(rng.choice(np.arange(1, n + 1), size=n // 4, replace=False))
    a1 = (np.concatenate([[0.0], t[keep1]]), np.concatenate([[0.0], p1[keep1]]))
    a2 = (np.concatenate([[0.0], t[keep2]]), np.concatenate([[0.0], p2[keep2]]))

    result = cross.hayashi_yoshida(a1, a2)
    assert result["status"] == STATUS_OK
    assert result["correlation"] == pytest.approx(rho, abs=0.05)
