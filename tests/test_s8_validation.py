"""S8: 検証系 (γ 推定・長さ MLE・プール定常性・propagator・平方根則) のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, run
from simchart.validation.micro import (
    iceberg_stats,
    metaorder_length_check,
    pool_stationarity,
)
from simchart.validation.suite import _meta_metrics


def test_length_mle_recovers_alpha():
    rng = np.random.default_rng(7)
    for alpha in (1.4, 1.6, 1.8):
        u = rng.random(200_000)
        lengths = np.floor((1.0 - u) ** (-1.0 / alpha))
        res = metaorder_length_check(lengths, alpha)
        assert res["status"] == "ok"
        assert abs(res["alpha_hat"] - alpha) < 0.02, (alpha, res["alpha_hat"])


def test_length_mle_detects_wrong_alpha():
    """検定力: 仕様 1.6 に対し α=1.3 のデータは差分で見える。"""
    rng = np.random.default_rng(8)
    u = rng.random(100_000)
    lengths = np.floor((1.0 - u) ** (-1.0 / 1.3))
    res = metaorder_length_check(lengths, alpha_spec=1.6)
    assert abs(res["alpha_hat"] - 1.3) < 0.02
    assert abs(res["difference"]) > 0.25  # ゲート帯 ±0.1 で確実に落ちる


def test_pool_stationarity_passes_and_fails():
    rng = np.random.default_rng(9)
    flat = rng.poisson(5.0, 50_000).astype(float)
    res = pool_stationarity(flat)
    assert res["status"] == "ok" and abs(res["rel_diff"]) < 0.05
    trending = flat + np.linspace(0, 5, flat.size)  # 平均 5 → 10 へ線形増加
    res2 = pool_stationarity(trending)
    assert abs(res2["rel_diff"]) > 0.10  # ゲートが検出できること (検定力)


def test_iceberg_stats_na_paths():
    assert iceberg_stats(None)["status"] == "not_applicable"
    assert iceberg_stats({"n_iceberg_orders": 0})["status"] == "not_applicable"


@pytest.fixture(scope="module")
def meta_metrics_small():
    cfg = Config.load(
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "configs" / "s7.yaml"
    ).replace(
        stage="S8", seed=515, n_days=40, steps_per_day=390,
        enable_metaorder=True, enable_iceberg=True,
        book_window_half_ticks=9500,
    )
    r = run(cfg)
    return _meta_metrics(r, cfg), cfg


def test_meta_metrics_all_branches_ok(meta_metrics_small):
    m, _ = meta_metrics_small
    for key in ("sign_acf_gamma", "length_fit", "pool", "flow_balance", "iceberg",
                "response_mid", "propagator_mid", "propagator_stability",
                "sqrt_law", "impact_vs_size", "impact_deficit"):
        assert m[key]["status"] == "ok", (key, m[key])


def test_flow_balance_identity(meta_metrics_small):
    m, cfg = meta_metrics_small
    fb = m["flow_balance"]
    assert abs(fb["balance_ratio"] - 1.0) < 0.05


def test_impact_deficit_directions(meta_metrics_small):
    """壊れ方の方向 (§8.1): β は目標より小さく、サイズ応答はほぼ線形。

    線形性はビン平均の傾き (impact_vs_size) で見る — frozen の
    sqrt_law_check は正のみ選別が小 N を上方バイアスして傾きが潰れる
    (実測: 生 Q でも 0.37。ビン平均だと 0.89)。
    """
    m, _ = meta_metrics_small
    d = m["impact_deficit"]
    assert d["beta_deficit"] < 0.0, d  # 板は適応しない → 減衰の赤字
    assert d["sqrt_law_exponent"] > 0.6, d  # 子数比例の単純加算 (ビン平均傾き)


def test_propagator_nearly_flat(meta_metrics_small):
    """G(ℓ) がほぼ減衰しない (β ≈ 0) — インパクトが実質恒久 (§8.1)。"""
    m, _ = meta_metrics_small
    p = m["propagator_mid"]
    assert p["status"] == "ok"
    assert p["beta"] < 0.10, p["beta"]  # 目標 (1−γ)/2 ≈ 0.2 に遠く届かない


def test_meta_metrics_na_when_disabled():
    cfg = Config(stage="S8", n_days=5, steps_per_day=390, enable_book=True)
    r = run(cfg)
    m = _meta_metrics(r, cfg)
    assert all(v["status"] == "not_applicable" for v in m.values())