"""S9: 検証系 (η・OBI 予測相関・戻り曲線・_qr_metrics) のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, run
from simchart.validation.micro import (
    estimate_eta,
    mean_reversion_profile,
    obi_predictive,
)
from simchart.validation.suite import _qr_metrics


def test_estimate_eta_calibration():
    rng = np.random.default_rng(1)
    # iid ±1 変化 → η ≈ 0.5
    iid = np.cumsum(rng.choice([-1.0, 1.0], 50_000))
    e_iid = estimate_eta(iid)
    assert abs(e_iid["eta"] - 0.5) < 0.02, e_iid["eta"]
    # 完全交替 → η → 0
    alt = np.tile([0.0, 1.0], 5_000)
    assert estimate_eta(alt)["eta"] < 0.01
    # 継続バイアス (p_cont = 0.75) → η = p/(2(1−p)) = 1.5
    steps = np.empty(50_000)
    steps[0] = 1.0
    u = rng.random(50_000)
    for i in range(1, 50_000):
        steps[i] = steps[i - 1] if u[i] < 0.75 else -steps[i - 1]
    e_cont = estimate_eta(np.cumsum(steps))
    assert abs(e_cont["eta"] - 1.5) < 0.1, e_cont["eta"]


def test_obi_predictive_power_and_null():
    rng = np.random.default_rng(2)
    n = 50_000
    imb = rng.uniform(-1, 1, n)
    noise = rng.normal(0, 1, n)
    mid = np.cumsum(0.5 * imb + noise)  # I が次の変化を先行
    # 理論相関 = 0.5·SD(I)/SD(0.5I+ノイズ) ≈ 0.277 (実測 0.276 と一致)
    res = obi_predictive(imb, np.concatenate([[0.0], mid[:-1]]), horizons=(1, 5))
    assert res["status"] == "ok" and res["corr_h1"] > 0.2
    # 帰無: 独立
    res0 = obi_predictive(rng.uniform(-1, 1, n), np.cumsum(rng.normal(0, 1, n)))
    assert abs(res0["corr_h1"]) < 0.02


def test_mean_reversion_profile_detects_reversion():
    rng = np.random.default_rng(3)
    n = 60_000
    signs = rng.choice([-1.0, 1.0], n)
    # インパクト 1 → 半分だけ戻る合成ミッド (h=1 で +1、h>=3 で +0.5)
    mid = np.zeros(n)
    impact = np.zeros(n + 4)
    for k in range(n):
        impact[k + 1] += signs[k] * 1.0
        impact[k + 3] -= signs[k] * 0.5
    mid = np.cumsum(impact[:n]) + rng.normal(0, 0.5, n)
    res = mean_reversion_profile(signs, mid)
    assert res["status"] == "ok"
    assert res["monotone_nondecreasing"] is False
    assert res["reversion_frac"] > 0.2
    # 帰無: 符号と独立なミッド → プロファイルはゼロ近傍で戻り検出なし
    res0 = mean_reversion_profile(signs, np.cumsum(rng.normal(0, 1, n)))
    assert abs(res0["impact_peak"]) < 0.05


@pytest.fixture(scope="module")
def qr_metrics_small():
    cfg = Config.load(
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "configs" / "s8.yaml"
    ).replace(stage="S9", seed=917, n_days=40, enable_queue_reactive=True)
    r = run(cfg)
    return _qr_metrics(r, cfg), cfg


def test_qr_metrics_all_branches_ok(qr_metrics_small):
    m, _ = qr_metrics_small
    for key in ("eta_trade", "eta_trade_rows", "eta_mid", "mid_return_acf",
                "signature_mid", "obi", "state_diag", "reversion",
                "depth_tick_profile"):
        assert m[key]["status"] == "ok", (key, m[key])


def test_qr_metrics_bands(qr_metrics_small):
    m, _ = qr_metrics_small
    assert 0.05 < m["eta_trade"]["eta"] < 0.35
    assert m["eta_trade"]["uz_enabled"] is False
    assert m["mid_return_acf"]["change_sign_corr_event"] < -0.02
    assert m["obi"]["corr_h1"] > 0.08
    assert m["state_diag"]["inspread_monotone"] is True
    assert m["reversion"]["monotone_nondecreasing"] is False
    assert 2 <= m["depth_tick_profile"]["peak_tick_distance"] <= 10


def test_qr_metrics_na_when_disabled():
    cfg = Config.load(
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "configs" / "s8.yaml"
    ).replace(seed=917, n_days=5)
    r = run(cfg)
    m = _qr_metrics(r, cfg)
    assert all(v["status"] == "not_applicable" for v in m.values())