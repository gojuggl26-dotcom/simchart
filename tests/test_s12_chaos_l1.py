"""S12: χ₁/χ₃ (L1 のカオス駆動) のテスト。

中心は 4 つ:
1. **S11 経路のビット単位不変** (χ フラグ off。χ は乱数を消費しない)。
2. **3 系列の独立性** (|corr| < 0.1 — 同一系・異初期値・異写像)。
3. **χ₁ の平均中立** (数値凸性補正で E[e^{a₁χ₁−c}] = 1) と分散シェア。
4. **χ₃ の决定論・シード非依存** (窓再現性検証 §8 の前提)。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from simchart import Config, run
from test_s10_coupling import ROOT, S, _s10_cfg


def _s12_cfg(seed: int = 1301, n_days: int = 40, **extra) -> Config:
    kw = dict(
        stage="S12", seed=seed, n_days=n_days, kappa=0.2, c_vol=0.65,
        sigma_bar=0.2217, enable_feedback=True,
        fb_b_delta=0.3, fb_b_place=0.3, fb_b_n=2.0,
        fb_c_delta=0.018, fb_c_place=0.018,
        enable_chaos_lambda=True, enable_chaos_branching=True, chi3_b=1.0,
    )
    kw.update(extra)
    return _s10_cfg(**kw)


def test_s11_production_digest_bit_identical():
    metrics_path = ROOT / "results" / "S11" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("S11 本番の results が無い環境ではスキップ")
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    want = stored["metrics"]["runtime"]["determinism"]["digest_first"]
    r = run(Config.load(ROOT / "configs" / "s11.yaml"))
    assert r.digest() == want, "S11 経路の出力が変わった — S12 改修が前段を汚染"


def test_chi_independence():
    from simchart.chaos import chi_window
    from simchart.layers.l2_price import prepare_chaos_component

    cfg = _s12_cfg()
    n_days = 2000.0
    t1, x1, _ = chi_window(cfg, n_days, "chi1")
    t3, x3, _ = chi_window(cfg, n_days, "chi3")
    t2, x2, _, _, _ = prepare_chaos_component(cfg, n_days)
    grid = np.arange(1.0, n_days - 1.0, 0.5)
    g1 = np.interp(grid, t1, x1)
    g2 = np.interp(grid, t2, x2)
    g3 = np.interp(grid, t3, x3)
    for a, b, name in ((g1, g2, "chi1-chi2"), (g1, g3, "chi1-chi3"),
                       (g2, g3, "chi2-chi3")):
        c = float(np.corrcoef(a, b)[0, 1])
        assert abs(c) < 0.1, f"{name}: corr={c:.3f}"


@pytest.fixture(scope="module")
def chaos_run():
    cfg = _s12_cfg()
    return run(cfg), cfg


def test_chi1_mean_neutral_and_budget(chaos_run):
    r, cfg = chaos_run
    d = r.meta["l3"]["cvol"]["chi1"]
    # 数値凸性補正の閉ループ: E[e^{a₁χ₁ − c}] = 1 (窓上、厳密)
    assert abs(d["e_factor"] - 1.0) < 1e-9, d["e_factor"]
    # a₁ の導出: c_vol·√(share/(1−share))
    want = cfg.c_vol * np.sqrt(cfg.chi1_var_share / (1 - cfg.chi1_var_share))
    assert abs(d["a1"] - want) < 1e-12
    # 数値補正はガウス公式と有意に違う (補正が効いている証拠 — S5 と同型)
    assert abs(d["c_chi1_numerical"] - d["c_chi1_gaussian_formula"]) > 1e-4


def test_chi3_seed_independent(chaos_run):
    """χ₃ はシード非依存 (窓再現性 §8 の前提)。"""
    r1, cfg = chaos_run
    r2 = run(_s12_cfg(seed=1302))
    d1 = r1.meta["l3"]["chi3"]
    d2 = r2.meta["l3"]["chi3"]
    assert d1["sha256"] == d2["sha256"]
    # n_t は χ₃ で動く: nt_sd が χ₃ なし (b_chi=0 相当 = S11 構成) より大きい
    r0 = run(_s12_cfg(seed=1301, enable_chaos_branching=False, chi3_b=0.0))
    sd_with = r1.meta["l3"]["feedback"]["nt_sd"]
    sd_without = r0.meta["l3"]["feedback"]["nt_sd"]
    assert sd_with > sd_without, (sd_with, sd_without)


def test_deterministic(chaos_run):
    r, cfg = chaos_run
    r2 = run(cfg)
    assert r.digest() == r2.digest()


def test_config_validation():
    with pytest.raises(ValueError, match="c_vol"):
        _s10_cfg(stage="S12", kappa=0.2, sigma_bar=0.2217, c_vol=0.0,
                 enable_chaos_lambda=True)
    with pytest.raises(ValueError, match="enable_feedback"):
        _s10_cfg(stage="S12", kappa=0.2, c_vol=0.65,
                 enable_chaos_branching=True, chi3_b=1.0)
    with pytest.raises(ValueError, match="no-op|暗黙"):
        _s12_cfg(chi3_b=0.0)
    with pytest.raises(ValueError, match="既定値|no-op"):
        _s10_cfg(stage="S12", kappa=0.2, c_vol=0.65, chi1_ic=0.7)
    base = _s12_cfg().without_book()
    assert base.enable_chaos_lambda is False
    assert base.enable_chaos_branching is False and base.chi3_b == 0.0