"""S5: 決定論的カオス成分 chi_2 のテスト。

中心は 2 つ:
- test_chi_removal_recovers_s4_exactly — chi は決定論の加算なので、引けば S4 の
  log σ が機械精度で戻る。「既存成分を触っていない」ことの最強の検定
- test_rng_fingerprint_is_identical_to_s4 — chi は乱数を 1 draw も消費しない
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simchart import Config, run
from simchart.chaos import chaos_generate, mackey_glass
from simchart.layers.l2_price import prepare_chaos_component
from simchart.validation.chaos import (
    correlation_dimension,
    lyapunov_rosenstein,
    test_0_1_chaos,
)
from simchart.validation.scaling import (
    cross_seed_correlation,
    marginal_normality,
    spectral_peak,
)

S4KW = dict(
    enable_msm=True, enable_slow_ou=True, enable_rough=True,
    enable_jump=True, enable_leverage=True,
    enable_seasonality=True, enable_overnight=True,
    jump_lambda_per_year=5.0, jump_eta_down=35.0, jump_eta_up=56.0,
    jump_qv_share_target=0.12, jump_p_up=0.42,
    leverage_rho_rough=-0.60, leverage_rho_slow=-0.35,
)
SMALL = dict(n_days=400, steps_per_day=390)


def _pair(seed: int = 42, **extra):
    c5 = Config(stage="S5", seed=seed, enable_chaos_vol=True, **extra, **S4KW, **SMALL)
    c4 = Config(stage="S4", seed=seed, **extra, **S4KW, **SMALL)
    return run(c5), run(c4)


# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------
def test_mackey_glass_is_deterministic():
    t1, x1 = mackey_glass(length_units=500.0, dt=0.1)
    t2, x2 = mackey_glass(length_units=500.0, dt=0.1)
    assert np.array_equal(x1, x2)
    assert np.array_equal(t1, t2)


def test_mackey_glass_rejects_non_integer_delay_ratio():
    with pytest.raises(ValueError):
        mackey_glass(length_units=100.0, dt=0.3, tau=17.0)  # 17/0.3 は非整数


def test_chaos_cache_roundtrip(tmp_path):
    """キャッシュ保存 → ロードでハッシュが一致し、値もビット単位で同一。"""
    kwargs = dict(
        system="mackey_glass", params={"tau": 17.0}, length_units=300.0,
        dt=0.1, ic=1.2, burn_in_units=200.0,
    )
    first = chaos_generate(cache_dir=tmp_path, **kwargs)
    assert first.cache_path is not None
    second = chaos_generate(cache_dir=tmp_path, **kwargs)
    assert second.sha256 == first.sha256
    assert np.array_equal(second.x, first.x)
    fresh = chaos_generate(cache_dir=None, **kwargs)
    assert fresh.sha256 == first.sha256


def test_chaos_validators_separate_chaos_from_null():
    """MG は (正 Lyapunov, 低次元飽和, K~1)。正弦波は (λ~0, K~0)。"""
    _, x = mackey_glass(length_units=8000.0, dt=0.1)
    ly = lyapunov_rosenstein(x, dt=0.1)
    assert ly["status"] == "ok" and ly["lyapunov_per_unit"] > 0.003
    assert ly["fit_r2"] > 0.9
    cd = correlation_dimension(x, dt=0.1)
    assert cd["status"] == "ok" and 1.5 <= cd["d2"] <= 3.0
    z1 = test_0_1_chaos(x, subsample=60)
    assert z1["K"] > 0.8

    tt = np.arange(x.size) * 0.1
    sine = np.sin(2 * np.pi * tt / 49.65)
    ly_s = lyapunov_rosenstein(sine, dt=0.1)
    assert abs(ly_s["lyapunov_per_unit"]) < 0.001
    z1_s = test_0_1_chaos(sine, subsample=60)
    assert z1_s["K"] < 0.3


# ---------------------------------------------------------------------------
# 中心: 厳密復元と RNG 不変
# ---------------------------------------------------------------------------
def test_chi_removal_recovers_s4_exactly():
    """S5 の log σ から chi 項を引くと S4 の log σ が機械精度で戻る。

    chi は決定論の加算なので (置換ではない §15)、これが成り立たなければ
    既存成分のどこかを触っている。許容 1e-12 は float64 の加減算誤差の水準。
    """
    r5, r4 = _pair()
    s5, s4 = r5.meta["l2"]["vol_subsample"], r4.meta["l2"]["vol_subsample"]
    recovered = s5["log_vol"] - s5["chi_term"] + s5["c_chi"]
    assert np.abs(recovered - s4["log_vol"]).max() < 1e-12


def test_rng_fingerprint_is_identical_to_s4():
    """chi_2 は乱数を 1 draw も消費しない 。"""
    r5, r4 = _pair()
    assert r5.rng_fingerprint == r4.rng_fingerprint
    assert sorted(r5.meta["rng_streams_used"]) == sorted(r4.meta["rng_streams_used"])


def test_stochastic_stream_digests_unchanged():
    r5, r4 = _pair()
    for path in (("diffusion_digest",), ("msm", "switch_digest"), ("rough", "y_digest")):
        a, b = r5.meta["l2"], r4.meta["l2"]
        for part in path:
            a, b = a[part], b[part]
        assert a == b, f"{path} が変化しました"


def test_chi_is_identical_across_seeds():
    """chi は config のみから決まる — シードを変えてもビット単位で同一。"""
    r_a = run(Config(stage="S5", seed=42, enable_chaos_vol=True, **S4KW, **SMALL))
    r_b = run(Config(stage="S5", seed=99, enable_chaos_vol=True, **S4KW, **SMALL))
    assert (
        r_a.meta["l2"]["chaos"]["sha256"] == r_b.meta["l2"]["chaos"]["sha256"]
    )
    np.testing.assert_array_equal(
        r_a.meta["l2"]["vol_subsample"]["chi_term"],
        r_b.meta["l2"]["vol_subsample"]["chi_term"],
    )


# ---------------------------------------------------------------------------
# 数値凸性補正と予算
# ---------------------------------------------------------------------------
def test_numerical_convexity_differs_from_gaussian_and_is_exact():
    """MG はガウスでないので c_chi ≠ a² (ガウス公式)。数値補正なら
    time-mean(e^{2(aχ−c_χ)}) = 1 が厳密に成り立つ (これが E[σ²]=σ̄² の根拠)。"""
    cfg = Config(stage="S5", enable_chaos_vol=True, **S4KW, **SMALL)
    r = run(cfg)
    ch = r.meta["l2"]["chaos"]
    assert abs(ch["c_chi_difference"]) > 1e-4, "ガウス公式と区別がつかない (補正が数値になっていない?)"

    _, chi, a, c_chi, _ = prepare_chaos_component(cfg, float(cfg.n_days))
    assert float(np.mean(np.exp(2.0 * (a * chi - c_chi)))) == pytest.approx(1.0, abs=1e-12)
    # ガウス公式を流用した場合のずれ (これがなんとなくボラが高いの正体)
    wrong = float(np.mean(np.exp(2.0 * (a * chi - a * a))))
    assert abs(wrong - 1.0) > 1e-3


def test_ensemble_e_sigma2_holds_with_chi():
    """E[σ²] = σ̄² が chi 込みで保たれる (アンサンブル断面 + chi 周辺分布)。"""
    from simchart.validation.ensemble import vol_cross_section

    cfg = Config(stage="S5", enable_chaos_vol=True, **S4KW, **SMALL)
    out = vol_cross_section(cfg, n_paths=100_000)
    assert out["status"] == "ok"
    assert abs(out["e_sigma2_ratio"] - 1.0) < 0.02
    assert out["shares_of_budget"]["chaos"] == pytest.approx(0.20, abs=0.01)
    assert 0.235 <= out["var_log_sigma"] <= 0.265


def test_chi_path_variance_is_exactly_a_squared():
    """窓正規化により chi 項の経路分散は a² = 0.05 (サブサンプルで ±1%)。"""
    r = run(Config(stage="S5", enable_chaos_vol=True, **S4KW, **SMALL))
    v = float(np.var(r.meta["l2"]["vol_subsample"]["chi_term"]))
    assert v == pytest.approx(0.05, rel=0.02)


def test_budget_guard_rejects_overallocation():
    """chi のシェア 25% 超は (3)(18) を薄めるため config が弾く (§15)。"""
    with pytest.raises(ValueError):
        Config(
            stage="S5", enable_chaos_vol=True, vol_var_target_chaos=0.10,
            **S4KW, **SMALL,
        )


def test_jump_qv_share_is_preserved_with_chi():
    """chi の Jensen 効果 (E[e^{aχ−c}] < 1) を強度補正が打ち消す。"""
    r5, r4 = _pair()
    j5, j4 = r5.meta["l2"]["jump"], r4.meta["l2"]["jump"]
    assert j5["jv_share_theory"] == pytest.approx(j4["jv_share_theory"], abs=5e-4)
    jf = r5.meta["l2"]["chaos"]["jensen_intensity_factor"]
    assert jf < 1.0  # Cauchy-Schwarz
    assert j5["intensity_scale_s4"] == pytest.approx(j4["intensity_scale_s4"] / jf, rel=1e-9)


# ---------------------------------------------------------------------------
# 時間写像・周辺分布・希釈
# ---------------------------------------------------------------------------
def test_spectral_peak_lands_in_the_mandated_window():
    cfg = Config(stage="S5", enable_chaos_vol=True, **S4KW, **SMALL)
    _, chi, _a, _c, diag = prepare_chaos_component(cfg, float(cfg.n_days))
    sp = spectral_peak(chi, diag["grid_spacing_days"])
    assert sp["status"] == "ok"
    assert 20.0 <= sp["peak_period_days"] <= 40.0
    assert sp["daily_band_power_share"] < 0.01


def test_composite_log_vol_stays_unimodal():
    """MG 単体は 4 峰だが、分散比 1:4 の合成で単峰になる (§3.2 の案 A で足りる根拠)。"""
    r = run(Config(stage="S5", enable_chaos_vol=True, **S4KW, **SMALL))
    sub = r.meta["l2"]["vol_subsample"]
    lv = np.asarray(sub["log_vol"]) - np.asarray(sub["log_phi_sigma"])
    mn = marginal_normality(lv)
    assert mn["unimodal"] and abs(mn["excess_kurtosis"]) < 1.0


def test_ecdf_normal_option_is_deterministic_and_gaussianizes():
    cfg = Config(
        stage="S5", enable_chaos_vol=True, chaos_normalization="ecdf_normal",
        **S4KW, **SMALL,
    )
    _, chi1, _a, _c1, _ = prepare_chaos_component(cfg, float(cfg.n_days))
    _, chi2, _a2, _c2, _ = prepare_chaos_component(cfg, float(cfg.n_days))
    assert np.array_equal(chi1, chi2)
    from scipy import stats as st

    assert abs(st.kurtosis(chi1)) < 0.2  # 周辺が正規化されている


def test_dilution_sd_ratio_matches_theory():
    """sd(log σ_S4)/sd(log σ_S5) ≈ sqrt(V/(V+0.05)) (2026-08-21 裁定の判定計器)。"""
    r5, _ = _pair()
    sub = r5.meta["l2"]["vol_subsample"]
    lv_with = np.asarray(sub["log_vol"]) - np.asarray(sub["log_phi_sigma"])
    lv_without = lv_with - np.asarray(sub["chi_term"]) + sub["c_chi"]
    ratio = float(np.sqrt(lv_without.var() / lv_with.var()))
    v = float(lv_without.var())
    theory = math.sqrt(v / (v + 0.05))
    assert ratio == pytest.approx(theory, abs=0.02)
    assert 0.85 <= ratio <= 0.95


def test_cross_seed_correlation_reflects_chi_share():
    """シード横断相関 ≈ Var(chi)/Var(log σ)。400 日では遅い成分の経路分散が
    不安定なので帯は広め — 本番 (5000 日 × 45 対) の判定は [0.17, 0.23]。"""
    paths = []
    for seed in (42, 43, 44):
        r = run(Config(stage="S5", seed=seed, enable_chaos_vol=True, **S4KW, **SMALL))
        sub = r.meta["l2"]["vol_subsample"]
        paths.append(np.asarray(sub["log_vol"]) - np.asarray(sub["log_phi_sigma"]))
        del r
    csc = cross_seed_correlation(paths)
    assert csc["status"] == "ok"
    assert 0.05 < csc["mean"] < 0.45


def test_chi_has_no_directional_leak():
    """§15 の第一禁止事項: chi はリターンの方向と無相関 (σ にのみ入る)。"""
    r = run(Config(stage="S5", enable_chaos_vol=True, **S4KW, **SMALL))
    obs = r.observation
    rd = obs.to_bars(obs.session_seconds).returns()
    sub = r.meta["l2"]["vol_subsample"]
    per_day = 390
    chi_daily = np.asarray(sub["chi_term"])[: rd.size * per_day].reshape(rd.size, per_day).mean(axis=1)
    c = float(np.corrcoef(rd, chi_daily)[0, 1])
    assert abs(c) < 4.0 / math.sqrt(rd.size)


# ---------------------------------------------------------------------------
# 検証スイート・フラグ
# ---------------------------------------------------------------------------
def test_suite_produces_chaos_branch():
    from simchart.validation import run_all

    m5 = run_all(run(Config(stage="S5", enable_chaos_vol=True, **S4KW, **SMALL)))
    ch = m5["chaos"]
    assert ch["generator"]["status"] == "ok"
    assert ch["chi_tests"]["lyapunov"]["lyapunov_per_unit"] > 0
    assert ch["latent_gph_ablation"]["delta_bw050"] is not None
    assert ch["dilution"]["sd_ratio"] is not None
    assert m5["vol"]["ensemble"]["shares_of_budget"]["chaos"] is not None

    m4 = run_all(run(Config(stage="S4", **S4KW, **SMALL)))
    assert m4["chaos"]["generator"]["status"] == "not_applicable"


def test_chaos_params_guarded_when_flag_off():
    with pytest.raises(ValueError):
        Config(stage="S4", chaos_tau_delay=30.0, **S4KW, **SMALL)


def test_chaos_alone_without_stochastic_vol_works():
    """確率ボラ全 off + chi のみでも log σ が定数 + chi になる (早期リターン修正の検証)。"""
    cfg = Config(stage="S5", enable_chaos_vol=True, n_days=50, steps_per_day=390)
    r = run(cfg)
    lv = r.price.log_vol
    assert float(lv.std()) == pytest.approx(math.sqrt(0.05), rel=0.05)


def test_scale_invariance_of_chi_daily_stats():
    """chi は市場日で定義されるので steps_per_day に依存しない (日次 RV が一致)。"""
    kw = dict(stage="S5", enable_chaos_vol=True, **S4KW)
    rv = {}
    for spd in (390, 780):
        r = run(Config(steps_per_day=spd, n_days=120, **kw))
        step_r = np.diff(r.observation.log_price)
        rv[spd] = (step_r**2).reshape(120, spd).sum(axis=1)
        del r
    # 乱数列が違うので値は別物 — 比べるのは chi 由来の日次パターン (相関で見る)。
    sub_corr = float(np.corrcoef(np.log(rv[390]), np.log(rv[780]))[0, 1])
    assert sub_corr > 0.2  # 共通の決定論成分 (chi + 日次パターン) が乗っている
