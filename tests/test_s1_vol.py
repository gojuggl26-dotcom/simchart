"""S1 の確率ボラ (MSM + 緩慢 OU) の単体検証。

分散予算・凸性補正・厳密離散化・RNG 分離という「後段全体を規定する」性質を固定する。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from simchart import Config, run
from simchart.layers.l2_price import (
    compose_log_sigma,
    msm_theoretical_var_log_sigma,
    solve_m0,
)
from simchart.pipeline import rng_diffusion_check
from simchart.validation.ensemble import vol_cross_section
from simchart.validation.suite import standardized_returns

S1 = dict(stage="S1", enable_msm=True, enable_slow_ou=True)


# ---------------------------------------------------------------------------
# 分散予算ユーティリティ
# ---------------------------------------------------------------------------
def test_solve_m0_matches_the_spec_value() -> None:
    """k=10, target=0.125 -> m0 ~ 1.220 (指示書 §6 の数値例)。"""
    assert solve_m0(10, 0.125) == pytest.approx(1.220, abs=0.001)


@pytest.mark.parametrize("k,target", [(10, 0.125), (8, 0.10), (14, 0.2), (1, 0.01)])
def test_solve_m0_round_trips(k: int, target: float) -> None:
    m0 = solve_m0(k, target)
    assert 1.0 < m0 < 2.0
    assert msm_theoretical_var_log_sigma(k, m0) == pytest.approx(target, rel=1e-12)


def test_solve_m0_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        solve_m0(0, 0.1)
    with pytest.raises(ValueError):
        solve_m0(10, 0.0)


# ---------------------------------------------------------------------------
# 凸性補正と正規化 (アンサンブル断面)
# ---------------------------------------------------------------------------
def test_convexity_correction_normalizes_e_sigma2() -> None:
    """E[sigma^2] = sigma_bar^2 (断面 10 万本, 3 標準誤差以内)。"""
    cfg = Config(**S1)
    res = vol_cross_section(cfg, n_paths=100_000)
    assert res["status"] == "ok"
    assert abs(res["e_sigma2_ratio"] - 1.0) < 3.0 * res["e_sigma2_se"] + 1e-9


def test_missing_convexity_correction_is_detected() -> None:
    """補正 -Var(X) を落とすと E[sigma^2] が e^{2Var} 倍に膨らむ (帰無対照)。

    「なんとなくボラが高い」としか見えない典型的バグを、断面検定が数値で
    捕まえられることの確認。
    """
    cfg = Config(**S1)
    rng = np.random.default_rng(0)
    n = 100_000
    x = rng.normal(0.0, math.sqrt(cfg.vol_var_target_slow), n)
    # 正しい合成 (補正あり) と壊した合成 (補正なし)
    good = compose_log_sigma(0.0, 0.0, x, cfg.vol_var_target_slow)
    bad = compose_log_sigma(0.0, 0.0, x, 0.0)
    assert float(np.mean(np.exp(2 * good))) == pytest.approx(1.0, abs=0.02)
    assert float(np.mean(np.exp(2 * bad))) == pytest.approx(
        math.exp(2 * cfg.vol_var_target_slow), rel=0.02
    )


def test_variance_budget_shares_in_cross_section() -> None:
    """断面の分散シェアが配分どおり。

    シェアには分母の違う 2 種類があり、取り違えると 1.4 倍ずれる (一度やった):
    - shares_of_budget: 分母 = 最終予算 0.25 → MSM 50% / OU 20% (§6 の定義)
    - shares_of_current: 分母 = 現在合計 0.175 → MSM 71% / OU 29%
    """
    res = vol_cross_section(Config(**S1), n_paths=100_000)
    assert res["var_log_sigma"] == pytest.approx(0.175, abs=0.005)
    assert res["shares_of_budget"]["msm"] == pytest.approx(0.50, abs=0.02)
    assert res["shares_of_budget"]["slow_ou"] == pytest.approx(0.20, abs=0.01)
    assert res["shares_of_current"]["msm"] == pytest.approx(0.125 / 0.175, abs=0.02)
    assert res["shares_of_current"]["slow_ou"] == pytest.approx(0.050 / 0.175, abs=0.02)
    assert res["budget_used_fraction"] == pytest.approx(0.70, abs=0.02)


def test_overspending_the_budget_is_rejected() -> None:
    """配分合計が最終予算 0.25 を超える設定は構成エラー (指示書 §6)。"""
    with pytest.raises(ValueError, match="予算"):
        Config(stage="S1", enable_msm=True, enable_slow_ou=True,
               vol_var_target_msm=0.22, vol_var_target_slow=0.06)


def test_composition_coefficient_bug_is_detected() -> None:
    """0.5 係数を落とす (log M をそのまま足す) と断面分散が 4 倍になる (帰無対照)。"""
    cfg = Config(**S1)
    rng = np.random.default_rng(1)
    k, m0 = cfg.msm_k, solve_m0(cfg.msm_k, cfg.vol_var_target_msm)
    states = rng.integers(0, 2, size=(100_000, k))
    log_m = np.where(states == 1, math.log(m0), math.log(2 - m0)).sum(axis=1)
    var_correct = float((0.5 * log_m).var())
    var_broken = float(log_m.var())
    assert var_correct == pytest.approx(0.125, abs=0.01)
    assert var_broken == pytest.approx(0.5, abs=0.04)


# ---------------------------------------------------------------------------
# OU の厳密離散化
# ---------------------------------------------------------------------------
def test_ou_autocorrelation_matches_exact_discretization() -> None:
    """X の 1 日ラグ自己相関が e^{-theta} に一致する (厳密遷移の検証)。"""
    cfg = Config(n_days=4000, steps_per_day=39, **S1)
    result = run(cfg)
    sub = result.meta["l2"]["vol_subsample"]
    x = np.asarray(sub["x_slow"])
    t_days = np.asarray(sub["t_days"])
    per_day = int(round(1.0 / (t_days[1] - t_days[0])))
    x0 = x[:-per_day]
    x1 = x[per_day:]
    rho = float(np.corrcoef(x0, x1)[0, 1])
    theta = math.log(2) / cfg.ou_half_life_days
    assert rho == pytest.approx(math.exp(-theta), abs=0.03)


def test_ou_stationary_variance_across_seeds() -> None:
    """OU の実現分散の平均が目標 Var に一致する (10 シード)。"""
    samples = [
        run(Config(seed=s, n_days=600, steps_per_day=39, **S1)).meta["l2"]["slow_ou"][
            "sample_var"
        ]
        for s in range(10)
    ]
    assert float(np.mean(samples)) == pytest.approx(0.050, abs=0.012)


# ---------------------------------------------------------------------------
# RNG の分離と再現性
# ---------------------------------------------------------------------------
def test_diffusion_stream_is_untouched_by_s1() -> None:
    """S1 のフラグを立てても l2.diffusion の消費列が S0 相当と一致する。

    S0 の RNG 設計 (名前ハッシュ) の最初の実地テスト (指示書 §2)。
    """
    cfg = Config(n_days=20, steps_per_day=390, **S1)
    result = run(cfg)
    report = rng_diffusion_check(cfg, result)
    assert report["match"], report

    s0 = run(Config(n_days=20, steps_per_day=390))
    assert s0.meta["l2"]["diffusion_digest"] == result.meta["l2"]["diffusion_digest"]


def test_s1_price_path_differs_from_s0_only_through_vol() -> None:
    """同一シードの S0 と S1 で、標準化革新が同一であること。

    z が同じで sigma だけが違う、という分解の直接検証。S1 の増分を実現 sigma で
    割り戻すと S0 の増分を sigma_bar で割り戻したものと一致する。
    """
    cfg1 = Config(n_days=10, steps_per_day=390, **S1)
    cfg0 = Config(n_days=10, steps_per_day=390)
    r1 = run(cfg1)
    r0 = run(cfg0)

    z1 = standardized_returns(r1, 60.0)
    z0 = standardized_returns(r0, 60.0)
    np.testing.assert_allclose(z1, z0, atol=1e-10)


def test_determinism_with_s1_enabled() -> None:
    cfg = Config(n_days=30, steps_per_day=390, **S1)
    assert run(cfg).digest() == run(cfg).digest()


def test_msm_only_and_ou_only_compose() -> None:
    """片方だけ有効でも動き、log_vol の分散が対応する配分に近いこと。"""
    msm_only = run(Config(n_days=3000, steps_per_day=39, stage="S1", enable_msm=True))
    ou_only = run(Config(n_days=3000, steps_per_day=39, stage="S1", enable_slow_ou=True))
    # 経路分散はゆらぐ (遅い成分) ので大づかみの範囲で確認する。
    assert 0.04 <= float(msm_only.price.log_vol.var()) <= 0.30
    assert 0.01 <= float(ou_only.price.log_vol.var()) <= 0.15
    # OU only の平均は -Var(X) の補正込みで log(sigma_bar) - 0.05 近傍。
    assert float(ou_only.price.log_vol.mean()) == pytest.approx(
        math.log(0.20) - 0.05, abs=0.1
    )


# ---------------------------------------------------------------------------
# 標準化リターン
# ---------------------------------------------------------------------------
def test_standardized_returns_are_iid_normal_under_s1() -> None:
    """r_std が N(0,1) に従う (std ~ 1, 尖度 ~ 3)。

    確率ボラの「効果を取り除く」変換が正しいことの検証。これが成り立つから
    acf_r / ljung_box の不変ゲートを標準化系列で判定できる。
    """
    result = run(Config(seed=5, n_days=400, steps_per_day=390, **S1))
    z = standardized_returns(result, 60.0)
    assert float(z.std(ddof=1)) == pytest.approx(1.0, abs=0.01)
    assert float(stats.kurtosis(z, fisher=False, bias=False)) == pytest.approx(3.0, abs=0.1)
    # 非標準化リターンの尖度は 3 より明確に大きい (ボラ混合の効果)。
    raw = result.observation.to_bars(60.0).returns()
    raw_kurt = float(stats.kurtosis(raw, fisher=False, bias=False))
    assert raw_kurt > 3.5


# ---------------------------------------------------------------------------
# フラグと従属パラメータのガード
# ---------------------------------------------------------------------------
def test_params_without_flag_are_rejected() -> None:
    with pytest.raises(ValueError, match="enable_msm"):
        Config(msm_k=12)
    with pytest.raises(ValueError, match="enable_slow_ou"):
        Config(ou_half_life_days=40.0)


def test_zero_budget_with_flag_is_rejected() -> None:
    with pytest.raises(ValueError):
        Config(stage="S1", enable_msm=True, vol_var_target_msm=0.0)
    with pytest.raises(ValueError):
        Config(stage="S1", enable_slow_ou=True, vol_var_target_slow=0.0)


def test_s1_stage_is_accepted_now() -> None:
    cfg = Config(stage="S1", enable_msm=True, enable_slow_ou=True)
    assert cfg.stage == "S1"


def test_log_vol_is_constant_in_s0_and_varying_in_s1() -> None:
    s0 = run(Config(n_days=5, steps_per_day=390))
    s1 = run(Config(n_days=5, steps_per_day=390, **S1))
    assert np.all(s0.price.log_vol == s0.price.log_vol[0])
    assert not np.all(s1.price.log_vol == s1.price.log_vol[0])
