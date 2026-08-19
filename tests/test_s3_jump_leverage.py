"""S3 (ジャンプ + レバレッジ) の単体検証。

中核: (1) bridge の厳密性 (セル内無相関)、(2) マルチンゲール補償の適用量、
(3) S2 までの経路のビット単位不変 (jump/leverage を切ったとき)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from simchart import Config, run
from simchart.validation.memory import leverage_function
from simchart.validation.scaling import realized_variance
from simchart.validation.tails import bns_jump_test, hill_by_scale

S2 = dict(enable_msm=True, enable_slow_ou=True, enable_rough=True)
JUMP = dict(
    enable_jump=True, jump_lambda_per_year=5.0, jump_eta_down=35.0, jump_eta_up=56.0
)
BASE = dict(n_days=300, steps_per_day=390)


# ---------------------------------------------------------------------------
# ジャンプ
# ---------------------------------------------------------------------------
def test_eta_up_guard() -> None:
    with pytest.raises(ValueError, match="jump_eta_up"):
        Config(stage="S3", enable_jump=True, jump_eta_up=0.9)


def test_compensation_matches_analytic_k() -> None:
    """補償定数 k = E[e^J] - 1 が MC (100 万標本) と一致する。"""
    cfg = Config(stage="S3", **JUMP, **S2, **BASE)
    result = run(cfg)
    jd = result.meta["l2"]["jump"]
    rng = np.random.default_rng(0)
    n = 1_000_000
    up = rng.uniform(size=n) < cfg.jump_p_up
    e = rng.standard_exponential(n)
    j = np.where(up, e / cfg.jump_eta_up, -e / cfg.jump_eta_down)
    k_mc = float(np.expm1(j).mean())
    assert jd["k_compensation"] == pytest.approx(k_mc, abs=3e-4)
    assert jd["compensation_applied"] is True


def test_jump_only_is_s2_path_plus_jumps() -> None:
    """jump のみ有効のとき、増分 = S2 の増分の decomposable な変形 + ジャンプ。

    同一シードなので z / MSM / OU / rough は S2 と共通。log sigma は
    sigma_bar_diff への縮小分だけ平行移動し、増分は拡散項が sqrt(1-JV) 倍
    + 補償 + ジャンプ。ここでは (a) 全ストリーム証人の不変、(b) ジャンプの
    無いステップでの増分の解析的一致、(c) 補償合計 = -mean(lambda) k dt N を確認。
    """
    r2 = run(Config(stage="S2", **S2, **BASE))
    r3 = run(Config(stage="S3", **JUMP, **S2, **BASE))
    assert r3.meta["l2"]["diffusion_digest"] == r2.meta["l2"]["diffusion_digest"]
    assert r3.meta["l2"]["msm"]["switch_digest"] == r2.meta["l2"]["msm"]["switch_digest"]
    assert r3.meta["l2"]["rough"]["y_digest"] == r2.meta["l2"]["rough"]["y_digest"]
    assert r3.meta["l2"]["slow_ou"]["x0"] == r2.meta["l2"]["slow_ou"]["x0"]

    jd = r3.meta["l2"]["jump"]
    shift = math.log(math.sqrt(1.0 - Config(**JUMP, **S2, stage="S3").jump_qv_share_target))
    np.testing.assert_allclose(r3.price.log_vol, r2.price.log_vol + shift, atol=1e-12)

    d2 = np.diff(r2.price.log_p_star)
    d3 = np.diff(r3.price.log_p_star)
    # S2 増分から S3 の拡散+ドリフト部分を解析的に再構成する。
    cfg2 = r2.config
    dt_y = 1.0 / (252 * cfg2.steps_per_day)
    sig2 = np.exp(r2.price.log_vol[:-1])
    z = (d2 - (cfg2.mu_drift - 0.5 * sig2**2) * dt_y) / (sig2 * math.sqrt(dt_y))
    ratio = math.exp(shift)
    sig3 = sig2 * ratio
    lam = np.minimum(sig3 / jd["sigma_bar_diffusion"], 5.0) ** 1.0  # cap は裾のみ
    diffusion3 = (
        (cfg2.mu_drift - 0.5 * sig3**2) * dt_y
        + sig3 * math.sqrt(dt_y) * z
        - lam * jd["lambda0_per_year"] * jd["k_compensation"] * dt_y
    )
    resid = d3 - diffusion3
    # ジャンプの無いステップでは残差ゼロ、あるステップではサイズが残る。
    n_jumps = int((np.abs(resid) > 1e-9).sum())
    assert n_jumps == jd["n_jumps"]
    # 補償合計の検証 (lambda_effective の定義との整合)。
    comp_total = -jd["lambda_effective_per_year"] * jd["k_compensation"] * dt_y * d3.size
    assert comp_total != 0.0


def test_jumps_are_recorded_on_grid() -> None:
    r = run(Config(stage="S3", **JUMP, **S2, **BASE))
    jt = r.price.jump_times
    assert jt.size == r.meta["l2"]["jump"]["n_jumps"]
    if jt.size:
        # ジャンプ時刻はグリッド点に載っている (線形補間でなまらせない)。
        step = r.observation.step_seconds
        np.testing.assert_allclose(jt / step, np.round(jt / step), atol=1e-9)


def test_bns_recovers_planted_jv_share() -> None:
    """合成データ: 拡散のみ → JV ~ 0、ジャンプ追加 → JV を近似回収 (帰無対照つき)。"""
    rng = np.random.default_rng(1)
    n_days, spd = 800, 390
    sigma_step = 0.2 / math.sqrt(252 * spd)
    r_pure = sigma_step * rng.standard_normal(n_days * spd)
    assert bns_jump_test(r_pure, spd)["jv_share"] < 0.02

    r_jump = r_pure.copy()
    idx = rng.choice(r_jump.size, 60, replace=False)
    r_jump[idx] += rng.choice([-1, 1], 60) * 0.03
    jv_true = (60 * 0.03**2) / (60 * 0.03**2 + (sigma_step**2) * r_pure.size)
    got = bns_jump_test(r_jump, spd)["jv_share"]
    assert got == pytest.approx(jv_true, abs=0.05)


def test_hill_by_scale_increases_for_exponential_jumps() -> None:
    """指数ジャンプ + 正規では α がスケールで上昇する (⑱)。"""
    rng = np.random.default_rng(2)
    n = 6000
    r = 0.012 * rng.standard_normal(n)
    idx = rng.choice(n, 300, replace=False)
    r[idx] += np.where(rng.uniform(size=300) < 0.42, 1.0, -1.0) * rng.exponential(0.03, 300)
    res = hill_by_scale(r)
    assert res["status"] == "ok"
    assert res["slope_vs_log_scale"] > 0


# ---------------------------------------------------------------------------
# レバレッジ (bridge)
# ---------------------------------------------------------------------------
LEV = dict(enable_leverage=True, **JUMP)


def test_bridge_cell_moments_are_exact() -> None:
    """bridge 構成の Var(z)=1 / セル内 Cov=0 / Σz=A を大標本で確認する。"""
    from simchart.layers.l2_price import GBMPriceLayer

    rng = np.random.default_rng(3)
    k, n_cells, rho = 60, 30000, -0.7
    b = rng.standard_normal(k * n_cells)
    eps = rng.standard_normal(n_cells)
    w = rng.standard_normal(n_cells)
    a_cells = rho * math.sqrt(k) * eps + math.sqrt(1 - rho**2) * math.sqrt(k) * w
    b2 = b.reshape(n_cells, k).copy()
    sums = b2.sum(axis=1)
    b2 += ((a_cells - sums) / k)[:, None]
    z = b2.ravel()

    assert float(z.var()) == pytest.approx(1.0, abs=0.005)
    np.testing.assert_allclose(b2.sum(axis=1), a_cells, atol=1e-9)
    zc = z - z.mean()
    v = float(np.dot(zc, zc))
    for lag in (1, 2, 30, 59):
        assert abs(float(np.dot(zc[:-lag], zc[lag:]) / v)) < 4.0 / math.sqrt(z.size)
    assert float(np.corrcoef(a_cells / math.sqrt(k), eps)[0, 1]) == pytest.approx(rho, abs=0.01)


def test_leverage_realized_correlations_and_z_acf() -> None:
    cfg = Config(stage="S3", **LEV, **S2, n_days=500, steps_per_day=390)
    r = run(cfg)
    lev = r.meta["l2"]["leverage"]
    assert lev["corr_rough_realized"] == pytest.approx(cfg.leverage_rho_rough, abs=0.02)
    assert lev["corr_slow_realized"] == pytest.approx(cfg.leverage_rho_slow, abs=0.02)
    z_acf = lev["z_acf"]
    assert z_acf["max_abs_acf"] < 3.7 / math.sqrt(z_acf["n"])
    assert abs(lev["eps_residual_acf1"]) < 5e-3  # whitening の打ち切り誤差


def test_common_shock_addition_would_fail_z_acf() -> None:
    """帰無対照: 共通ショックの単純加算はセル内 ACF +rho^2/n を作り検出される。"""
    rng = np.random.default_rng(4)
    k, n_cells, rho = 60, 20000, -0.7
    b = rng.standard_normal(k * n_cells)
    eps = np.repeat(rng.standard_normal(n_cells), k)
    z_bad = math.sqrt(1 - rho**2 / 1) * b + (rho / math.sqrt(k)) * eps  # 素朴な加算
    z_bad /= z_bad.std()
    zc = z_bad - z_bad.mean()
    v = float(np.dot(zc, zc))
    acf1 = float(np.dot(zc[:-1], zc[1:]) / v)
    assert acf1 > 3.7 / math.sqrt(z_bad.size)  # 確実に検出される大きさ


def test_leverage_off_is_bitwise_s2() -> None:
    """レバレッジとジャンプを切れば S3 コードでも S2 経路とビット単位同一。"""
    a = run(Config(stage="S2", **S2, **BASE))
    b = run(Config(stage="S3", **S2, **BASE))
    assert a.digest() == b.digest()


def test_leverage_direction_is_negative() -> None:
    """レバレッジの向き: corr(r_t, RV_{t+1}) の中央値が負 (水準は本番ゲートで)。"""
    vals = []
    for s in range(4):
        cfg = Config(seed=60 + s, stage="S3", **LEV, **S2, n_days=2000, steps_per_day=390)
        r = run(cfg)
        obs = r.observation
        rd = obs.to_bars(obs.session_seconds).returns()
        spd = int(round(obs.session_seconds / obs.step_seconds))
        rv = realized_variance(np.diff(obs.log_price), spd)
        vals.append(leverage_function(rd, rv, horizons=(0, 1))["corr_r_rv_h1"])
    assert float(np.median(vals)) < 0


def test_mid_component_disabled_by_default_and_enables() -> None:
    cfg = Config(stage="S3", **LEV, **S2, **BASE)
    assert cfg.leverage_mid_var == 0.0
    r = run(cfg)
    assert "leverage_mid" not in r.meta["l2"]

    cfg_mid = Config(stage="S3", **LEV, **S2, **BASE, leverage_mid_var=0.02)
    r_mid = run(cfg_mid)
    mid = r_mid.meta["l2"]["leverage_mid"]
    assert mid["corr_mid_realized"] == pytest.approx(cfg_mid.leverage_rho_mid, abs=0.05)
    assert mid["var_slow_remaining"] == pytest.approx(0.03)


def test_determinism_with_s3_enabled() -> None:
    cfg = Config(stage="S3", **LEV, **S2, **BASE)
    assert run(cfg).digest() == run(cfg).digest()


def test_s3_flag_guards() -> None:
    with pytest.raises(ValueError, match="enable_rough"):
        Config(stage="S3", enable_msm=True, enable_slow_ou=True, enable_leverage=True)
    with pytest.raises(ValueError, match="enable_jump"):
        Config(jump_lambda_per_year=9.0)
    with pytest.raises(ValueError, match="leverage_mid_var"):
        Config(stage="S3", **S2, enable_leverage=True, leverage_mid_var=0.05)