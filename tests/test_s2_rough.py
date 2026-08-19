"""S2 のラフボラティリティ成分 (Davies-Harte fGn + fOU) の単体検証。

S2 の合否は「何が変わらなかったか」で決まるため、S1 経路のビット単位不変も
ここで固定する。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from simchart import Config, run
from simchart.layers.l2_price import (
    compose_log_sigma,
    davies_harte_fgn,
    rough_discrete_stationary_variance,
    solve_eta_rough,
)
from simchart.validation.scaling import path_stationarity, roughness_exponent

S2 = dict(stage="S2", enable_msm=True, enable_slow_ou=True, enable_rough=True)
S1 = dict(stage="S1", enable_msm=True, enable_slow_ou=True)


# ---------------------------------------------------------------------------
# Davies-Harte fGn
# ---------------------------------------------------------------------------
def test_dh_autocovariance_matches_theory() -> None:
    """生成した fGn の標本自己共分散が理論値と一致する (厳密生成の検証)。"""
    hurst = 0.10
    n = 2**17
    g = davies_harte_fgn(n, hurst, np.random.default_rng(0))
    assert float(g.var()) == pytest.approx(1.0, abs=0.02)
    for lag in (1, 2, 5):
        emp = float(np.mean(g[:-lag] * g[lag:]))
        theo = 0.5 * ((lag + 1) ** 0.2 - 2 * lag**0.2 + (lag - 1) ** 0.2)
        assert emp == pytest.approx(theo, abs=0.01), lag


def test_dh_aggregation_scaling() -> None:
    """fBm の自己相似性: Var(k 個の和) = k^{2H}。"""
    hurst = 0.10
    g = davies_harte_fgn(2**17, hurst, np.random.default_rng(1))
    for k in (4, 16, 64):
        agg = g[: (g.size // k) * k].reshape(-1, k).sum(axis=1)
        assert float(agg.var()) == pytest.approx(k ** (2 * hurst), rel=0.06), k


def test_dh_deterministic_and_seed_dependent() -> None:
    a = davies_harte_fgn(1000, 0.1, np.random.default_rng(7))
    b = davies_harte_fgn(1000, 0.1, np.random.default_rng(7))
    c = davies_harte_fgn(1000, 0.1, np.random.default_rng(8))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# 分散予算まわり
# ---------------------------------------------------------------------------
def test_solve_eta_rough_round_trips() -> None:
    hurst, theta = 0.08, math.log(2) / 0.75
    eta = solve_eta_rough(hurst, theta, 0.025)
    var_cont = eta**2 * math.gamma(2 * hurst + 1) / (2 * theta ** (2 * hurst))
    assert var_cont == pytest.approx(0.025, rel=1e-12)


def test_discrete_variance_close_to_continuous() -> None:
    """離散定常分散が連続式と ~1% で一致する (刻み 60 秒, HL 0.75 日)。

    差が小さいことを確認しつつ、凸性補正には離散値を使う (l2_price 参照)。
    """
    hurst, theta = 0.08, math.log(2) / 0.75
    dt = 60.0 / 23400.0
    eta = solve_eta_rough(hurst, theta, 0.025)
    var_disc = rough_discrete_stationary_variance(hurst, theta, dt, eta)
    assert var_disc == pytest.approx(0.025, rel=0.02)


def test_budget_guard_includes_rough() -> None:
    with pytest.raises(ValueError, match="予算"):
        Config(stage="S2", enable_msm=True, enable_slow_ou=True, enable_rough=True,
               vol_var_target_rough=0.09)


def test_rough_param_guards() -> None:
    with pytest.raises(ValueError, match="1 日以下"):
        Config(stage="S2", enable_rough=True, rough_half_life_days=1.5)
    with pytest.raises(ValueError, match="rough_hurst"):
        Config(stage="S2", enable_rough=True, rough_hurst=0.6)
    with pytest.raises(ValueError, match="enable_rough"):
        Config(rough_hurst=0.12)  # フラグ off でパラメータ変更


# ---------------------------------------------------------------------------
# fOU の性質
# ---------------------------------------------------------------------------
def test_fou_increment_anticorrelation() -> None:
    """増分の 1 ラグ自己相関が fGn の理論値 2^{2H-1} - 1 に一致する (反持続性)。"""
    cfg = Config(n_days=2000, steps_per_day=390, **S2)
    result = run(cfg)
    y = np.asarray(result.meta["l2"]["vol_subsample"]["y_rough"])
    dy = np.diff(y)
    rho1 = float(np.corrcoef(dy[:-1], dy[1:])[0, 1])
    assert rho1 == pytest.approx(2 ** (2 * cfg.rough_hurst - 1) - 1, abs=0.02)
    assert rho1 < 0


def test_fou_path_variance_matches_discrete_theory() -> None:
    """経路の標本分散が離散定常分散に一致する (半減期が短いので良く推定できる)。"""
    samples = []
    for seed in range(5):
        r = run(Config(seed=seed, n_days=2000, steps_per_day=39, **S2))
        d = r.meta["l2"]["rough"]
        samples.append(d["sample_var"] / d["var_discrete"])
    assert float(np.mean(samples)) == pytest.approx(1.0, abs=0.03)


def test_fou_stationarity() -> None:
    result = run(Config(seed=3, n_days=3000, steps_per_day=39, **S2))
    y = np.asarray(result.meta["l2"]["vol_subsample"]["y_rough"])
    report = path_stationarity(y)
    assert report["stationary"] is True, report["checks"]


def test_nonstationary_path_is_detected() -> None:
    """帰無対照: 分散が t^{2H} で増大する非定常過程 (fBm) は stationarity が落ちる。

    指示書 §10-1 の「最頻の事故」を検査が本当に捕まえられることの確認。
    """
    g = davies_harte_fgn(200_000, 0.1, np.random.default_rng(11))
    fbm = np.cumsum(g)  # 非定常 (Var ~ t^{2H})
    report = path_stationarity(fbm)
    assert report["stationary"] is False


# ---------------------------------------------------------------------------
# H 推定器
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hurst", [0.10, 0.30])
def test_roughness_exponent_recovers_planted_h(hurst: float) -> None:
    """合成 fBm から既知の H を回収する (推定器自体の検証)。"""
    g = davies_harte_fgn(2**19, hurst, np.random.default_rng(2))
    fbm = np.cumsum(g)
    result = roughness_exponent(fbm, scales_steps=(5, 10, 20, 30, 60, 120, 240))
    assert result["status"] == "ok"
    assert result["h"] == pytest.approx(hurst, abs=0.02)
    assert result["linearity_r2"] > 0.99


def test_roughness_exponent_handles_constant_and_empty() -> None:
    assert roughness_exponent(np.zeros(100_000), (5, 10, 20))["status"] == "not_applicable"
    assert roughness_exponent(np.zeros(0), ())["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# S1 経路の不変性 (S2 の本質)
# ---------------------------------------------------------------------------
def test_s1_streams_are_bitwise_unchanged_by_s2() -> None:
    """ラフ成分を有効にしても S1 のストリーム消費が 1 draw も変わらないこと。"""
    base = dict(seed=9, n_days=100, steps_per_day=390)
    r1 = run(Config(**S1, **base))
    r2 = run(Config(**S2, **base))
    assert r1.meta["l2"]["diffusion_digest"] == r2.meta["l2"]["diffusion_digest"]
    assert r1.meta["l2"]["msm"]["switch_digest"] == r2.meta["l2"]["msm"]["switch_digest"]
    assert r1.meta["l2"]["slow_ou"]["x0"] == r2.meta["l2"]["slow_ou"]["x0"]
    assert r1.meta["l2"]["slow_ou"]["sample_var"] == r2.meta["l2"]["slow_ou"]["sample_var"]
    # log sigma の差はちょうどラフ成分 (+凸性補正) だけ。
    diff = r2.price.log_vol - r1.price.log_vol
    var_rough = r2.meta["l2"]["rough"]["var_discrete"]
    y0 = diff[0] + var_rough
    assert abs(float(diff.mean()) + var_rough - 0.0) < 0.2  # 平均 ~ E[Y]=0 − Var
    assert float(np.abs(diff + var_rough).max()) > 0.01  # ラフ成分が実在する
    del y0


def test_rough_path_is_resolution_independent() -> None:
    """ラフ経路が steps_per_day に依存しない (専用物理グリッドの検証)。"""
    a = run(Config(seed=5, n_days=200, steps_per_day=390, **S2))
    b = run(Config(seed=5, n_days=200, steps_per_day=1950, **S2))
    assert a.meta["l2"]["rough"]["y_digest"] == b.meta["l2"]["rough"]["y_digest"]


def test_compose_with_rough_inplace_matches_bitwise() -> None:
    rng = np.random.default_rng(4)
    msm = rng.normal(0, 0.3, 50_000)
    slow = rng.normal(0, 0.2, 50_000)
    rough = rng.normal(0, 0.15, 50_000)
    expected = compose_log_sigma(math.log(0.2), msm.copy(), slow, 0.05, rough, 0.025)
    got = compose_log_sigma(math.log(0.2), msm.copy(), slow, 0.05, rough, 0.025, inplace=True)
    np.testing.assert_array_equal(expected, got)


def test_e_sigma2_normalization_with_rough() -> None:
    from simchart.validation.ensemble import vol_cross_section

    res = vol_cross_section(Config(**S2), n_paths=100_000)
    assert res["status"] == "ok"
    assert abs(res["e_sigma2_ratio"] - 1.0) < 3.0 * res["e_sigma2_se"] + 1e-9
    assert res["shares_of_budget"]["rough"] == pytest.approx(0.10, abs=0.01)
    assert res["var_log_sigma"] == pytest.approx(0.200, abs=0.005)


def test_determinism_with_s2_enabled() -> None:
    cfg = Config(n_days=50, steps_per_day=390, **S2)
    assert run(cfg).digest() == run(cfg).digest()
