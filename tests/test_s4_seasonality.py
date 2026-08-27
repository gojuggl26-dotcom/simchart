"""S4: 日内季節性とオーバーナイトのテスト。

このファイルの中心は `test_deseasonalization_recovers_s3_exactly` である。
S4 の設計 (季節性を確率ボラへの決定論的な乗法変調に限定したこと) が正しければ、
φ で割ると S3 の系列が厳密に戻るはずで、それが S4 の主張の全根拠になる。
ジャンプを切って検定するのは、λ(t) がボラ変調されているため S4 では S3 と別の
ジャンプ抽選になり、その差 (実測で差分分散の 100%) が復元誤差を覆い隠すからである。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simchart import Config, run
from simchart.layers.l0_calendar import (
    ConstantCalendar,
    SeasonalCalendar,
    build_calendar,
    fourier_profile,
    normalize_phi_lambda,
    normalize_phi_sigma,
)
from simchart.rng import RNGRegistry
from simchart.validation import seasonality as sz
from simchart.validation.memory import gph_estimator

# 全テスト共通の小さめ設定 (200 日 x 390 ステップ = 78,000 点)。
SMALL = dict(n_days=200, steps_per_day=390)
CORE = dict(enable_msm=True, enable_slow_ou=True, enable_rough=True)
JUMP = dict(enable_jump=True, jump_lambda_per_year=5.0)


def _cal(cfg: Config):
    return build_calendar(cfg, RNGRegistry(cfg.seed))


# ---------------------------------------------------------------------------
# φ の表現と正規化
# ---------------------------------------------------------------------------
def test_phi_sigma_normalizes_the_square_not_the_level():
    """φ_σ は二乗平均が 1。一乗平均は Jensen により 1 より小さい。"""
    cfg = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    cal = _cal(cfg)
    u = np.linspace(0.0, 1.0, 20001)
    phi = cal.phi_sigma_of_u(u)
    assert np.trapezoid(phi**2, u) == pytest.approx(1.0, abs=1e-6)
    # 起伏があれば一乗平均は必ず 1 未満 (E[X]^2 < E[X^2])。
    assert np.trapezoid(phi, u) < 1.0 - 1e-3


def test_phi_lambda_normalizes_the_level_not_the_square():
    """φ_λ は一乗平均が 1 (強度なので)。二乗平均は 1 より大きい。"""
    cfg = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    cal = _cal(cfg)
    u = np.linspace(0.0, 1.0, 20001)
    lam = cal.phi_lambda_of_u(u)
    assert np.trapezoid(lam, u) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(lam**2, u) > 1.0 + 1e-3


def test_linear_slope_is_required_for_open_to_differ_from_close():
    """周期 Fourier だけでは φ(0) == φ(1) になる — 傾き項が要る根拠。"""
    u = np.array([0.0, 1.0])
    periodic = fourier_profile(u, (0.5, 0.2, 0.0), (0.3, -0.1, 0.0), slope=0.0)
    assert periodic[0] == pytest.approx(periodic[1])
    tilted = fourier_profile(u, (0.5, 0.2, 0.0), (0.3, -0.1, 0.0), slope=-0.3)
    assert tilted[0] > tilted[1]


def test_phi_shape_matches_the_stylised_facts():
    """ボラは寄付最大・引けは中位、出来高は引け最大。最小はどちらも日中。"""
    cfg = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    check = sz.phi_normalization_check(_cal(cfg))
    assert check["status"] == "ok"
    assert check["phi_sigma_open_gt_close"]
    assert check["phi_lambda_close_gt_open"]
    assert check["phi_sigma_min_interior"]
    assert check["phi_lambda_min_interior"]
    assert 3.0 <= check["phi_sigma_sq_max_min_ratio"] <= 6.0
    assert 4.0 <= check["phi_lambda_max_min_ratio"] <= 10.0


def test_constant_calendar_has_no_seasonality():
    """S0〜S3 の L0 は φ ≡ 1 のまま (S4 のコードが漏れ出していないこと)。"""
    cfg = Config(stage="S3", **CORE, **SMALL)
    cal = _cal(cfg)
    assert isinstance(cal, ConstantCalendar)
    assert not isinstance(cal, SeasonalCalendar)
    t = np.linspace(0.0, 23400.0, 101)
    assert np.allclose(cal.phi_sigma(t), 1.0)
    assert np.allclose(cal.phi_lambda(t), 1.0)


def test_extreme_coefficients_stay_positive_and_normalizable():
    """クリップ下限 0.05 が効くほど強い係数でも φ が正値で正規化できる。

    負の φ ができると ``log`` や ``sqrt`` が静かに NaN を返し、ボラ経路が壊れた
    ことに気づけない。境界は例外ではなくクリップで守る設計なので、ここでは
    クリップが機能していることを固定する。
    """
    # 正規化定数は 20001 点の台形則で作られる。クリップで折れ点ができるため
    # 粗い格子で測ると台形則の誤差が残る (1e-6 では落ちる)。同じ密度で測る。
    u = np.linspace(0.0, 1.0, 20001)
    g = fourier_profile(u, (5.0, 0.0, 0.0), (0.0, 0.0, 0.0), slope=0.0)
    assert g.min() >= 0.05
    assert np.isfinite(np.log(g)).all()

    c_s = normalize_phi_sigma((5.0,), (0.0,))
    c_l = normalize_phi_lambda((5.0,), (0.0,))
    assert c_s > 0.0 and c_l > 0.0
    assert np.trapezoid((c_s * g) ** 2, u) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(c_l * g, u) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 中心: 脱季節化の厳密性
# ---------------------------------------------------------------------------
def test_deseasonalization_recovers_s3_exactly():
    """φ で割ると S3 の系列がバー単位で厳密に戻る (ジャンプ・ON 無効)。

    残差は Itô のドリフト項 ``-0.5 sigma^2 dt`` だけである。これは φ の二乗で
    入るので φ の一乗で割っても戻らない (構造的で、除去の欠陥ではない)。その大きさは
    リターンの標準偏差比で ~1e-4 なので、閾値 1e-3 は「ドリフト以外の何かが
    混ざっていれば落ちる」水準になっている。
    """
    kw = dict(enable_leverage=True, leverage_rho_rough=-0.6, leverage_rho_slow=-0.35)
    c4 = Config(stage="S4", enable_seasonality=True, **CORE, **kw, **SMALL)
    c3 = Config(stage="S3", **CORE, **kw, **SMALL)
    r4, r3 = run(c4), run(c3)

    step = r4.observation.step_seconds
    b4 = r4.observation.to_bars(step).returns_2d()
    b3 = r3.observation.to_bars(step).returns_2d()
    truth = sz.true_phi_bars(_cal(c4), b4.shape[1], steps_per_day=SMALL["steps_per_day"])
    assert truth["status"] == "ok"
    assert truth["mode"] == "generator_discrete"

    diff = sz.deseasonalize(b4, np.asarray(truth["value"])) - b3
    assert diff.std() / b3.std() < 1e-3

    # 同じ主張を統計量でも: 脱季節化後の GPH d は S3 と 4 桁で一致する。
    bwe = c4.validation.gph_bandwidth_exponent
    d4 = gph_estimator(np.abs(sz.deseasonalize(b4, np.asarray(truth["value"]))).ravel(), bwe)
    d3 = gph_estimator(np.abs(b3).ravel(), bwe)
    assert d4["value"] == pytest.approx(d3["value"], abs=1e-4)


def test_continuous_quadrature_phi_is_worse_than_generator_discrete():
    """生成側の離散化に合わせた φ のほうが厳密 (合わせない実装への回帰防止)。

    Euler-Maruyama は各ステップの左端の σ を使うので、連続時間の積分で φ を
    作ると O(φ'/steps_per_day) の系統差が残る。実測で 18 倍の差がつく。
    """
    c4 = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    c3 = Config(stage="S3", **CORE, **SMALL)
    r4, r3 = run(c4), run(c3)
    step = r4.observation.step_seconds
    b4 = r4.observation.to_bars(step).returns_2d()
    b3 = r3.observation.to_bars(step).returns_2d()
    cal = _cal(c4)

    disc = np.asarray(
        sz.true_phi_bars(cal, b4.shape[1], steps_per_day=SMALL["steps_per_day"])["value"]
    )
    cont = np.asarray(sz.true_phi_bars(cal, b4.shape[1])["value"])
    e_disc = (sz.deseasonalize(b4, disc) - b3).std()
    e_cont = (sz.deseasonalize(b4, cont) - b3).std()
    assert e_disc < e_cont / 5.0


def test_deseasonalize_rejects_mismatched_phi():
    r = np.ones((5, 4))
    with pytest.raises(ValueError):
        sz.deseasonalize(r, np.ones(3))
    with pytest.raises(ValueError):
        sz.deseasonalize(r, np.array([1.0, 0.0, 1.0, 1.0]))


def test_deseasonalize_preserves_overall_scale():
    """φ の二乗平均が 1 なので、除去しても全体の分散水準は動かない。"""
    rng = np.random.default_rng(0)
    r = rng.standard_normal((300, 26))
    phi = np.linspace(0.6, 1.4, 26)
    phi /= math.sqrt((phi**2).mean())
    scaled = r * phi[None, :]
    assert sz.deseasonalize(scaled, phi).std() == pytest.approx(r.std(), rel=1e-12)


# ---------------------------------------------------------------------------
# 推定経路
# ---------------------------------------------------------------------------
def test_estimate_phi_recovers_a_known_profile():
    """真値を知らない推定器が、既知の φ を注入した合成データで φ を当てる。"""
    rng = np.random.default_rng(7)
    n_days, n_bars = 600, 26
    u = sz.bin_centers(n_bars)
    phi = 1.0 + 0.45 * np.cos(2 * math.pi * u) - 0.25 * (u - 0.5)
    phi /= math.sqrt((phi**2).mean())
    r = rng.standard_normal((n_days, n_bars)) * phi[None, :]

    est = sz.estimate_phi(r, n_harmonics=3)
    assert est["status"] == "ok"
    rec = sz.phi_recovery(np.asarray(est["value"]), phi)
    assert rec["correlation"] > 0.99
    assert rec["max_abs_rel_error"] < 0.10


def test_estimate_phi_is_robust_to_jumps():
    """中央値ベースの既定は少数の巨大リターンで φ̂ が跳ねない。

    ``rms`` は同じデータでスパイクを拾うので、両者を比べて既定の優位を固定する。
    """
    rng = np.random.default_rng(11)
    n_days, n_bars = 400, 26
    u = sz.bin_centers(n_bars)
    phi = 1.0 + 0.4 * np.cos(2 * math.pi * u)
    phi /= math.sqrt((phi**2).mean())
    r = rng.standard_normal((n_days, n_bars)) * phi[None, :]
    r[17, 9] += 60.0  # 1 本だけ巨大なジャンプ

    med = sz.phi_recovery(np.asarray(sz.estimate_phi(r, method="median_abs")["value"]), phi)
    rms = sz.phi_recovery(np.asarray(sz.estimate_phi(r, method="rms")["value"]), phi)
    assert med["max_abs_rel_error"] < rms["max_abs_rel_error"]
    assert med["max_abs_rel_error"] < 0.15


def test_estimate_phi_can_be_fitted_out_of_sample():
    """``fit_days`` で訓練期間を限れる (予測用途でのルックアヘッド回避)。"""
    rng = np.random.default_rng(3)
    r = rng.standard_normal((400, 26))
    est = sz.estimate_phi(r, fit_days=slice(0, 200))
    assert est["status"] == "ok"
    assert est["n_days_used"] == 200


def test_intraday_profile_returns_na_when_too_few_days():
    out = sz.intraday_profile(np.ones((5, 26)))
    assert out["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# 除去の効き目を測る道具
# ---------------------------------------------------------------------------
def test_spectral_test_detects_seasonality_and_clears_after_removal():
    cfg = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    result = run(cfg)
    r_2d = result.observation.to_bars(cfg.validation.primary_bar_sec).returns_2d()
    truth = sz.true_phi_bars(_cal(cfg), r_2d.shape[1], steps_per_day=SMALL["steps_per_day"])

    raw = sz.spectral_periodicity_test(r_2d)
    removed = sz.spectral_periodicity_test(sz.deseasonalize(r_2d, np.asarray(truth["value"])))
    assert raw["mean_ratio"] > 20.0
    assert raw["p_value"] < 1e-6
    assert removed["mean_ratio"] < 5.0


def test_spectral_test_is_calibrated_on_a_seasonless_series():
    """帰無対照: 季節性の無い S3 では帰無水準 (比 ~1) に留まる。"""
    result = run(Config(stage="S3", **CORE, **SMALL))
    r_2d = result.observation.to_bars(60).returns_2d()
    out = sz.spectral_periodicity_test(r_2d)
    assert out["status"] == "ok"
    assert out["mean_ratio"] < 5.0


def test_profile_flatness_falls_to_the_noise_floor():
    cfg = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    result = run(cfg)
    r_2d = result.observation.to_bars(cfg.validation.primary_bar_sec).returns_2d()
    truth = sz.true_phi_bars(_cal(cfg), r_2d.shape[1], steps_per_day=SMALL["steps_per_day"])

    raw = sz.profile_flatness(r_2d)
    removed = sz.profile_flatness(sz.deseasonalize(r_2d, np.asarray(truth["value"])))
    assert raw["excess_over_se"] > 2.0
    assert removed["excess_over_se"] < 1.35


def test_coarsen_matches_direct_resampling():
    """細かいバーの和 == 粗いバーの再標本化 (周期検定の前処理の健全性)。"""
    cfg = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    obs = run(cfg).observation
    fine = obs.to_bars(60).returns_2d()
    coarse = obs.to_bars(1800).returns_2d()
    assert np.allclose(sz.coarsen(fine, coarse.shape[1]), coarse, atol=1e-12)


def test_acf_periodicity_has_no_convexity_bias_on_null():
    """帰無 (季節性なし) で excess が 0 付近 (2 次外挿の効果)。"""
    result = run(Config(stage="S3", **CORE, **SMALL))
    r_2d = result.observation.to_bars(1800).returns_2d()
    out = sz.acf_periodicity_test(np.abs(r_2d).ravel(), 13, n_multiples=5)
    assert out["status"] == "ok"
    assert abs(out["z_approx"]) < 3.0


# ---------------------------------------------------------------------------
# オーバーナイト
# ---------------------------------------------------------------------------
def test_overnight_variance_share_hits_the_target():
    cfg = Config(
        stage="S4", enable_seasonality=True, enable_overnight=True, **CORE, **JUMP, **SMALL
    )
    result = run(cfg)
    r_daily = result.observation.to_bars(result.observation.session_seconds).returns()
    out = sz.overnight_stats(result.price.overnight_gaps, r_daily)
    assert out["status"] == "ok"
    assert 0.13 <= out["variance_share"] <= 0.30
    assert out["kurtosis_gap"] > out["kurtosis_intraday_daily"]


def test_overnight_gap_does_not_predict_next_day():
    """帰無対照。ギャップが翌日の日中方向を予測したら実装が未来を漏らしている。"""
    cfg = Config(
        stage="S4", enable_seasonality=True, enable_overnight=True, **CORE, **JUMP, **SMALL
    )
    result = run(cfg)
    r_daily = result.observation.to_bars(result.observation.session_seconds).returns()
    out = sz.overnight_stats(result.price.overnight_gaps, r_daily)
    assert abs(out["corr_gap_next_intraday"]) <= 3.0 * out["corr_gap_next_intraday_se"]


def test_overnight_links_to_closing_volatility():
    cfg = Config(
        stage="S4", enable_seasonality=True, enable_overnight=True, **CORE, **JUMP, **SMALL
    )
    result = run(cfg)
    obs = result.observation
    spd = SMALL["steps_per_day"]
    close_idx = np.arange(1, SMALL["n_days"]) * spd - 1
    sigma_close = np.exp(result.price.log_vol[close_idx])
    r_daily = obs.to_bars(obs.session_seconds).returns()
    out = sz.overnight_stats(result.price.overnight_gaps, r_daily, sigma_close)
    assert out["corr_abs_gap_sigma_close"] > 0.20


def test_close_to_close_composition():
    r = np.array([0.1, 0.2, 0.3])
    g = np.array([0.01, 0.02])
    assert np.allclose(sz.close_to_close_returns(r, g), [0.21, 0.32])
    with pytest.raises(ValueError):
        sz.close_to_close_returns(r, np.array([0.01]))


def test_overnight_requires_jumps():
    """ON はジャンプ機構を借りるので enable_jump なしでは組めない (明示的に失敗)。"""
    from simchart.layers.l2_price import build_price_layer

    cfg = Config(stage="S4", enable_overnight=True, **CORE, **SMALL)
    rng = RNGRegistry(cfg.seed)
    with pytest.raises(ValueError):
        build_price_layer(cfg, rng, build_calendar(cfg, rng), None)


def test_gaps_are_absent_when_overnight_is_off():
    result = run(Config(stage="S4", enable_seasonality=True, **CORE, **SMALL))
    assert result.price.overnight_gaps.size == 0


# ---------------------------------------------------------------------------
# S3 の予算・ストリームが動いていないこと
# ---------------------------------------------------------------------------
def test_s4_does_not_touch_s1_s3_streams():
    """季節性・ON を足しても拡散・MSM・ラフのストリームはビット単位で不変。"""
    kw = dict(**CORE, **JUMP, **SMALL)
    r4 = run(Config(stage="S4", enable_seasonality=True, enable_overnight=True, **kw))
    r3 = run(Config(stage="S3", **kw))
    for path in (("diffusion_digest",), ("msm", "switch_digest"), ("rough", "y_digest")):
        a, b = r4.meta["l2"], r3.meta["l2"]
        for part in path:
            a, b = a[part], b[part]
        assert a == b, f"{path} が変化しました"


def test_jump_qv_share_is_preserved_by_the_intensity_correction():
    """S4 の強度補正が無いと JV シェアが跳ねる (12.7% → 14.9% を実測)。"""
    kw = dict(**CORE, **JUMP, **SMALL)
    r4 = run(Config(stage="S4", enable_seasonality=True, enable_overnight=True, **kw))
    r3 = run(Config(stage="S3", **kw))
    j4, j3 = r4.meta["l2"]["jump"], r3.meta["l2"]["jump"]
    assert j3["intensity_scale_s4"] == 1.0
    assert j4["intensity_scale_s4"] < 1.0
    assert j4["jv_share_theory"] == pytest.approx(j3["jv_share_theory"], abs=0.005)


def test_intensity_cap_binding_is_reported():
    """cap が黙って効いていないことを確認できる診断が出ている。"""
    result = run(Config(stage="S4", enable_seasonality=True, **CORE, **JUMP, **SMALL))
    assert result.meta["l2"]["jump"]["cap_binding_fraction"] < 0.01


def test_sigma_bar_diffusion_carves_out_overnight():
    from simchart.layers.l2_price import build_price_layer

    kw = dict(**CORE, **JUMP, **SMALL)
    cfg = Config(stage="S4", enable_seasonality=True, enable_overnight=True, **kw)
    rng = RNGRegistry(cfg.seed)
    layer = build_price_layer(cfg, rng, build_calendar(cfg, rng), None)
    expected = (
        cfg.sigma_bar
        * math.sqrt(1 - cfg.jump_qv_share_target)
        * math.sqrt(1 - cfg.overnight_variance_share)
    )
    assert layer.sigma_bar_diffusion == pytest.approx(expected)


def test_daily_integrated_variance_is_unchanged_by_seasonality():
    """φ_σ の二乗正規化により日次積分分散が不変 (日次ゲートが動かない根拠)。

    日次リターンの標本分散で比べてはならない。200 日の標本分散はボラの
    長期記憶のせいで少数の高ボラ日に支配され、有効標本数が 20〜30 しかない
    (実際にそれで比べたら 5.8% ずれて落ちた)。σ の経路は S3 と共通なので、
    日ごとに対応づけた実現分散の比を見るのが正しい設計で、共通のゆらぎが
    完全に相殺されて主張そのものを検定できる。
    """
    kw = dict(**CORE, **SMALL)
    r4 = run(Config(stage="S4", enable_seasonality=True, **kw))
    r3 = run(Config(stage="S3", **kw))
    spd = SMALL["steps_per_day"]

    def rv_by_day(result):
        r = np.diff(result.observation.log_price)
        return (r**2).reshape(SMALL["n_days"], spd).sum(axis=1)

    ratio = rv_by_day(r4) / rv_by_day(r3)
    # 平均比 = mean(phi^2) = 1。グリッド上の左 Riemann 和なので O(1/390) の
    # ずれが残る (0.26%)。日内で phi^2 と sigma^2 が独立なぶんの標本誤差も乗る。
    assert ratio.mean() == pytest.approx(1.0, abs=0.01)
    assert np.median(ratio) == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# 季節性が汚す 3 つの推定器 (S4 を作る動機の本体)
#
# 日内季節性は決定論的な時間構造だが、それを除かずに測ると 3 つの推定器が
# 3 つとも別の方向に壊れる。除去でどれも戻ることを固定する。
# ---------------------------------------------------------------------------
def _s4_s3_pair(**extra):
    kw = dict(**CORE, **SMALL)
    kw.update(extra)
    c4 = Config(stage="S4", enable_seasonality=True, **kw)
    return run(c4), run(Config(stage="S3", **kw)), c4


def test_seasonality_inflates_the_roughness_exponent():
    """H は季節性で大きく上振れし、脱季節化で戻る (本番実測 0.136 → 0.310)。

    φ(u) は日内で滑らかに変わる決定論的成分なので、5 分〜4 時間の増分が滑らかに
    なって H が跳ね上がる。GPH の汚染 (+0.02) よりはるかに大きい。
    """
    from simchart.validation import run_all

    m4 = run_all(run(Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)))
    assert m4["rough"]["h_latent_deseasonalized"] is True
    h_clean = m4["rough"]["h_latent"]["h"]
    h_raw = m4["rough"]["h_latent_raw"]["h"]
    assert h_raw > h_clean + 0.05, f"季節性による H の上振れが見えません ({h_raw} vs {h_clean})"

    m3 = run_all(run(Config(stage="S3", **CORE, **SMALL)))
    assert m3["rough"]["h_latent_deseasonalized"] is False
    assert h_clean == pytest.approx(m3["rough"]["h_latent"]["h"], abs=0.02)


def test_seasonality_biases_the_variance_ratio_and_removal_fixes_it():
    """VR は重複窓の端の重みで系統的に下がる。マルチンゲール性の破れではない。

    セッションの端 (= φ² が最大の寄付・引け) は窓に入る回数が少ないので、
    q 期分散だけが過小評価される。φ だけから予測できることを併せて確認する
    (乱数も価格も使わない予測が実測に一致するなら、原因は推定量の重み付け)。
    """
    from simchart.validation import run_all

    cfg = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    m = run_all(run(cfg))
    assert m["scaling"]["variance_ratio_deseasonalized"] is True
    clean = m["scaling"]["variance_ratio"]["max_abs_dev"]
    raw = m["scaling"]["variance_ratio_raw"]["max_abs_dev"]

    # 絶対閾値ではなく同じ標本量・同じシードの S3 を対照に使う。200 日では
    # VR 自体の推定誤差が大きく (本番 5000 日の 0.02 に対しここでは 0.09 前後)、
    # 絶対値で書くと標本量を変えたときに意味が変わってしまう。
    m3 = run_all(run(Config(stage="S3", **CORE, **SMALL)))["scaling"]["variance_ratio"]
    base = m3["max_abs_dev"]
    assert raw > base * 2, f"季節性による VR の低下が見えません (raw {raw} vs S3 {base})"
    assert clean <= base * 1.3, f"脱季節化後が S3 水準に戻っていません ({clean} vs {base})"

    # φ だけからの予測がS4/S3 の比に一致するか (q = 最大)。
    # 生の VR をそのまま予測と比べてはならない — S3 自体も 1 ではないので、
    # 季節性以外の要因 (ボラ変動下での推定バイアス) が混ざる。比を取れば消える。
    phi = np.asarray(
        sz.true_phi_bars(_cal(cfg), 390, steps_per_day=SMALL["steps_per_day"])["value"]
    )
    phi2 = phi**2
    csum = np.concatenate([[0.0], np.cumsum(phi2)])
    q = max(int(x) for x in cfg.validation.vr_qs)
    predicted = float((csum[q:] - csum[:-q]).mean() / (q * phi2.mean()))

    def _vr_at(table, qq):
        return next(row["vr"] for row in table if row["q"] == qq)

    ratio = _vr_at(m["scaling"]["variance_ratio_raw"]["table"], q) / _vr_at(m3["table"], q)
    assert ratio == pytest.approx(predicted, abs=0.04), (
        f"φ からの予測 {predicted:.4f} が実測比 {ratio:.4f} と合いません — "
        f"原因が重み付け以外にある可能性"
    )


def test_seasonality_biases_the_intraday_gph_in_either_direction():
    """GPH の汚染は符号が固定されない (高調波が推定バンドのどこに落ちるか次第)。

    符号つきのゲートを書くと、設定を変えたときに正しい実装が落ちる。
    ここでは大きさだけを固定し、方向は主張しない。
    """
    from simchart.validation import run_all

    m = run_all(run(Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)))
    g = m["seasonality"]["gph_abs_r"]
    assert g["d_raw"] is not None and g["d_true_phi_removed"] is not None
    assert abs(g["d_raw_minus_true_phi"]) > 0.003
    # 推定 φ̂ でもほぼ同じところへ行く (道具として使えるか)
    assert g["d_est_phi_removed"] == pytest.approx(g["d_true_phi_removed"], abs=0.02)


def test_daily_gph_is_immune_to_seasonality():
    """日次リターンの GPH は汚染されない — φ_σ の二乗正規化で日次分散が不変だから。

    潜在 log σ の日次平均は日内平均 log φという全日共通の定数しか受け取らず、
    GPH は定数シフトに不変なので厳密に一致する。
    """
    from simchart.validation import run_all

    kw = dict(**CORE, **SMALL)
    m4 = run_all(run(Config(stage="S4", enable_seasonality=True, **kw)))
    m3 = run_all(run(Config(stage="S3", **kw)))
    assert m4["daily"]["latent_gph_d"]["d"] == pytest.approx(
        m3["daily"]["latent_gph_d"]["d"], abs=1e-9
    )


# ---------------------------------------------------------------------------
# S7 への引き渡し
# ---------------------------------------------------------------------------
def test_time_change_makes_a_seasonal_poisson_uniform():
    """φ_λ による時間変更で季節ポアソンが定常ポアソンになる (Hawkes 推定の前処理)。

    これを通さずに Hawkes を当てると、活発な時間帯へのイベント集中が自己励起と
    説明され分岐比 n が過大推定される (Filimonov-Sornette)。S7 の中核。
    """
    cfg = Config(stage="S4", enable_seasonality=True, **CORE, **SMALL)
    cal = _cal(cfg)
    session = cal.session_seconds()
    rng = np.random.default_rng(5)

    lam_max = float(cal.phi_lambda_of_u(np.linspace(0, 1, 2001)).max())
    chunks = []
    for d in range(150):
        m = rng.poisson(400 * lam_max)
        cand = np.sort(rng.random(m))
        keep = cand[rng.random(m) < cal.phi_lambda_of_u(cand) / lam_max]
        chunks.append((d + keep) * session)
    times = np.concatenate(chunks)

    changed = sz.time_change_by_phi_lambda(times, cal, session)
    h_raw, _ = np.histogram(np.mod(times / session, 1.0), bins=20, range=(0, 1))
    h_new, _ = np.histogram(np.mod(changed / session, 1.0), bins=20, range=(0, 1))
    assert h_raw.max() / h_raw.min() > 4.0
    assert h_new.max() / h_new.min() < 1.3
    # 時間変更は単調 (順序を壊さない)
    assert np.all(np.diff(sz.time_change_by_phi_lambda(np.sort(times), cal, session)) >= 0)


def test_time_change_requires_a_seasonal_calendar():
    cfg = Config(stage="S3", **CORE, **SMALL)
    with pytest.raises(ValueError):
        sz.time_change_by_phi_lambda(np.array([0.0, 1.0]), _cal(cfg), 23400.0)


# ---------------------------------------------------------------------------
# 無効時の挙動
# ---------------------------------------------------------------------------
def test_validation_functions_return_na_without_seasonality():
    """S0〜S3 でも例外を投げず構造化された N/A を返す (検証スイートの規約)。"""
    cfg = Config(stage="S3", **CORE, **SMALL)
    cal = _cal(cfg)
    assert sz.phi_normalization_check(cal)["status"] == "not_applicable"
    assert sz.true_phi_bars(cal, 26)["status"] == "not_applicable"
    assert sz.overnight_stats(np.empty(0), np.zeros(10))["status"] == "not_applicable"


def test_suite_produces_seasonality_branch_for_both_stages():
    from simchart.validation import run_all

    for stage, extra in (("S4", dict(enable_seasonality=True, enable_overnight=True)), ("S3", {})):
        cfg = Config(stage=stage, **extra, **CORE, **JUMP, **SMALL)
        m = run_all(run(cfg))["seasonality"]
        assert set(m) >= {"phi_normalization", "deseasonalization", "overnight"}
        if stage == "S3":
            assert m["deseasonalization"]["status"] == "not_applicable"
            assert m["overnight"]["status"] == "not_applicable"
        else:
            assert m["phi_normalization"]["status"] == "ok"
            assert m["gph_abs_r"]["d_raw_minus_true_phi"] is not None


def test_session_type_other_than_continuous_is_rejected():
    """未実装の分割セッションを黙って連続扱いしない (§14 の禁止事項)。"""
    with pytest.raises(NotImplementedError):
        Config(stage="S4", session_type="split", enable_seasonality=True, **CORE, **SMALL)
