"""S7: Hawkes MLE・分岐比 3 経路再推定・残差検定・分枝表現の相互検証。

閾値は全て事前測定済み (60 日合成 36 万イベント / エンジン 40 日 36 万イベント):
合成復元 n̂ 誤差 +0.0016・max|â−a|=0.007、エンジン 3 経路 −0.0006/−0.0005/+0.063、
再スケーリング KS p=0.43〜0.84・mean_tau=1.000。帯はその数倍を取ってある。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, run
from simchart.validation.hawkes import (
    branching_three_ways,
    estimate_phi_lambda,
    hawkes_mle,
    marks_from_eventlog,
    phi_cumulative,
    phi_lookup,
    simulate_branching,
    time_rescaling_test,
)

S = 23400.0


def _params(cfg: Config):
    a = np.asarray(cfg.hawkes_a)
    betas = 1.0 / np.asarray(cfg.hawkes_tau_seconds)  # [1/秒]
    w = np.asarray(cfg.hawkes_weights)
    mu_day = np.array([
        2.0 * cfg.hawkes_mu_mo, 2.0 * cfg.hawkes_mu_lo,
        cfg.hawkes_delta0 * cfg.hawkes_nbar_ref,
    ])
    n_design = float(np.max(np.abs(np.linalg.eigvals(a))))
    return a, betas, w, mu_day, n_design


# ---------------------------------------------------------------------------
# 合成データ (分枝表現) — 推定器の較正
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic():
    cfg = Config()
    a, betas, w, mu_day, n_design = _params(cfg)
    t_end = 60 * S
    rng = np.random.default_rng(123)
    times, marks = simulate_branching(mu_day / S, a, betas, w, t_end, rng)
    return times, marks, t_end, a, betas, w, mu_day, n_design


def test_mle_recovers_synthetic_params(synthetic):
    times, marks, t_end, a, betas, w, mu_day, n_design = synthetic
    fit = hawkes_mle(times, marks, t_end, betas, w)
    assert fit["converged"]
    assert abs(fit["n_hat"] - n_design) < 0.02, fit["n_hat"]
    assert np.abs(fit["a_hat"] - a).max() < 0.05
    mu_hat_day = fit["mu_hat_per_sec"] * S
    assert np.abs(mu_hat_day / mu_day - 1.0).max() < 0.08


def test_time_rescaling_calibrated(synthetic):
    """正しいモデル (真値・当てはめ値の両方) で Exp(1) を棄却しないこと。"""
    times, marks, t_end, a, betas, w, mu_day, _ = synthetic
    res_true = time_rescaling_test(times, marks, t_end, betas, w, mu_day / S, a)
    assert res_true["ks_pvalue"] > 0.005
    assert abs(res_true["mean_tau"] - 1.0) < 0.02
    fit = hawkes_mle(times, marks, t_end, betas, w)
    res_fit = time_rescaling_test(
        times, marks, t_end, betas, w, fit["mu_hat_per_sec"], fit["a_hat"]
    )
    assert res_fit["ks_pvalue"] > 0.005


def test_time_rescaling_rejects_wrong_model(synthetic):
    """検定力の確認: 励起を無視した Poisson モデルは強く棄却されること。

    (棄却できない検定は検定ではない — CLAUDE.md の検査項目)
    """
    times, marks, t_end, a, betas, w, mu_day, n_design = synthetic
    # 実現レートに合わせた励起なしモデル (レートは正しいが構造が誤り)
    rates = np.array([(marks == y).sum() / t_end for y in range(3)])
    res = time_rescaling_test(
        times, marks, t_end, betas, w, rates, np.zeros((3, 3))
    )
    assert res["ks_pvalue"] < 1e-6, res


def test_raw_path_inflated_by_seasonality_synthetic(synthetic):
    """Filimonov–Sornette の罠の実証 (合成): U 字を除去しないと n̂ が過大。"""
    _, _, _, a, betas, w, mu_day, n_design = synthetic
    m = 512
    u = (np.arange(m) + 0.5) / m
    phi_raw = 0.55 + 2.7 * (2.0 * u - 1.0) ** 2  # 比 5.9 の U 字
    phi = phi_raw / phi_raw.mean()
    t_end = 60 * S
    rng = np.random.default_rng(456)
    times, marks = simulate_branching(
        mu_day / S, a, betas, w, t_end, rng, phi_table=phi, session_seconds=S
    )
    three = branching_three_ways(times, marks, t_end, betas, w, S, phi, n_design)
    assert three["status"] == "ok" and three["converged"]
    assert abs(three["true_phi_minus_design"]) < 0.02
    assert abs(three["est_phi_minus_design"]) < 0.03
    assert three["raw_inflation_over_true"] > 0.03, three["raw_inflation_over_true"]


# ---------------------------------------------------------------------------
# エンジン出力に対する 3 経路 (本物の DGP — CX ベースラインの不一致込み)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine_run():
    cfg = Config.load(
        __import__("pathlib").Path(__file__).resolve().parent.parent / "configs" / "s6.yaml"
    )
    cfg = cfg.replace(
        stage="S7", seed=555, n_days=40, steps_per_day=390, enable_hawkes=True
    )
    r = run(cfg)
    times, marks = marks_from_eventlog(r.events)
    from simchart.layers.l0_calendar import build_calendar
    from simchart.rng import RNGRegistry

    cal = build_calendar(cfg, RNGRegistry(cfg.seed))
    m_phi = 4096
    u = (np.arange(m_phi) + 0.5) / m_phi
    phi = np.asarray(cal.phi_lambda_of_u(u))
    return cfg, times, marks, phi


def test_three_ways_on_engine_output(engine_run):
    """中心ゲートの帯そのもの: ±0.05 / ±0.08 / raw > +0.03。

    実測 (seed 555, 40 日): −0.0006 / −0.0005 / +0.063 — CX ベースラインの
    モデル不一致 (δ0·N(t) を定数で近似) はこの規模では無視できる。
    """
    cfg, times, marks, phi = engine_run
    _, betas, w, _, n_design = _params(cfg)
    t_end = cfg.n_days * S
    three = branching_three_ways(
        times, marks, t_end, betas, w, S, phi, n_design, block_days=10
    )
    assert three["status"] == "ok" and three["converged"]
    assert abs(three["true_phi_minus_design"]) < 0.05
    assert abs(three["est_phi_minus_design"]) < 0.08
    assert three["raw_inflation_over_true"] > 0.03
    blk = three["blocks"]
    assert blk["n_blocks"] >= 3
    assert blk["n_hat_sd"] < 0.03  # ブロック間で暴れない (実測 0.006)


def test_rescaling_on_engine_output(engine_run):
    cfg, times, marks, phi = engine_run
    _, betas, w, _, _ = _params(cfg)
    t_end = cfg.n_days * S
    fit = hawkes_mle(times, marks, t_end, betas, w, phi_table=phi, session_seconds=S)
    res = time_rescaling_test(
        times, marks, t_end, betas, w, fit["mu_hat_per_sec"], fit["a_hat"],
        phi_table=phi, session_seconds=S,
    )
    assert res["ks_pvalue"] > 0.01, res  # 実測 0.84
    assert abs(res["mean_tau"] - 1.0) < 0.02


def test_phi_estimate_tracks_true_profile(engine_run):
    cfg, times, marks, phi = engine_run
    est = estimate_phi_lambda(times, S)
    assert not est["degenerate"]
    m = phi.size
    centers = ((np.arange(52) + 0.5) / 52 * m).astype(int)
    c = float(np.corrcoef(est["table"], phi[centers])[0, 1])
    assert c > 0.95, c  # 実測 0.993


# ---------------------------------------------------------------------------
# 分枝表現 vs thinning (エンジン) の相互検証 — 生成器の独立実装比較
# ---------------------------------------------------------------------------
def test_branching_vs_thinning_cross_validation():
    """同一パラメータの 2 つの生成器 (分枝表現 / エンジンの thinning) で
    レートとクラスタリング統計が整合すること。

    厳密一致は期待しない: エンジンの CX ベースラインは φ·δ0·N(t) で、
    分枝表現は定数 μ_cx = δ0·N̄ref。実現 N̄ の目標比のずれ (−3〜6%) が
    そのままレート差になる。許容 ±10%。
    """
    cfg = Config(
        stage="S7", seed=31, enable_book=True, enable_hawkes=True,
        n_days=30, steps_per_day=390,
    )
    a, betas, w, mu_day, _ = _params(cfg)
    t_end = cfg.n_days * S

    r = run(cfg)
    t_e, m_e = marks_from_eventlog(r.events)
    rng = np.random.default_rng(31)
    t_b, m_b = simulate_branching(mu_day / S, a, betas, w, t_end, rng)

    for y, name in ((0, "MO"), (1, "LO"), (2, "CX")):
        r_e = (m_e == y).sum() / cfg.n_days
        r_b = (m_b == y).sum() / cfg.n_days
        ratio = r_e / r_b
        assert 0.90 < ratio < 1.10, f"{name}: engine {r_e:.0f}/日 vs branching {r_b:.0f}/日"

    def fano(t):
        c, _ = np.histogram(t, bins=np.arange(0.0, t_end + 60.0, 60.0))
        return float(c.var() / c.mean())

    f_e, f_b = fano(t_e), fano(t_b)
    assert 0.6 < f_e / f_b < 1.6, f"Fano engine {f_e:.1f} vs branching {f_b:.1f}"


# ---------------------------------------------------------------------------
# 部品の単体
# ---------------------------------------------------------------------------
def test_phi_cumulative_exact():
    """区分一定テーブルの厳密積分: 解析値と一致し、セッション周期で線形に伸びる。"""
    table = np.array([2.0, 0.5, 1.0, 0.5])  # mean 1
    t = np.array([0.0, S / 4, S / 2, S, 1.5 * S, 3.0 * S])
    got = phi_cumulative(t, table, S)
    # t=1.5S: 1 セッション (=S·mean=S) + 前半 2 ビン全部 (2.0·S/4 + 0.5·S/4)
    want = np.array([0.0, 2.0 * S / 4, 2.5 * S / 4, S, S + 2.5 * S / 4, 3.0 * S])
    assert np.allclose(got, want)


def test_phi_lookup_matches_engine_convention():
    table = np.array([10.0, 20.0, 30.0, 40.0])
    t = np.array([0.0, S * 0.25, S * 0.999, S * 1.1])
    assert np.allclose(phi_lookup(t, table, S), [10.0, 20.0, 40.0, 10.0])


def test_marks_from_eventlog_drops_init_rows():
    cfg = Config(
        stage="S7", seed=5, enable_book=True, enable_hawkes=True,
        n_days=2, steps_per_day=390,
    )
    r = run(cfg)
    times, marks = marks_from_eventlog(r.events)
    assert (times > 0).all()
    assert set(np.unique(marks)) <= {0, 1, 2}
    # 初期化行 (t=0 の LIMIT_ADD 60 本) が落ちていること
    n_t0 = int((r.events.t == 0.0).sum())
    assert n_t0 == 2 * cfg.book_init_levels
