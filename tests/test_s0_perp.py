"""S0-perp の骨格ゲート (§11 の perp 側)。

株式保全は tests/test_perp_fork.py が担当。こちらは perp 設定の完走・決定性・
時間軸・週次平坦・validation/perp.py の規約を小規模設定で固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart.config import Config
from simchart.grid import TimeGrid
from simchart.pipeline import determinism_check, run


def _perp_cfg(**over) -> Config:
    kw = dict(
        stage="S0", market_type="perp_clob", session_type="24h",
        n_days=28, steps_per_day=1440, sigma_bar=0.60,
    )
    kw.update(over)
    return Config(**kw)


# ---------------------------------------------------------------------------
# 時間軸 (§4)
# ---------------------------------------------------------------------------
def test_time_grid_from_config() -> None:
    tg = TimeGrid.from_config(_perp_cfg())
    assert tg.ann_days == 365
    assert tg.seconds_per_day == 86400.0
    assert tg.dt_seconds == 60.0
    assert tg.steps_per_year == 365 * 1440
    assert tg.days_to_steps(2.5) == 3600
    assert tg.per_day_to_per_step(1440.0) == 1.0

    eq = TimeGrid.from_config(Config(stage="S0"))
    assert eq.ann_days == 252
    assert eq.seconds_per_day == 23400.0


def test_perp_calendar_is_24_7() -> None:
    from simchart.layers.l0_calendar import PerpCalendar, build_calendar
    from simchart.rng import RNGRegistry

    cal = build_calendar(_perp_cfg(), RNGRegistry(1))
    assert isinstance(cal, PerpCalendar)
    assert cal.session_seconds() == 86400.0
    assert cal.overnight_gaps().sum() == 0.0
    # 週内位置: 7 日でちょうど 1 周
    t = np.array([0.0, 86400.0 * 3.5, 86400.0 * 7.0])
    w = cal.weekly_position(t)
    assert w[0] == 0.0 and abs(w[1] - 0.5) < 1e-12 and w[2] == 0.0


def test_annualization_uses_365() -> None:
    """GBM の per-step 分散が 365 日換算になっていること (§4.2)。

    24/7 化の最重要検査 — ここが 252 のままだと年率 0.60 指定が実効 0.71 に
    なる (√(365/252) 倍) 形で静かにずれる。
    """
    cfg = _perp_cfg()
    assert abs(cfg.sigma_step - 0.60 / np.sqrt(365.0 * 1440.0)) < 1e-15


# ---------------------------------------------------------------------------
# 骨格の完走・決定性・GBM 統計 (小規模スモーク — 本番は run_stage)
# ---------------------------------------------------------------------------
def test_pipeline_runs_and_deterministic() -> None:
    cfg = _perp_cfg()
    result = run(cfg)
    assert result.observation.log_price.shape[0] == 28 * 1440 + 1
    det = determinism_check(cfg, first=result)
    assert det["bitwise_identical"]


def test_gbm_daily_sd_matches_sigma() -> None:
    cfg = _perp_cfg(n_days=365)
    r = run(cfg)
    daily = np.diff(r.observation.log_price[:: cfg.steps_per_day])
    ann = daily.std(ddof=1) * np.sqrt(365.0)
    assert 0.54 < ann < 0.66, ann  # SE ≈ 0.60/√(2·365) ≈ 0.022


def test_perp_and_equity_paths_differ() -> None:
    """同一シードでも perp と equity は別経路 (時間換算が違う) — かつ
    乱数消費列そのものは同一 (market_type は spawn 鍵に入らない §6.2)。"""
    r_p = run(_perp_cfg())
    r_e = run(Config(stage="S0", n_days=28, steps_per_day=1440))
    # 拡散乱数の消費は同一 (ダイジェスト一致)
    assert (
        r_p.meta["l2"]["diffusion_digest"] == r_e.meta["l2"]["diffusion_digest"]
    )
    # だが価格経路は年率換算の違いで異なる
    assert r_p.price.log_p_star[-1] != r_e.price.log_p_star[-1]


# ---------------------------------------------------------------------------
# validation/perp.py の規約 (§8)
# ---------------------------------------------------------------------------
def test_perp_validation_all_callable_without_exception() -> None:
    from simchart.validation import perp as pv

    for fn in (
        pv.basis_stats, pv.funding_stats, pv.funding_sawtooth,
        pv.arb_band_analysis, pv.oi_dynamics, pv.liquidation_cascade_sizes,
        pv.liq_density_profile, pv.g_liquidation_derived,
        pv.block_discretization_effect,
    ):
        out = fn()
        assert out["status"] == "not_applicable" and out["value"] is None


def test_weekly_profile_flat_on_gbm() -> None:
    rng = np.random.default_rng(7)
    n = 8 * 7 * 24  # 8 週の時間足
    x = np.abs(rng.standard_normal(n))
    t = np.arange(n) * 3600.0
    from simchart.validation.perp import weekly_profile

    out = weekly_profile(x, t)
    assert out["status"] == "ok"
    assert out["max_over_min"] < 1.25  # 1344 点/ビン 192 のゆらぎ帯


def test_phi_normalization_check_distinguishes_kinds() -> None:
    from simchart.layers.l0_calendar import fourier_profile, normalize_phi_sigma
    from simchart.validation.perp import phi_normalization_check

    u = np.linspace(0.0, 1.0, 20001)
    coeffs = ((0.3439, 0.0777, 0.0), (0.0, 0.0, 0.0), -0.2219)
    phi = normalize_phi_sigma(*coeffs) * fourier_profile(u, *coeffs)
    assert phi_normalization_check(phi, "sigma")["abs_error"] < 1e-3
    # 同じ φ を lambda 規約で測ると外れる (規約の違いが §3.1 の要点)
    assert phi_normalization_check(phi, "lambda")["abs_error"] > 5e-3


# ---------------------------------------------------------------------------
# L4 スタブ (§7)
# ---------------------------------------------------------------------------
def test_l4_layer_is_none_at_s0_perp() -> None:
    from simchart.layers.l4_positions import PositionLayer, build_position_layer

    assert build_position_layer(_perp_cfg()) is None
    layer = PositionLayer(_perp_cfg())
    with pytest.raises(NotImplementedError):
        layer.open_interest()
    with pytest.raises(NotImplementedError):
        layer.scan_liquidations(100.0)


def test_types_perp_constructible() -> None:
    from simchart.types_perp import FundingState, LiquidationEvent, PositionBook

    pb = PositionBook(price_edges=np.linspace(4.0, 5.0, 11))
    fs = FundingState()
    ev = LiquidationEvent(t_sec=1.0, side=+1, size=2.0, trigger_price=4.5)
    assert pb.price_edges.shape == (11,) and fs.current_rate == 0.0 and ev.side == 1
