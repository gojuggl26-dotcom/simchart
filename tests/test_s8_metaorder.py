"""S8: メタオーダー分割とアイスバーグのテスト。

中心は 4 つ:
1. S6/S7 経路のビット単位不変 — 本番設定の再実行がダイジェスト一致。
2. 役割分離 — メタオーダーは符号だけを変え、タイミング (Hawkes) は法則不変。
3. プールの会計 — 生成 = 到着 + 空プール生成、子数 = Σ n_exec、長さ分布は
   離散 Pareto 裾に厳密一致、ψ 混合は二項の範囲。
4. アイスバーグの板整合 — 補充込みの完全リプレイ・数量保存 (隠れ量込み)。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from simchart import Config, run
from simchart.types import EventType
from simchart.validation.engine import engine_invariants
from simchart.validation.memory import acf_powerlaw_fit

ROOT = Path(__file__).resolve().parent.parent
S = 23400.0


def _s8_cfg(seed: int = 808, n_days: int = 30, **extra) -> Config:
    cfg = Config.load(ROOT / "configs" / "s7.yaml")
    kw = dict(stage="S8", seed=seed, n_days=n_days, steps_per_day=390,
              enable_metaorder=True, book_debug_invariants=True)
    kw.update(extra)
    return cfg.replace(**kw)


@pytest.fixture(scope="module")
def s8_result():
    cfg = _s8_cfg(enable_iceberg=True)
    return run(cfg), cfg


# ---------------------------------------------------------------------------
# 1. 前段のビット単位不変 (最重要の回帰)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage", ["S6", "S7"])
def test_prior_stage_digest_bit_identical(stage: str):
    metrics_path = ROOT / "results" / stage / "metrics.json"
    if not metrics_path.exists():
        pytest.skip(f"{stage} 本番の results が無い環境ではスキップ")
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    want = stored["metrics"]["runtime"]["determinism"]["digest_first"]
    r = run(Config.load(ROOT / "configs" / f"{stage.lower()}.yaml"))
    assert r.digest() == want, f"{stage} 経路の出力が変わった"


# ---------------------------------------------------------------------------
# 2. 役割分離: タイミングの法則は S7 のまま
# ---------------------------------------------------------------------------
def test_branching_ratio_unchanged(s8_result):
    """n̂ (真の φ 経路) が設計 0.830 ± 0.05 のまま (§3.1 の分離の検証)。"""
    from simchart.layers.l0_calendar import build_calendar
    from simchart.rng import RNGRegistry
    from simchart.validation.hawkes import hawkes_mle, marks_from_eventlog

    r, cfg = s8_result
    times, marks = marks_from_eventlog(r.events)
    betas = 1.0 / np.asarray(cfg.hawkes_tau_seconds)
    w = np.asarray(cfg.hawkes_weights)
    cal = build_calendar(cfg, RNGRegistry(cfg.seed))
    u = (np.arange(4096) + 0.5) / 4096
    phi = np.asarray(cal.phi_lambda_of_u(u))
    fit = hawkes_mle(times, marks, cfg.n_days * S, betas, w,
                     phi_table=phi, session_seconds=S)
    assert fit["converged"]
    assert abs(fit["n_hat"] - 0.830) < 0.05, fit["n_hat"]


def test_overdispersion_still_present(s8_result):
    r, cfg = s8_result
    ev = r.events
    keep = (ev.event_type != int(EventType.TRADE)) & (
        ev.event_type != int(EventType.MODIFY)
    ) & (ev.t > 0)
    t = ev.t[keep]
    edges = np.arange(S, cfg.n_days * S + 60.0, 60.0)
    c, _ = np.histogram(t[t >= S], bins=edges)
    assert c.var() / c.mean() > 2.0  # S7 実測 14 前後


# ---------------------------------------------------------------------------
# 3. プールの会計と分布
# ---------------------------------------------------------------------------
def test_metaorder_accounting(s8_result):
    r, cfg = s8_result
    d = r.meta["l3"]["meta"]
    mo = r.events.meta["metaorders"]
    assert d["n_metaorders"] == mo["sign"].size
    assert d["arrivals"] + d["spawned_on_empty"] == d["n_metaorders"]
    assert int(mo["n_exec"].sum()) == d["children"]
    done = mo["n_exec"] >= mo["n_total"]
    assert int(done.sum()) == d["completed"]
    assert (mo["n_exec"] <= mo["n_total"]).all()
    # 完走したものは執行数が長さに厳密一致
    assert np.array_equal(mo["n_exec"][done], mo["n_total"][done])
    # 実行スパンの整合: 子を持つものは t_first <= t_last、ミッドが記録済み
    started = mo["n_exec"] > 0
    assert (mo["t_first"][started] <= mo["t_last"][started]).all()
    assert (mo["mid_first"][started] > 0).all()
    assert (mo["vol_last"][started] >= mo["vol_first"][started]).all()


def test_psi_mixing_is_binomial(s8_result):
    r, cfg = s8_result
    d = r.meta["l3"]["meta"]
    n = d["children"] + d["noise_trades"]
    z = (d["children"] - cfg.meta_psi * n) / math.sqrt(
        cfg.meta_psi * (1 - cfg.meta_psi) * n
    )
    assert abs(z) < 4.0, z


def test_length_distribution_exact_tail(s8_result):
    """離散裾 P(N ≥ n) = n^{-α} (N_min=1) の厳密性を複数点の二項 z で検定。"""
    r, cfg = s8_result
    n_tot = r.events.meta["metaorders"]["n_total"]
    m = n_tot.size
    for n0 in (2, 5, 20, 100):
        p_th = float(n0) ** (-cfg.meta_alpha)
        obs = float((n_tot >= n0).mean())
        z = (obs - p_th) / math.sqrt(p_th * (1 - p_th) / m)
        assert abs(z) < 4.0, f"P(N>={n0}): obs {obs:.4f} vs {p_th:.4f} (z={z:.1f})"
    # 符号は 50/50
    sgn = r.events.meta["metaorders"]["sign"]
    zb = (float((sgn > 0).sum()) - 0.5 * m) / math.sqrt(0.25 * m)
    assert abs(zb) < 4.0


def test_sign_acf_level_and_decay(s8_result):
    """C(1) は帯の緩め版、γ̂ は存在確認まで。

    30 日の単一シードで γ̂ を狭く縛ってはならない: α<2 では ACF の裾が
    標本内の最大メタオーダー (whale) に支配され、収束が遅い (実測: 30 日で
    γ̂ ∈ [0.30, 0.96] の散らばり、250 日 8 シードで中央値 0.626 [0.52, 0.66])。
    量的判定は本番 (250 日 × 10 シードの中央値) のゲートが行う。
    """
    r, cfg = s8_result
    ev = r.events
    s = np.asarray(ev.meta["agg_trade_side"], dtype=np.float64)
    t = np.asarray(ev.meta["agg_trade_t"])
    s = s[t >= cfg.book_burn_in_days * S]
    d0 = s - s.mean()
    c1 = float(d0[:-1] @ d0[1:]) / float(d0 @ d0)
    assert 0.04 < c1 < 0.30, c1  # 30 日の短窓なのでゲート帯より緩く
    fit = acf_powerlaw_fit(s, (2, 300), max_lag=300)
    assert fit["status"] == "ok"
    assert 0.25 < fit["gamma"] < 1.1, fit["gamma"]  # 存在確認 (量は本番ゲート)


def test_sequential_mode_matches_pool_gamma():
    """§3.4 相互検証: 逐次版でも γ = α−1 が成立し、プール版と整合する。

    単一シードでは whale 支配で γ̂ が散らばるため、3 シードの中央値同士で比べる。
    """
    med = {}
    for label, seq in (("pool", False), ("sequential", True)):
        gs = []
        for seed in (901, 902, 903):
            cfg = _s8_cfg(seed=seed, n_days=60, meta_sequential=seq,
                          book_debug_invariants=False,
                          book_window_half_ticks=9500)
            r = run(cfg)
            s = np.asarray(r.events.meta["agg_trade_side"], dtype=np.float64)
            t = np.asarray(r.events.meta["agg_trade_t"])
            s = s[t >= cfg.book_burn_in_days * S]
            fit = acf_powerlaw_fit(s, (2, 400), max_lag=400)
            assert fit["status"] == "ok", label
            gs.append(fit["gamma"])
            if seq:
                # 逐次版はアクティブ数が常に 1 以下
                assert float(r.events.meta["pool_grid"].max()) <= 1.0
        med[label] = float(np.median(gs))
    assert abs(med["pool"] - med["sequential"]) < 0.25, med
    for g in med.values():
        assert 0.35 < g < 0.95, med


# ---------------------------------------------------------------------------
# 4. アイスバーグと板の整合
# ---------------------------------------------------------------------------
def test_engine_invariants_with_iceberg(s8_result):
    r, _ = s8_result
    inv = engine_invariants(r.meta["l3"], r.events.t)
    assert inv["status"] == "ok"
    assert inv["all_passed"], inv


def test_full_replay_with_iceberg_refills(s8_result):
    from _book_replay import replay_and_verify

    r, cfg = s8_result
    d = r.meta["l3"]["iceberg"]
    assert d["refills"] > 500  # 検定力: 補充が実際に起きていること
    assert replay_and_verify(r.events, refill_tail=True) > 10_000


def test_iceberg_keep_priority_mode_replays():
    cfg = _s8_cfg(seed=811, n_days=10, enable_iceberg=True,
                  book_iceberg_refill_tail=False)
    r = run(cfg)
    from _book_replay import replay_and_verify

    assert r.meta["l3"]["iceberg"]["refills"] > 100
    assert replay_and_verify(r.events, refill_tail=False) > 3_000


def test_iceberg_ledger_identity(s8_result):
    """隠れ量込みの数量保存: lo_in = live + cancelled + passive + entry。"""
    from simchart.layers.book_engine import (
        C_ICE_HIDDEN_IN,
        C_ICE_REFILL_VOL,
        C_LIVE_VOL,
        C_VOL_CANCELLED,
        C_VOL_LO_ENTRY_EXEC,
        C_VOL_LO_IN,
        C_VOL_PASSIVE,
    )

    r, _ = s8_result
    c = r.meta["l3"]["counters"]
    lhs = c[C_VOL_LO_IN]
    rhs = c[C_LIVE_VOL] + c[C_VOL_CANCELLED] + c[C_VOL_PASSIVE] + c[C_VOL_LO_ENTRY_EXEC]
    assert abs(lhs - rhs) < 1e-6 * max(lhs, 1.0), (lhs, rhs)
    # 補充は投入済み隠れ量を超えない
    assert c[C_ICE_REFILL_VOL] <= c[C_ICE_HIDDEN_IN] + 1e-9


def test_deterministic(s8_result):
    r, cfg = s8_result
    r2 = run(cfg)
    assert r.digest() == r2.digest()


def test_throughput(s8_result):
    r, _ = s8_result
    assert r.meta["l3"]["throughput_events_per_sec"] > 50_000


# ---------------------------------------------------------------------------
# 設定の検証
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1.0, 0.8, 2.0, 2.4])
def test_config_rejects_alpha_outside_open_interval(alpha):
    with pytest.raises(ValueError, match="meta_alpha"):
        _s8_cfg(meta_alpha=alpha)


def test_config_rejects_bad_psi_and_ratio():
    with pytest.raises(ValueError, match="meta_psi"):
        _s8_cfg(meta_psi=1.5)
    with pytest.raises(ValueError, match="meta_supply_ratio"):
        _s8_cfg(meta_supply_ratio=1.2)


def test_config_requires_book():
    with pytest.raises(ValueError, match="enable_book"):
        Config(stage="S8", enable_metaorder=True)
    with pytest.raises(ValueError, match="enable_book"):
        Config(stage="S8", enable_iceberg=True)


def test_without_book_strips_s8():
    cfg = _s8_cfg(enable_iceberg=True)
    base = cfg.without_book()
    assert base.enable_metaorder is False and base.enable_iceberg is False
    assert base.meta_supply_ratio == Config().meta_supply_ratio