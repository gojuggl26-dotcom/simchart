"""S9: queue-reactive (状態依存の意思決定層) のテスト。

中心は 4 つ:
1. **S6/S7/S8 経路のビット単位不変** (QR off の乱数消費列は完全一致)。
2. **状態依存の実在** — スプレッドが広いほど板内配置率が上がり、近い注文ほど
   取り消されやすい (実測分布で確認)。
3. **板の健全性** — 重み付き取消の下でも定常 (前面殲滅の正帰還が起きない —
   w_floor の存在理由)。
4. **不変量** — γ/C(1) が動かない (状態依存は符号に触れない)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simchart import Config, run
from simchart.types import EventType
from simchart.validation.engine import engine_invariants

ROOT = Path(__file__).resolve().parent.parent
S = 23400.0


def _s9_cfg(seed: int = 909, n_days: int = 30, **extra) -> Config:
    cfg = Config.load(ROOT / "configs" / "s8.yaml")
    kw = dict(stage="S9", seed=seed, n_days=n_days,
              enable_queue_reactive=True, book_debug_invariants=True)
    kw.update(extra)
    return cfg.replace(**kw)


@pytest.fixture(scope="module")
def s9_result():
    cfg = _s9_cfg()
    return run(cfg), cfg


def test_s8_production_digest_bit_identical():
    metrics_path = ROOT / "results" / "S8" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("S8 本番の results が無い環境ではスキップ")
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    want = stored["metrics"]["runtime"]["determinism"]["digest_first"]
    r = run(Config.load(ROOT / "configs" / "s8.yaml"))
    assert r.digest() == want, "S8 経路の出力が変わった — S9 改修が前段を汚染"


def test_engine_invariants_and_replay(s9_result):
    from _book_replay import replay_and_verify

    r, _ = s9_result
    inv = engine_invariants(r.meta["l3"], r.events.t)
    assert inv["status"] == "ok" and inv["all_passed"], inv
    assert replay_and_verify(r.events) > 10_000


def test_book_stays_healthy(s9_result):
    """★w_floor の回帰: 遠方重みゼロは前面殲滅でスプレッド 3,700 tick を実測。"""
    r, cfg = s9_result
    ev = r.events
    burn = cfg.book_burn_in_days * S
    bb = ev.meta["best_bid_tick"]
    ba = ev.meta["best_ask_tick"]
    ok = (bb >= 0) & (ba >= 0) & (ev.t >= burn)
    sp = ba[ok] - bb[ok]
    assert 1 <= np.median(sp) <= 5, np.median(sp)
    assert np.percentile(sp, 95) < 30
    from simchart.layers.book_engine import C_EMPTY_SIDE_TIME

    frac = r.meta["l3"]["counters"][C_EMPTY_SIDE_TIME] / (cfg.n_days * S)
    assert frac < 0.005


def test_inspread_rate_rises_with_spread(s9_result):
    """§5 の実在確認: 配置直前スプレッド別の板内配置率が単調に上がる。"""
    r, cfg = s9_result
    ev = r.events
    burn = cfg.book_burn_in_days * S
    lo = (ev.event_type == int(EventType.LIMIT_ADD)) & (ev.t >= burn)
    bb = ev.meta["best_bid_tick"].astype(np.float64)
    ba = ev.meta["best_ask_tick"].astype(np.float64)
    # 直前行の板状態 (配置前) — S8 の prev-mid と同じ 1 行前規約
    sp_prev = np.concatenate([[np.nan], (ba - bb)[:-1]])
    tick = ev.meta["tick_size"]
    base = ev.meta["base_price"]
    px_ticks = np.round((ev.price[lo] - base) / tick)
    bb_prev = np.concatenate([[np.nan], bb[:-1]])[lo]
    ba_prev = np.concatenate([[np.nan], ba[:-1]])[lo]
    side_lo = ev.side[lo]
    inside = np.where(
        side_lo == 1, px_ticks > bb_prev, px_ticks < ba_prev
    ) & np.isfinite(bb_prev) & np.isfinite(ba_prev)
    s_at = sp_prev[lo]
    rates = []
    for s_lo, s_hi in ((2, 3), (3, 5), (5, 9)):
        m = np.isfinite(s_at) & (s_at >= s_lo) & (s_at < s_hi)
        if m.sum() > 500:
            rates.append(float(inside[m].mean()))
    assert len(rates) >= 2
    assert all(b > a for a, b in zip(rates, rates[1:])), rates


def test_cancel_distance_tilt():
    """§6 の実在確認: 傾斜を入れると取消が best 寄りに偏る。

    ★本番 (small tick) の既定は**中立** — 前方傾斜は前面を薄くして赤字指標を
    悪化させることが実測で判ったため (config の注記)。ここでは傾斜を明示的に
    有効化して機構そのものの動作を確認する (large tick レジームで使う側)。
    """
    cfg = _s9_cfg(qr_cx_dist_decay=0.10, qr_cx_w_floor=0.25,
                  qr_cx_len_pow=0.3, qr_cx_back=1.0)
    r = run(cfg)
    ev = r.events
    burn = cfg.book_burn_in_days * S
    cx = (ev.event_type == int(EventType.CANCEL)) & (ev.t >= burn)
    tick = ev.meta["tick_size"]
    base = ev.meta["base_price"]
    px = np.round((ev.price[cx] - base) / tick)
    bb = ev.meta["best_bid_tick"].astype(np.float64)
    ba = ev.meta["best_ask_tick"].astype(np.float64)
    bb_prev = np.concatenate([[np.nan], bb[:-1]])[cx]
    ba_prev = np.concatenate([[np.nan], ba[:-1]])[cx]
    side_cx = ev.side[cx]
    dist = np.where(side_cx == 1, bb_prev - px, px - ba_prev)
    dist = dist[np.isfinite(dist) & (dist >= 0)]
    # 一様取消 (S8) と比べ取消距離の中央値が短いこと — 同一シードの S8 と比較
    cfg8 = Config.load(ROOT / "configs" / "s8.yaml").replace(
        seed=cfg.seed, n_days=cfg.n_days
    )
    r8 = run(cfg8)
    ev8 = r8.events
    cx8 = (ev8.event_type == int(EventType.CANCEL)) & (ev8.t >= burn)
    px8 = np.round((ev8.price[cx8] - ev8.meta["base_price"]) / ev8.meta["tick_size"])
    bb8 = ev8.meta["best_bid_tick"].astype(np.float64)
    ba8 = ev8.meta["best_ask_tick"].astype(np.float64)
    bb8p = np.concatenate([[np.nan], bb8[:-1]])[cx8]
    ba8p = np.concatenate([[np.nan], ba8[:-1]])[cx8]
    d8 = np.where(ev8.side[cx8] == 1, bb8p - px8, px8 - ba8p)
    d8 = d8[np.isfinite(d8) & (d8 >= 0)]
    assert np.median(dist) < np.median(d8), (np.median(dist), np.median(d8))


def test_sign_structure_untouched(s9_result):
    """状態依存は符号に触れない: C(1) が S8 帯のまま。"""
    r, cfg = s9_result
    s_arr = np.asarray(r.events.meta["agg_trade_side"], dtype=np.float64)
    t = np.asarray(r.events.meta["agg_trade_t"])
    s_arr = s_arr[t >= cfg.book_burn_in_days * S]
    d = s_arr - s_arr.mean()
    c1 = float(d[:-1] @ d[1:]) / float(d @ d)
    assert 0.09 < c1 < 0.18, c1


def test_obi_emerges_mechanically(s9_result):
    """⑩: バイアスなし (qr_obi_bias=0) でも I と次のミッド変化に正相関。"""
    r, cfg = s9_result
    assert cfg.qr_obi_bias == 0.0
    ev = r.events
    burn = cfg.book_burn_in_days * S
    pm = np.asarray(ev.meta["agg_trade_prev_mid_tick"], dtype=np.float64)
    t = np.asarray(ev.meta["agg_trade_t"])
    tr = ev.event_type == int(EventType.TRADE)
    tr_t = ev.t[tr]
    starts = np.flatnonzero(np.concatenate([[True], np.diff(tr_t) > 0]))
    trade_idx = np.flatnonzero(tr)
    pre_idx = np.maximum(trade_idx[starts] - 2, 0)
    db = np.asarray(ev.meta["depth_bid"], dtype=np.float64)[pre_idx]
    da = np.asarray(ev.meta["depth_ask"], dtype=np.float64)[pre_idx]
    imb = (db - da) / np.maximum(db + da, 1e-9)
    keep = (
        np.isfinite(pm[:-1]) & np.isfinite(pm[1:]) & (t[:-1] >= burn)
    )
    dm = (pm[1:] - pm[:-1])[keep]
    ik = imb[:-1][keep]
    c = float(np.corrcoef(ik, dm)[0, 1])
    assert c > 0.05, c


def test_uz_transform_lowers_eta():
    """UZ fallback: η̂ が uz_eta と単調に対応し、生ミッドより下がる。"""
    from simchart.layers.book_engine import uz_transform

    rng = np.random.default_rng(3)
    mid = np.cumsum(rng.normal(0, 0.3, 200_000))  # tick 単位の効率価格

    def eta_of(series):
        ch = np.diff(series)
        ch = ch[ch != 0]
        sg = np.sign(ch)
        cont = float((sg[1:] == sg[:-1]).sum())
        alt = float((sg[1:] != sg[:-1]).sum())
        return cont / (2.0 * alt)

    etas = [eta_of(uz_transform(mid, e)) for e in (0.05, 0.15, 0.35)]
    assert etas[0] < etas[1] < etas[2], etas
    assert etas[0] < 0.35


def test_uz_layer_wires_into_observation():
    cfg = _s9_cfg(n_days=5, enable_uncertainty_zones=True)
    r = run(cfg)
    assert r.observation.source.endswith("+uz)")
    cfg_off = _s9_cfg(n_days=5)
    r2 = run(cfg_off)
    assert not r2.observation.source.endswith("+uz)")
    # イベントログは生のまま (観測だけの層)
    assert np.array_equal(r.events.t, r2.events.t)


def test_deterministic_and_throughput(s9_result):
    r, cfg = s9_result
    assert r.meta["l3"]["throughput_events_per_sec"] > 50_000
    r2 = run(cfg)
    assert r.digest() == r2.digest()


def test_config_validation():
    with pytest.raises(ValueError, match="qr_cx_w_floor"):
        _s9_cfg(qr_cx_w_floor=0.0)
    with pytest.raises(ValueError, match="enable_book"):
        Config(stage="S9", enable_queue_reactive=True)
    with pytest.raises(ValueError, match="uz_eta"):
        _s9_cfg(enable_uncertainty_zones=True, uz_eta=0.6)
    base = _s9_cfg().without_book()
    assert base.enable_queue_reactive is False
    assert base.qr_inspread_slope == Config().qr_inspread_slope