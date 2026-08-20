"""S6: ZI 板とイベントエンジンのテスト。

中心は §9 の不変条件 (非クロス・保存則・FIFO・単調時刻) と、エンジンを**意図的に
壊しにいく**ケース (複数レベルを掃く aggressive limit・枯渇時の棄却・部分約定後の
取消)。「あとで入れる」と必ず入らない類のテストなので S6 で入れる。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simchart import Config, run
from simchart.layers.book_engine import (
    C_AGGRESSIVE_LO,
    C_CROSS_VIOL,
    C_INV_FIFO_VIOL,
    C_INV_VOL_VIOL,
    C_LO_INSTANT,
    C_MO_REJECT_EVENTS,
    C_MO_REJECT_VOL,
)
from simchart.types import EventType
from simchart.validation.engine import engine_invariants, throughput

S5KW = dict(
    enable_msm=True, enable_slow_ou=True, enable_rough=True,
    enable_jump=True, enable_leverage=True,
    enable_seasonality=True, enable_overnight=True, enable_chaos_vol=True,
    jump_lambda_per_year=5.0, jump_eta_down=35.0, jump_eta_up=56.0,
    jump_qv_share_target=0.12, jump_p_up=0.42,
    leverage_rho_rough=-0.60, leverage_rho_slow=-0.35,
)
SMALL = dict(n_days=20, steps_per_day=390)


def _run_book(seed: int = 42, debug: bool = True, **extra):
    cfg = Config(
        stage="S6", seed=seed, enable_book=True, book_debug_invariants=debug,
        **extra, **S5KW, **SMALL,
    )
    return run(cfg), cfg


@pytest.fixture(scope="module")
def book_result():
    return _run_book()


# ---------------------------------------------------------------------------
# 不変条件 (指示書 §9)
# ---------------------------------------------------------------------------
def test_engine_invariants_all_pass(book_result):
    r, _ = book_result
    inv = engine_invariants(r.meta["l3"], r.events.t)
    assert inv["status"] == "ok"
    assert inv["no_cross"]
    assert inv["order_conservation"]
    assert inv["volume_conservation"]
    assert inv["lo_volume_ledger"]
    assert inv["fifo_priority"]
    assert inv["level_volume_consistency"]
    assert inv["monotone_time"]


def test_best_quotes_never_cross(book_result):
    r, _ = book_result
    bb = r.events.meta["best_bid_tick"]
    ba = r.events.meta["best_ask_tick"]
    both = (bb >= 0) & (ba >= 0)
    assert int(((ba[both] - bb[both]) <= 0).sum()) == 0


def test_determinism_bitwise(book_result):
    """同一シード 2 回でイベントログがビット単位一致 (指示書 §13)。"""
    r1, cfg = book_result
    r2 = run(cfg)
    assert r1.digest() == r2.digest()
    np.testing.assert_array_equal(r1.events.t, r2.events.t)
    np.testing.assert_array_equal(r1.events.price, r2.events.price)


def test_l2_streams_untouched_by_book(book_result):
    """板は l3.* しか消費しない — L2 のダイジェストが板 off と完全一致 (凍結検証)。"""
    r_book, cfg = book_result
    r_ref = run(cfg.without_book())
    a, b = r_book.meta["l2"], r_ref.meta["l2"]
    assert a["diffusion_digest"] == b["diffusion_digest"]
    assert a["msm"]["switch_digest"] == b["msm"]["switch_digest"]
    assert a["rough"]["y_digest"] == b["rough"]["y_digest"]
    assert a["chaos"]["sha256"] == b["chaos"]["sha256"]
    np.testing.assert_array_equal(
        a["vol_subsample"]["log_vol"], b["vol_subsample"]["log_vol"]
    )


def test_throughput_after_warmup(book_result):
    r, _ = book_result
    tp = throughput(r.meta["l3"])
    assert tp["status"] == "ok"
    # ゲートは 50k。テストは CI 環境の揺れを見込んで 10 倍の余裕を置いても通る値
    # (実測 ~10M ev/s) だが、下限はゲートと同じにしておく。
    assert tp["events_per_sec"] > 50_000


# ---------------------------------------------------------------------------
# マッチングの意地悪ケース
# ---------------------------------------------------------------------------
def test_depletion_rejection_keeps_ledgers():
    """板を薄くして枯渇棄却の経路を踏み、それでも保存則が守られることを確認する。

    崩壊レジーム (μ/α や δ の上げすぎ) は窓逸脱で明示的に失敗するため、
    5 日の短期 + 薄い初期化で「時々枯渇する」程度に留める。枯渇時の配置基準
    (記憶した best) から aggressive limit の経路も踏まれ得る (踏まれた数は記録)。
    """
    # 健全な定常レートのまま**初期板だけ極薄**にする: 序盤 (板が立ち上がる前) に
    # 枯渇棄却が発生し、その後は定常へ回復する。持続的に μ >= α にすると板が
    # 崩壊して窓逸脱で止まる (それはそれで正しい失敗動作 — 別のテストではない)。
    cfg = Config(
        stage="S6", seed=7, enable_book=True, book_debug_invariants=True,
        n_days=5, steps_per_day=390,
        book_init_levels=1, book_init_size=1.0,
        **S5KW,
    )
    r = run(cfg)
    c = r.meta["l3"]["counters"]
    inv = engine_invariants(r.meta["l3"], r.events.t)
    # 薄い板では成行の反対側枯渇が起きる (起きなければこのテストは検定力ゼロ)
    assert c[C_MO_REJECT_EVENTS] > 0
    assert c[C_MO_REJECT_VOL] > 0
    # それでも保存則・非クロスは成立
    assert inv["order_conservation"]
    assert inv["volume_conservation"]
    assert inv["lo_volume_ledger"]
    assert c[C_CROSS_VIOL] == 0


def test_full_replay_verifies_fifo_and_matching(book_result):
    """★イベントログを**独立に完全リプレイ**して照合する最強の検定。

    ログだけから板を再構成し、全 TRADE 行が「その価格のキュー先頭」を消費している
    ことを検証する — FIFO (時間優先)・価格優先・部分約定・取消の全てを同時に縛る。
    注文 id はプールスロットで再利用されるため「id → 配置時刻」の辞書では検証
    できない (最初のテストはそれで自分が壊れていた) — リプレイは再利用と無関係。
    """
    from collections import defaultdict, deque

    r, _ = book_result
    ev = r.events
    queues: dict[float, deque] = defaultdict(deque)  # price -> deque[[oid, rem]]
    n_trades_checked = 0
    for i in range(ev.t.size):
        et = int(ev.event_type[i])
        oid = int(ev.order_id[i])
        px = float(ev.price[i])
        if et == int(EventType.LIMIT_ADD):
            rest = float(ev.size[i]) - float(ev.meta["exec_size"][i])
            if oid >= 0 and rest > 0:
                queues[px].append([oid, rest])
        elif et == int(EventType.TRADE):
            q = queues[px]
            assert q, f"約定 {i}: 価格 {px} のキューが空 (リプレイ破綻)"
            head = q[0]
            assert head[0] == oid, (
                f"約定 {i}: FIFO 違反 — キュー先頭 {head[0]} でなく {oid} が約定"
            )
            head[1] -= float(ev.size[i])
            assert head[1] > -1e-9, f"約定 {i}: 残量が負 ({head[1]})"
            if head[1] <= 1e-9:
                q.popleft()
            n_trades_checked += 1
        elif et == int(EventType.CANCEL):
            q = queues[px]
            for j, item in enumerate(q):
                if item[0] == oid:
                    assert abs(item[1] - float(ev.size[i])) < 1e-9, (
                        f"取消 {i}: 残量不一致 (板 {item[1]} vs ログ {ev.size[i]})"
                    )
                    del q[j]
                    break
            else:
                raise AssertionError(f"取消 {i}: 対象注文 {oid} がキューに存在しない")
    assert n_trades_checked > 10_000  # 検定力の確認


def test_trade_rows_aggregate_to_aggressor_orders(book_result):
    """集約系列の総量 = TRADE 行の総量、符号は各攻撃で一様。"""
    r, _ = book_result
    ev = r.events
    tr = ev.event_type == int(EventType.TRADE)
    assert ev.meta["agg_trade_size"].sum() == pytest.approx(float(ev.size[tr].sum()))
    assert set(np.unique(ev.meta["agg_trade_side"])) <= {-1, 1}


def test_sign_acf_artifact_removed(book_result):
    """★約定行のままだと機械的な正の符号相関 (+0.4 級) が出る。集約後は消える。"""
    r, _ = book_result
    ev = r.events
    tr_side = ev.side[ev.event_type == int(EventType.TRADE)].astype(float)
    d = tr_side - tr_side.mean()
    acf_rows = float(d[:-1] @ d[1:] / (d @ d))
    s = ev.meta["agg_trade_side"].astype(float)
    d2 = s - s.mean()
    acf_agg = float(d2[:-1] @ d2[1:] / (d2 @ d2))
    assert acf_rows > 0.2  # 人工物の存在確認 (無ければこの検定は無意味)
    assert abs(acf_agg) < 4.0 / math.sqrt(s.size)


# ---------------------------------------------------------------------------
# 板の性質・ベースライン
# ---------------------------------------------------------------------------
def test_spread_and_depth_in_regime(book_result):
    from simchart.validation.micro import depth_profile, spread_distribution

    r, cfg = book_result
    burn = cfg.book_burn_in_days * 23400.0
    sp = spread_distribution(
        r.events.meta["best_bid_tick"], r.events.meta["best_ask_tick"], r.events.t, burn
    )
    assert sp["status"] == "ok"
    assert 1 <= sp["median"] <= 6  # 20 日の短窓なのでゲート帯より緩く
    assert sp["n_nonpositive"] == 0
    dp = depth_profile(r.book, burn, cfg.tick_size)
    assert dp["status"] == "ok"
    assert not dp["peak_is_best"]


def test_interevent_is_poisson(book_result):
    from simchart.validation.micro import interevent_times

    r, cfg = book_result
    ev = r.events
    is_order = (ev.event_type == int(EventType.LIMIT_ADD)) | (
        ev.event_type == int(EventType.MARKET)
    )
    out = interevent_times(ev.t[is_order & (ev.t > 5 * 23400)], 0.0)
    assert out["status"] == "ok"
    assert 0.85 <= out["cv2"] <= 1.15


def test_placement_and_size_match_spec(book_result):
    from simchart.validation import run_all

    r, cfg = book_result
    m = run_all(r)["book"]
    assert m["placement"]["status"] == "ok"
    assert abs(m["placement"]["difference"]) <= 0.25  # 20 日窓なのでゲートより緩く
    assert m["order_size"]["status"] == "ok"
    assert m["order_size"]["max_abs_z"] < 5.0


def test_pstar_wiring_matches_price_process(book_result):
    """カーネル内の p* 補間が PriceProcess.at と一致する (§10 の配線検証)。"""
    r, _ = book_result
    ev = r.events
    sub = np.linspace(0, ev.t.size - 1, 200).astype(int)
    t_q = ev.t[sub]
    expected = r.price.at(np.minimum(t_q, r.price.t_end))
    recorded = ev.meta["log_pstar"][sub]
    np.testing.assert_allclose(recorded, expected, atol=1e-12)


def test_corr_mid_pstar_is_zero(book_result):
    from simchart.validation import run_all

    r, _ = book_result
    m = run_all(r)["book"]["corr_mid_pstar"]
    assert m["status"] == "ok"
    assert m["abs_z"] < 4.0


def test_trade_price_shows_bounce(book_result):
    """約定価格リターンに bid-ask bounce 由来の負の 1 次自己相関 (soft の実体確認)。"""
    r, _ = book_result
    px = np.asarray(r.events.meta["agg_trade_log_vwap"])
    rr = np.diff(px)
    d = rr - rr.mean()
    acf1 = float(d[:-1] @ d[1:] / (d @ d))
    assert acf1 < -0.1


def test_observation_is_mid_on_grid(book_result):
    r, cfg = book_result
    obs = r.observation
    assert obs.source == "l3.zi_book(mid)"
    assert obs.n_points == cfg.total_steps + 1
    assert np.all(np.isfinite(obs.log_price))


def test_book_snapshots_are_ordered(book_result):
    r, _ = book_result
    b = r.book
    assert not b.is_empty
    # 買いはレベルが下がる方向、売りは上がる方向 (NaN は許す)
    bd = np.diff(b.bid_px, axis=1)
    ad = np.diff(b.ask_px, axis=1)
    assert np.all(bd[np.isfinite(bd)] < 0)
    assert np.all(ad[np.isfinite(ad)] > 0)
    # スナップショットでも非クロス
    both = np.isfinite(b.bid_px[:, 0]) & np.isfinite(b.ask_px[:, 0])
    assert np.all(b.bid_px[both, 0] < b.ask_px[both, 0])


def test_zero_spread_never_recorded(book_result):
    r, _ = book_result
    ev = r.events
    bb = ev.meta["best_bid_tick"]
    ba = ev.meta["best_ask_tick"]
    both = (bb >= 0) & (ba >= 0)
    assert (ba[both] - bb[both]).min() >= 1


def test_book_params_guarded_when_flag_off():
    with pytest.raises(ValueError):
        Config(stage="S5", book_mu_mo=500.0, **S5KW, **SMALL)


def test_duplicate_yaml_keys_rejected(tmp_path):
    """YAML の重複キーは黙って後勝ちになる — 明示エラーにする (実際に踏んだ事故)。"""
    p = tmp_path / "dup.yaml"
    p.write_text("seed: 1\nseed: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重複"):
        Config.from_yaml(p)
