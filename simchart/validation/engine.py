"""S6: イベントエンジンの正当性検証 (指示書 §9)。

エンジン内の不変条件カウンタ (毎イベントの軽量チェック + 抜き取りの完全チェック) を
構造化して返す。**「あとで入れる」と必ず入らない**類のものなので、カーネル本体と
同時に実装されている — ここはその読み出し口。

保存則の設計 (初版で実際に間違えた点)
------------------------------------
「買い約定総量 = 売り約定総量」を**攻撃側の買い量 vs 売り量**と読むのは誤り。
どちらの側が攻撃するかは確率的で、等しくなる理由がない。1 つの約定には攻撃側と
受動側が必ず存在するので、正しい実装形は:

- **数量保存**: 攻撃側の約定量合計 (MO/LO の exec を発注側で集計) =
  受動側の約定量合計 (板から削られた take を約定側で集計)。2 つは別の経路で
  集計されるので、マッチングのバグ (部分約定の数え間違い等) で乖離する
- **注文保存**: 発注 = 板上 + 取消 + 受動完全約定 + 入口完全約定
- **指値量台帳**: 投入量 = 板上残 + 取消量 + 受動約定量 + 入口約定量
"""

from __future__ import annotations

import numpy as np

from .base import na, num, ok

__all__ = ["engine_invariants", "throughput"]


def engine_invariants(l3_meta: dict | None, events_t: np.ndarray | None = None) -> dict:
    """エンジンのカウンタから不変条件の成否をまとめる。"""
    if not l3_meta or "counters" not in l3_meta:
        return na("板エンジンの診断がありません (enable_book=False)")
    from ..layers.book_engine import (
        C_AGGRESSIVE_LO,
        C_CANCELLED,
        C_CROSS_VIOL,
        C_EMPTY_SIDE_TIME,
        C_FILLED_ORDERS,
        C_INV_FIFO_VIOL,
        C_INV_VOL_VIOL,
        C_LIVE_ORDERS,
        C_LIVE_VOL,
        C_LO_INSTANT,
        C_MO_REJECT_EVENTS,
        C_MO_REJECT_VOL,
        C_SUBMITTED_LO,
        C_SUBMITTED_MO,
        C_VOL_AGGR,
        C_VOL_CANCELLED,
        C_VOL_LO_ENTRY_EXEC,
        C_VOL_LO_IN,
        C_VOL_PASSIVE,
    )

    c = np.asarray(l3_meta["counters"], dtype=np.float64)
    order_lhs = c[C_SUBMITTED_LO]
    order_rhs = c[C_LIVE_ORDERS] + c[C_CANCELLED] + c[C_FILLED_ORDERS] + c[C_LO_INSTANT]
    vol_lhs = c[C_VOL_LO_IN]
    vol_rhs = (
        c[C_LIVE_VOL] + c[C_VOL_CANCELLED] + c[C_VOL_PASSIVE] + c[C_VOL_LO_ENTRY_EXEC]
    )
    monotone = True
    if events_t is not None and np.asarray(events_t).size > 1:
        monotone = bool(np.all(np.diff(np.asarray(events_t)) >= 0))

    checks = {
        "no_cross": c[C_CROSS_VIOL] == 0,
        "order_conservation": abs(order_lhs - order_rhs) < 0.5,
        "volume_conservation": abs(c[C_VOL_AGGR] - c[C_VOL_PASSIVE]) < 1e-6,
        "lo_volume_ledger": abs(vol_lhs - vol_rhs) < 1e-6,
        "fifo_priority": c[C_INV_FIFO_VIOL] == 0,
        "level_volume_consistency": c[C_INV_VOL_VIOL] == 0,
        "monotone_time": monotone,
    }
    return ok(
        bool(all(checks.values())),
        all_passed=bool(all(checks.values())),
        **{k: bool(v) for k, v in checks.items()},
        cross_violations=num(c[C_CROSS_VIOL]),
        fifo_violations=num(c[C_INV_FIFO_VIOL]),
        level_volume_violations=num(c[C_INV_VOL_VIOL]),
        order_ledger={
            "submitted": num(order_lhs), "live": num(c[C_LIVE_ORDERS]),
            "cancelled": num(c[C_CANCELLED]), "filled": num(c[C_FILLED_ORDERS]),
            "instant_filled": num(c[C_LO_INSTANT]),
        },
        volume_ledger={
            "lo_in": num(vol_lhs), "live": num(c[C_LIVE_VOL]),
            "cancelled": num(c[C_VOL_CANCELLED]), "passive_exec": num(c[C_VOL_PASSIVE]),
            "entry_exec": num(c[C_VOL_LO_ENTRY_EXEC]),
            "aggr_exec": num(c[C_VOL_AGGR]),
        },
        n_submitted_mo=num(c[C_SUBMITTED_MO]),
        n_aggressive_lo=num(c[C_AGGRESSIVE_LO]),
        mo_reject_events=num(c[C_MO_REJECT_EVENTS]),
        mo_reject_volume=num(c[C_MO_REJECT_VOL]),
        empty_side_time_sec=num(c[C_EMPTY_SIDE_TIME]),
    )


def throughput(l3_meta: dict | None) -> dict:
    """イベント処理速度 (指示書 §4: ゲート >= 50,000 events/sec)。

    測定は JIT ウォーム後 (ZIBook.observe が計測前に使い捨て乱数で 0.2 日の
    ウォームアップを走らせ、コンパイル/キャッシュロードを計測から外す)。
    ウォームアップ無しの初回はコンパイル込みで ~30k ev/s に見えるが、
    実体は ~10M ev/s (実測) — 測るものを間違えるとゲートの意味が変わる。
    """
    if not l3_meta or "throughput_events_per_sec" not in l3_meta:
        return na("板エンジンの診断がありません (enable_book=False)")
    tput = l3_meta.get("throughput_events_per_sec")
    return ok(
        num(tput),
        events_per_sec=num(tput),
        n_events=num(l3_meta.get("n_events")),
        engine_runtime_sec=num(l3_meta.get("engine_runtime_sec")),
        n_trades=num(l3_meta.get("n_trades")),
    )
