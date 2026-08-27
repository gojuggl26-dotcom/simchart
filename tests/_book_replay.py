"""イベントログの完全リプレイ検証 (S6/S7 共用ヘルパ)。

ログだけから板を再構成し、全 TRADE 行がその価格のキュー先頭を消費している
ことを検証する — FIFO (時間優先)・価格優先・部分約定・取消の全てを同時に縛る。
注文 id はプールスロットで再利用されるためid → 配置時刻の辞書では検証
できない (S6 の最初のテストはそれで自分が壊れていた) — リプレイは再利用と無関係。
"""

from __future__ import annotations

from collections import defaultdict, deque

from simchart.types import EventLog, EventType


def replay_and_verify(ev: EventLog, refill_tail: bool = True) -> int:
    """全イベントをリプレイして照合し、検証した TRADE 行数を返す。

    S8: アイスバーグ対応 — LIMIT_ADD 行はメタ列 ``iceberg_display`` が正なら
    表示量だけをキューに載せ、MODIFY(3) 行 (補充) で ``refill_tail`` に応じて
    末尾 (時間優先を失う) か先頭 (優先保持 — 表示切れで一旦 pop された分の復元)
    に再投入する。
    """
    queues: dict[float, deque] = defaultdict(deque)  # price -> deque[[oid, rem]]
    ice_display = ev.meta.get("iceberg_display")
    n_trades_checked = 0
    for i in range(ev.t.size):
        et = int(ev.event_type[i])
        oid = int(ev.order_id[i])
        px = float(ev.price[i])
        if et == int(EventType.LIMIT_ADD):
            rest = float(ev.size[i]) - float(ev.meta["exec_size"][i])
            if ice_display is not None and float(ice_display[i]) > 0:
                rest = min(rest, float(ice_display[i]))
            if oid >= 0 and rest > 0:
                queues[px].append([oid, rest])
        elif et == int(EventType.MODIFY):
            # アイスバーグ補充
            if refill_tail:
                queues[px].append([oid, float(ev.size[i])])
            else:
                queues[px].appendleft([oid, float(ev.size[i])])
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
    return n_trades_checked
