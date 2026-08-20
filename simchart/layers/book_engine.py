"""S6: ZI 板のイベント駆動エンジン (numba JIT カーネル)。

なぜ numba か (指示書 §4)
-------------------------
S10 は 5000 日 × 10 シードの結合実行で ~5,000 万イベント/シードになる。純 Python の
イベントループは 3〜8k events/sec で 100 時間規模 — 目標 ≥ 50k events/sec のため
ホットループを numba nopython で書く。**S10 で作り直しにならないよう、ここで性能を
確保しておくのが S6 の設計要件**。

乱数の扱い
----------
numpy の ``Generator`` を njit にそのまま渡す。numba は Generator の状態を
**in-place で前進**させるので (実測で確認済み: njit 内の draw 列は numpy と同一、
消費後の状態も外側から可視)、RNG レジストリの会計 (フィンガープリント・ストリーム
独立性) がそのまま成立する。カーネル内では ``.random()`` (一様) だけを使い、
指数・べき則・離散は逆関数法で組み立てる — サポート面の最も固い口に限定するため。

消費順 (決定論の前提):
  l3.order_type : イベントごとに 2 draw (時間間隔、種別+サイド)
  l3.order_size : 注文 (MO/LO) ごとに 2 draw (混合の枝、値)
  l3.order_price: LO ごとに 1 draw (配置距離)
  l3.cancel     : 取消ごとに 1 draw (対象選択)

板の表現
--------
- 価格レベル: **絶対ティック整数を添字とする配列** (窓 ``2*half_ticks``、中心 p0)。
  SortedDict 等は使わない (オーバーヘッド、指示書 §5.1)
- 各レベル: 連結リストによる FIFO キュー (order_next/prev + level head/tail)
- 取消の O(1) 一様抽選: 生存注文の密な配列 + 位置索引 (swap-remove)
- best_bid/best_ask はインクリメンタル追跡。片側枯渇時は**直近の有効 best を記憶**
  し、新規指値はそれを基準に配置。空の側への成行は棄却してログ (指示書 §8.2)

ZI ミッドはランダムウォークするので窓の端に近づき得る。**黙って詰まらず**、
オーバーフローのフラグを立てて停止する (500 日の SD ~1,600 ティック vs 窓 ±65,536
なので実際には起きない — 起きたら設定の見直しが必要な異常)。

イベント種別コード (EventLog の EventType と対応):
  0 = LIMIT_ADD / 1 = CANCEL / 2 = MARKET / 4 = TRADE (約定ごとに 1 行、価格つき)
成行の未充足棄却は MARKET 行の exec < size として現れ、棄却量は診断カウンタにも出る。
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["run_zi_book", "EngineOutput", "N_EV_FIELDS"]

# イベントログの列 (SoA)。
EV_T = 0  # 時刻 [秒]
EV_TYPE = 1  # 0=LO, 1=CX, 2=MO, 4=TRADE
EV_SIDE = 2  # +1 買い / -1 売り (TRADE は攻撃側)
EV_PRICE = 3  # ティック (LO=配置、TRADE=約定、MO=攻撃直前の反対 best、CX=対象)
EV_SIZE = 4  # 注文サイズ (TRADE は約定サイズ)
EV_EXEC = 5  # 即時約定した量 (LO/MO)。TRADE 行では 0
EV_BB = 6  # イベント後の best bid ティック (-1 = 空)
EV_BA = 7  # イベント後の best ask ティック (-1 = 空)
EV_DBID = 8  # best から depth_ticks 以内の買いデプス (ロット)
EV_DASK = 9  # 同 売り
EV_PSTAR = 10  # log p*(t) — κ=0 でも配線して記録する (指示書 §10)
EV_OID = 11  # 注文 id (TRADE は受動側注文の id)
N_EV_FIELDS = 12


class EngineOutput:
    """カーネルの出力をまとめる入れ物 (Python 側)。"""

    def __init__(self, ev: np.ndarray, n_events: int, mids: np.ndarray,
                 snap_t: np.ndarray, snap_px: np.ndarray, snap_sz: np.ndarray,
                 n_snaps: int, counters: np.ndarray, runtime_sec: float) -> None:
        self.events = ev[:, :n_events]
        self.n_events = n_events
        self.mid_grid = mids
        self.snap_t = snap_t[:n_snaps]
        self.snap_px = snap_px[:n_snaps]
        self.snap_sz = snap_sz[:n_snaps]
        self.counters = counters
        self.runtime_sec = runtime_sec


# counters の添字
C_SUBMITTED_LO = 0
C_SUBMITTED_MO = 1
C_CANCELLED = 2
C_FILLED_ORDERS = 3  # 完全約定した指値の数
#: ★「買い約定総量 = 売り約定総量」の実装形。1 つの約定には攻撃側と受動側が必ず
#: 存在するので、**攻撃側の約定量 (別経路で集計) と受動側の約定量 (take ごとに
#: 集計) の一致**が数量保存になる。攻撃側の買い量と売り量を比べるのは誤り —
#: どちらの側が攻撃するかは確率的で、等しくなる理由がない (初版で実際に間違えた)。
C_VOL_AGGR = 4  # 攻撃側 (MO + aggressive LO) の約定量
C_VOL_PASSIVE = 5  # 受動側 (板上の注文) から削られた量
C_MO_REJECT_EVENTS = 6  # 反対側枯渇による棄却 (イベント数)
C_MO_REJECT_VOL = 7  # 同 (量)
C_CROSS_VIOL = 8  # best_bid >= best_ask の検出数 (0 であるべき)
C_EMPTY_SIDE_STEPS = 9  # 片側が空だったイベント数 (時間比率の分子は別で計測)
C_EMPTY_SIDE_TIME = 10  # 片側が空だった時間 [秒]
C_WINDOW_OVERFLOW = 11  # 板窓からの逸脱 (0 であるべき)
C_ORDER_POOL_FULL = 12  # 注文プール枯渇 (0 であるべき)
C_LOG_FULL = 13  # ログ容量到達 (0 であるべき)
C_INV_FIFO_VIOL = 14  # 抜き取り検証: FIFO 順序違反
C_INV_VOL_VIOL = 15  # 抜き取り検証: レベル総量の不一致
C_LIVE_ORDERS = 16  # 終了時の板上注文数
C_LIVE_VOL = 17  # 終了時の板上総量
C_AGGRESSIVE_LO = 18  # 反対 best を跨いだ指値 (クロスした LO)
C_LO_INSTANT = 19  # 板に載らず全量即時約定した指値 (台帳の保存則に必要)
C_VOL_LO_IN = 20  # 指値 (初期化込み) の投入総量
C_VOL_CANCELLED = 21  # 取消された量
C_VOL_LO_ENTRY_EXEC = 22  # aggressive LO が入り口で約定した量 (板に載らなかった分)
N_COUNTERS = 23


@njit(cache=True)
def _draw_size(rng_size, w_round, lot_cum, lot_vals, pareto_alpha):
    """サイズの混合分布 (ロット単位、>=1 の整数値)。消費は常に 2 draw。"""
    u_branch = rng_size.random()
    u_val = rng_size.random()
    if u_branch < w_round:
        for i in range(lot_cum.shape[0]):
            if u_val < lot_cum[i]:
                return lot_vals[i]
        return lot_vals[lot_cum.shape[0] - 1]
    # Pareto(alpha, xm=1) を切り上げてロット化 (>= 1)。
    v = (1.0 - u_val) ** (-1.0 / pareto_alpha)
    s = np.ceil(v)
    return s


@njit(cache=True)
def _refresh_best(lv_vol, start, direction, lo, hi):
    """start から direction 方向に最初の非空レベルを探す。無ければ -1。"""
    t = start
    while lo <= t <= hi:
        if lv_vol[t] > 0.0:
            return t
        t += direction
    return -1


@njit(cache=True)
def _depth_within(lv_vol, best, direction, n_ticks, lo, hi):
    """best から n_ticks 以内の総量 (best を含む)。best<0 なら 0。"""
    if best < 0:
        return 0.0
    s = 0.0
    for k in range(n_ticks):
        t = best + direction * k
        if t < lo or t > hi:
            break
        s += lv_vol[t]
    return s


@njit(cache=True)
def _check_level(lv_head, ord_next, ord_seq, ord_rem, lv_vol, tick):
    """1 レベルの FIFO 整合 (seq 昇順) と総量整合。戻り値 (fifo_ok, vol_ok)。"""
    i = lv_head[tick]
    prev_seq = -1
    total = 0.0
    while i >= 0:
        if ord_seq[i] <= prev_seq:
            return False, True
        prev_seq = ord_seq[i]
        total += ord_rem[i]
        i = ord_next[i]
    return True, abs(total - lv_vol[tick]) < 1e-6


@njit(cache=True)
def run_zi_book(
    # --- 乱数ストリーム (in-place で前進する) ---
    rng_type, rng_size, rng_price, rng_cancel,
    # --- 期間・レート ---
    n_days, session_seconds,
    mu_mo, alpha_lo, delta_cancel,  # [1/日]
    # --- 配置分布 ---
    place_cum,  # Δ=0..max の累積重み (正規化なし、単調増加)
    wneg,  # d=1..cap の in-spread 重み (単品)
    allow_inspread,
    # --- サイズ分布 ---
    w_round, lot_cum, lot_vals, pareto_alpha,
    # --- 板の初期化・窓 ---
    p0_tick, init_levels, init_size, half_ticks,
    # --- p* 配線 (κ=0 でも参照して記録する) ---
    log_pstar, pstar_step_sec,
    # --- 記録 ---
    depth_ticks, snapshot_interval_sec, snap_levels,
    mid_grid_step_sec,
    # --- 容量 ---
    max_orders, ev_capacity,
    # --- 検証 ---
    debug_invariants, invariant_stride,
):
    """ZI 板を最後まで走らせ、(イベントログ, グリッドミッド, スナップショット,
    カウンタ) を返す。単一スレッド・固定消費順で決定論。"""
    horizon_sec = n_days * session_seconds
    lo_tick = 0
    hi_tick = 2 * half_ticks
    # レベル配列
    n_ticks_arr = hi_tick + 1
    lv_head = np.full(n_ticks_arr, -1, dtype=np.int64)
    lv_tail = np.full(n_ticks_arr, -1, dtype=np.int64)
    lv_vol = np.zeros(n_ticks_arr, dtype=np.float64)
    lv_cnt = np.zeros(n_ticks_arr, dtype=np.int64)
    # 注文プール
    ord_price = np.zeros(max_orders, dtype=np.int64)
    ord_rem = np.zeros(max_orders, dtype=np.float64)
    ord_side = np.zeros(max_orders, dtype=np.int8)
    ord_next = np.full(max_orders, -1, dtype=np.int64)
    ord_prev = np.full(max_orders, -1, dtype=np.int64)
    ord_seq = np.zeros(max_orders, dtype=np.int64)
    free_stack = np.empty(max_orders, dtype=np.int64)
    for i in range(max_orders):
        free_stack[i] = max_orders - 1 - i
    n_free = max_orders
    # 生存注文の密配列 (取消の O(1) 一様抽選)
    live_ids = np.empty(max_orders, dtype=np.int64)
    live_pos = np.full(max_orders, -1, dtype=np.int64)
    n_live = 0
    # 出力
    ev = np.zeros((N_EV_FIELDS, ev_capacity), dtype=np.float64)
    n_grid = int(round(horizon_sec / mid_grid_step_sec)) + 1
    mid_grid = np.zeros(n_grid, dtype=np.float64)
    n_snap_cap = int(horizon_sec / snapshot_interval_sec) + 2
    snap_t = np.zeros(n_snap_cap, dtype=np.float64)
    snap_px = np.zeros((n_snap_cap, 2 * snap_levels), dtype=np.int64)
    snap_sz = np.zeros((n_snap_cap, 2 * snap_levels), dtype=np.float64)
    counters = np.zeros(N_COUNTERS, dtype=np.float64)

    best_bid = -1
    best_ask = -1
    ref_bid = p0_tick - 1  # 直近の有効 best (枯渇時の配置基準 §8.2)
    ref_ask = p0_tick + 1
    seq = 0
    n_events = 0
    global_oid = 0

    # --- 初期化: best±init_levels に種注文 (t=0 の LIMIT_ADD としてログ) ---
    t_now = 0.0
    pstar_n = log_pstar.shape[0]

    for side_i in range(2):
        side = 1 if side_i == 0 else -1
        for k in range(init_levels):
            tick = (p0_tick - 1 - k) if side == 1 else (p0_tick + 1 + k)
            oid = free_stack[n_free - 1]
            n_free -= 1
            ord_price[oid] = tick
            ord_rem[oid] = init_size
            ord_side[oid] = side
            ord_seq[oid] = seq
            seq += 1
            ord_next[oid] = -1
            ord_prev[oid] = lv_tail[tick]
            if lv_tail[tick] >= 0:
                ord_next[lv_tail[tick]] = oid
            else:
                lv_head[tick] = oid
            lv_tail[tick] = oid
            lv_vol[tick] += init_size
            lv_cnt[tick] += 1
            live_ids[n_live] = oid
            live_pos[oid] = n_live
            n_live += 1
            counters[C_SUBMITTED_LO] += 1.0
            counters[C_VOL_LO_IN] += init_size
            global_oid += 1
            if n_events < ev_capacity:
                ev[EV_T, n_events] = 0.0
                ev[EV_TYPE, n_events] = 0.0
                ev[EV_SIDE, n_events] = side
                ev[EV_PRICE, n_events] = tick
                ev[EV_SIZE, n_events] = init_size
                ev[EV_EXEC, n_events] = 0.0
                ev[EV_OID, n_events] = oid
                n_events += 1
    best_bid = p0_tick - 1
    best_ask = p0_tick + 1
    # 初期イベント行の best/デプス/p* を埋める
    p0_star = log_pstar[0]
    for j in range(n_events):
        ev[EV_BB, j] = best_bid
        ev[EV_BA, j] = best_ask
        ev[EV_DBID, j] = _depth_within(lv_vol, best_bid, -1, depth_ticks, lo_tick, hi_tick)
        ev[EV_DASK, j] = _depth_within(lv_vol, best_ask, 1, depth_ticks, lo_tick, hi_tick)
        ev[EV_PSTAR, j] = p0_star

    next_snap_t = 0.0
    next_grid_idx = 0
    place_total_pos = place_cum[place_cum.shape[0] - 1]
    inspread_cap = wneg.shape[0]

    day_sec = session_seconds

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------
    while True:
        # 総強度 [1/日] と次イベント時刻
        lam_total = 2.0 * mu_mo + 2.0 * alpha_lo + delta_cancel * n_live
        u_dt = rng_type.random()
        dt_days = -np.log(1.0 - u_dt) / lam_total
        t_next = t_now + dt_days * day_sec

        # グリッドミッドとスナップショットを t_next まで進める
        cur_bb = best_bid if best_bid >= 0 else ref_bid
        cur_ba = best_ask if best_ask >= 0 else ref_ask
        cur_mid = 0.5 * (cur_bb + cur_ba)
        while next_grid_idx < n_grid and next_grid_idx * mid_grid_step_sec <= t_next:
            mid_grid[next_grid_idx] = cur_mid
            next_grid_idx += 1
        while next_snap_t <= t_next and next_snap_t <= horizon_sec:
            si = int(next_snap_t / snapshot_interval_sec + 0.5)
            if si < n_snap_cap:
                snap_t[si] = next_snap_t
                # 買い側: best から下へ snap_levels 個の非空レベル
                t = best_bid
                for k in range(snap_levels):
                    tk = _refresh_best(lv_vol, t, -1, lo_tick, hi_tick) if t >= 0 else -1
                    if tk < 0:
                        snap_px[si, k] = -1
                        snap_sz[si, k] = 0.0
                    else:
                        snap_px[si, k] = tk
                        snap_sz[si, k] = lv_vol[tk]
                        t = tk - 1
                t = best_ask
                for k in range(snap_levels):
                    tk = _refresh_best(lv_vol, t, 1, lo_tick, hi_tick) if t >= 0 else -1
                    if tk < 0:
                        snap_px[si, snap_levels + k] = -1
                        snap_sz[si, snap_levels + k] = 0.0
                    else:
                        snap_px[si, snap_levels + k] = tk
                        snap_sz[si, snap_levels + k] = lv_vol[tk]
                        t = tk + 1
            next_snap_t += snapshot_interval_sec

        if t_next > horizon_sec:
            t_now = horizon_sec
            break
        # 片側枯渇時間の計測
        if best_bid < 0 or best_ask < 0:
            counters[C_EMPTY_SIDE_TIME] += t_next - t_now
        t_now = t_next

        if n_events + 8 >= ev_capacity:
            counters[C_LOG_FULL] += 1.0
            break

        # p* 参照 (κ=0 でも配線 §10)。等間隔グリッドの線形補間。
        pos = t_now / pstar_step_sec
        i0 = int(pos)
        if i0 >= pstar_n - 1:
            pstar_val = log_pstar[pstar_n - 1]
        else:
            frac = pos - i0
            pstar_val = log_pstar[i0] * (1.0 - frac) + log_pstar[i0 + 1] * frac

        # 種別の決定
        u_cat = rng_type.random() * lam_total
        # [0, mu) MO買い / [mu, 2mu) MO売り / [2mu, 2mu+a) LO買い /
        # [.., 2mu+2a) LO売り / 残り CX
        ev_row = n_events

        if u_cat < 2.0 * mu_mo:
            # ---------------- 成行 ----------------
            side = 1 if u_cat < mu_mo else -1
            size = _draw_size(rng_size, w_round, lot_cum, lot_vals, pareto_alpha)
            remaining = size
            counters[C_SUBMITTED_MO] += 1.0
            opp_best = best_ask if side == 1 else best_bid
            ev[EV_T, ev_row] = t_now
            ev[EV_TYPE, ev_row] = 2.0
            ev[EV_SIDE, ev_row] = side
            ev[EV_PRICE, ev_row] = opp_best if opp_best >= 0 else -1
            ev[EV_SIZE, ev_row] = size
            ev[EV_OID, ev_row] = -1
            n_events += 1
            # マッチング (価格優先・時間優先)
            direction = 1 if side == 1 else -1  # 買いは ask を上へ、売りは bid を下へ
            while remaining > 0.0:
                bt = best_ask if side == 1 else best_bid
                if bt < 0:
                    # 反対側枯渇 → 残量棄却。★remaining をゼロにして抜けてはならない:
                    # exec = size - remaining の計算が棄却分まで約定に数えてしまい、
                    # 数量保存 (攻撃側 = 受動側) が破れる (意地悪テストで実際に検出)。
                    counters[C_MO_REJECT_EVENTS] += 1.0
                    counters[C_MO_REJECT_VOL] += remaining
                    break
                head = lv_head[bt]
                while head >= 0 and remaining > 0.0:
                    take = ord_rem[head] if ord_rem[head] < remaining else remaining
                    ord_rem[head] -= take
                    remaining -= take
                    lv_vol[bt] -= take
                    counters[C_VOL_PASSIVE] += take
                    # TRADE 行
                    tr = n_events
                    if tr < ev_capacity:
                        ev[EV_T, tr] = t_now
                        ev[EV_TYPE, tr] = 4.0
                        ev[EV_SIDE, tr] = side
                        ev[EV_PRICE, tr] = bt
                        ev[EV_SIZE, tr] = take
                        ev[EV_OID, tr] = head
                        n_events += 1
                    if ord_rem[head] <= 0.0:
                        # 完全約定 → キューから除去
                        nxt = ord_next[head]
                        lv_head[bt] = nxt
                        if nxt >= 0:
                            ord_prev[nxt] = -1
                        else:
                            lv_tail[bt] = -1
                        lv_cnt[bt] -= 1
                        # 生存リストから除去
                        p = live_pos[head]
                        last = live_ids[n_live - 1]
                        live_ids[p] = last
                        live_pos[last] = p
                        n_live -= 1
                        live_pos[head] = -1
                        free_stack[n_free] = head
                        n_free += 1
                        counters[C_FILLED_ORDERS] += 1.0
                        head = lv_head[bt]
                    else:
                        break
                if lv_vol[bt] <= 0.0:
                    lv_vol[bt] = 0.0
                    # best の更新 (次の非空レベルへ)
                    if side == 1:
                        nb = _refresh_best(lv_vol, bt + 1, 1, lo_tick, hi_tick)
                        if nb >= 0:
                            ref_ask = nb
                        best_ask = nb
                    else:
                        nb = _refresh_best(lv_vol, bt - 1, -1, lo_tick, hi_tick)
                        if nb >= 0:
                            ref_bid = nb
                        best_bid = nb
            ev[EV_EXEC, ev_row] = size - remaining
            counters[C_VOL_AGGR] += size - remaining

        elif u_cat < 2.0 * mu_mo + 2.0 * alpha_lo:
            # ---------------- 指値 ----------------
            side = 1 if u_cat < 2.0 * mu_mo + alpha_lo else -1
            size = _draw_size(rng_size, w_round, lot_cum, lot_vals, pareto_alpha)
            counters[C_SUBMITTED_LO] += 1.0
            counters[C_VOL_LO_IN] += size
            # 配置基準 (枯渇時は記憶した best)
            base = best_bid if side == 1 else best_ask
            if base < 0:
                base = ref_bid if side == 1 else ref_ask
            # スプレッド (improvement の許容幅)
            sb = best_bid if best_bid >= 0 else ref_bid
            sa = best_ask if best_ask >= 0 else ref_ask
            spread = sa - sb
            max_impr = spread - 1
            if max_impr > inspread_cap:
                max_impr = inspread_cap
            if not allow_inspread:
                max_impr = 0
            # in-spread 部分の重み和 (小さいので都度合算)
            wneg_total = 0.0
            for d in range(1, max_impr + 1):
                wneg_total += wneg[d - 1]
            u_place = rng_price.random() * (wneg_total + place_total_pos)
            if u_place < wneg_total:
                # improvement: best から d ティック内側
                acc = 0.0
                d_sel = 1
                for d in range(1, max_impr + 1):
                    acc += wneg[d - 1]
                    if u_place < acc:
                        d_sel = d
                        break
                tick = base + d_sel if side == 1 else base - d_sel
            else:
                # 板内: best から Δ ティック外側 (累積表を二分探索)
                u2 = u_place - wneg_total
                lo_i = 0
                hi_i = place_cum.shape[0] - 1
                while lo_i < hi_i:
                    mid_i = (lo_i + hi_i) // 2
                    if place_cum[mid_i] > u2:
                        hi_i = mid_i
                    else:
                        lo_i = mid_i + 1
                delta = lo_i
                tick = base - delta if side == 1 else base + delta
            if tick <= lo_tick or tick >= hi_tick:
                counters[C_WINDOW_OVERFLOW] += 1.0
                break
            ev[EV_T, ev_row] = t_now
            ev[EV_TYPE, ev_row] = 0.0
            ev[EV_SIDE, ev_row] = side
            ev[EV_PRICE, ev_row] = tick
            ev[EV_SIZE, ev_row] = size
            n_events += 1
            remaining = size
            # aggressive limit: 反対 best を跨いだら価格制限つきで即時約定 (§5.2)
            crossed = False
            while remaining > 0.0:
                opp = best_ask if side == 1 else best_bid
                if opp < 0:
                    break
                if (side == 1 and tick < opp) or (side == -1 and tick > opp):
                    break
                crossed = True
                bt = opp
                head = lv_head[bt]
                while head >= 0 and remaining > 0.0:
                    take = ord_rem[head] if ord_rem[head] < remaining else remaining
                    ord_rem[head] -= take
                    remaining -= take
                    lv_vol[bt] -= take
                    counters[C_VOL_PASSIVE] += take
                    tr = n_events
                    if tr < ev_capacity:
                        ev[EV_T, tr] = t_now
                        ev[EV_TYPE, tr] = 4.0
                        ev[EV_SIDE, tr] = side
                        ev[EV_PRICE, tr] = bt
                        ev[EV_SIZE, tr] = take
                        ev[EV_OID, tr] = head
                        n_events += 1
                    if ord_rem[head] <= 0.0:
                        nxt = ord_next[head]
                        lv_head[bt] = nxt
                        if nxt >= 0:
                            ord_prev[nxt] = -1
                        else:
                            lv_tail[bt] = -1
                        lv_cnt[bt] -= 1
                        p = live_pos[head]
                        last = live_ids[n_live - 1]
                        live_ids[p] = last
                        live_pos[last] = p
                        n_live -= 1
                        live_pos[head] = -1
                        free_stack[n_free] = head
                        n_free += 1
                        counters[C_FILLED_ORDERS] += 1.0
                        head = lv_head[bt]
                    else:
                        break
                if lv_vol[bt] <= 0.0:
                    lv_vol[bt] = 0.0
                    if side == 1:
                        nb = _refresh_best(lv_vol, bt + 1, 1, lo_tick, hi_tick)
                        if nb >= 0:
                            ref_ask = nb
                        best_ask = nb
                    else:
                        nb = _refresh_best(lv_vol, bt - 1, -1, lo_tick, hi_tick)
                        if nb >= 0:
                            ref_bid = nb
                        best_bid = nb
            if crossed:
                counters[C_AGGRESSIVE_LO] += 1.0
            ev[EV_EXEC, ev_row] = size - remaining
            counters[C_VOL_AGGR] += size - remaining
            counters[C_VOL_LO_ENTRY_EXEC] += size - remaining
            if remaining > 0.0:
                # 残量を板に載せる
                if n_free == 0:
                    counters[C_ORDER_POOL_FULL] += 1.0
                    break
                oid = free_stack[n_free - 1]
                n_free -= 1
                ord_price[oid] = tick
                ord_rem[oid] = remaining
                ord_side[oid] = side
                ord_seq[oid] = seq
                seq += 1
                ord_next[oid] = -1
                ord_prev[oid] = lv_tail[tick]
                if lv_tail[tick] >= 0:
                    ord_next[lv_tail[tick]] = oid
                else:
                    lv_head[tick] = oid
                lv_tail[tick] = oid
                lv_vol[tick] += remaining
                lv_cnt[tick] += 1
                live_ids[n_live] = oid
                live_pos[oid] = n_live
                n_live += 1
                ev[EV_OID, ev_row] = oid
                # best 更新
                if side == 1:
                    if best_bid < 0 or tick > best_bid:
                        best_bid = tick
                        ref_bid = tick
                else:
                    if best_ask < 0 or tick < best_ask:
                        best_ask = tick
                        ref_ask = tick
            else:
                ev[EV_OID, ev_row] = -1
                counters[C_LO_INSTANT] += 1.0  # 板に載らず完結 (保存則の項)

        else:
            # ---------------- 取消 ----------------
            if n_live == 0:
                # 生存注文が無ければ空振り (レートが n_live 比例なのでほぼ来ない)
                continue
            u_pick = rng_cancel.random()
            idx = int(u_pick * n_live)
            if idx >= n_live:
                idx = n_live - 1
            oid = live_ids[idx]
            tick = ord_price[oid]
            side = ord_side[oid]
            rem = ord_rem[oid]
            # キューから除去
            prv = ord_prev[oid]
            nxt = ord_next[oid]
            if prv >= 0:
                ord_next[prv] = nxt
            else:
                lv_head[tick] = nxt
            if nxt >= 0:
                ord_prev[nxt] = prv
            else:
                lv_tail[tick] = prv
            lv_vol[tick] -= rem
            if lv_vol[tick] < 1e-12:
                lv_vol[tick] = 0.0
            lv_cnt[tick] -= 1
            last = live_ids[n_live - 1]
            live_ids[idx] = last
            live_pos[last] = idx
            n_live -= 1
            live_pos[oid] = -1
            free_stack[n_free] = oid
            n_free += 1
            counters[C_CANCELLED] += 1.0
            counters[C_VOL_CANCELLED] += rem
            # best 更新
            if lv_vol[tick] == 0.0:
                if side == 1 and tick == best_bid:
                    nb = _refresh_best(lv_vol, tick - 1, -1, lo_tick, hi_tick)
                    if nb >= 0:
                        ref_bid = nb
                    best_bid = nb
                elif side == -1 and tick == best_ask:
                    nb = _refresh_best(lv_vol, tick + 1, 1, lo_tick, hi_tick)
                    if nb >= 0:
                        ref_ask = nb
                    best_ask = nb
            ev[EV_T, ev_row] = t_now
            ev[EV_TYPE, ev_row] = 1.0
            ev[EV_SIDE, ev_row] = side
            ev[EV_PRICE, ev_row] = tick
            ev[EV_SIZE, ev_row] = rem
            ev[EV_EXEC, ev_row] = 0.0
            ev[EV_OID, ev_row] = oid
            n_events += 1

        # ---------------- イベント後の共通記録 ----------------
        db = _depth_within(lv_vol, best_bid, -1, depth_ticks, lo_tick, hi_tick)
        da = _depth_within(lv_vol, best_ask, 1, depth_ticks, lo_tick, hi_tick)
        for j in range(ev_row, n_events):
            ev[EV_BB, j] = best_bid
            ev[EV_BA, j] = best_ask
            ev[EV_DBID, j] = db
            ev[EV_DASK, j] = da
            ev[EV_PSTAR, j] = pstar_val

        # ---------------- 不変条件 ----------------
        if best_bid >= 0 and best_ask >= 0 and best_bid >= best_ask:
            counters[C_CROSS_VIOL] += 1.0
        if best_bid < 0 or best_ask < 0:
            counters[C_EMPTY_SIDE_STEPS] += 1.0
        if debug_invariants or (n_events % invariant_stride) < 3:
            # 抜き取り: best 近傍レベルの FIFO と総量整合
            for chk in range(2):
                bt = best_bid if chk == 0 else best_ask
                if bt >= 0:
                    fifo_ok, vol_ok = _check_level(
                        lv_head, ord_next, ord_seq, ord_rem, lv_vol, bt
                    )
                    if not fifo_ok:
                        counters[C_INV_FIFO_VIOL] += 1.0
                    if not vol_ok:
                        counters[C_INV_VOL_VIOL] += 1.0

    # 終了処理
    counters[C_LIVE_ORDERS] = n_live
    total_live_vol = 0.0
    for i in range(n_live):
        total_live_vol += ord_rem[live_ids[i]]
    counters[C_LIVE_VOL] = total_live_vol
    # 残りのグリッド点を埋める
    cur_bb = best_bid if best_bid >= 0 else ref_bid
    cur_ba = best_ask if best_ask >= 0 else ref_ask
    cur_mid = 0.5 * (cur_bb + cur_ba)
    while next_grid_idx < n_grid:
        mid_grid[next_grid_idx] = cur_mid
        next_grid_idx += 1
    n_snaps = int(min(next_snap_t / snapshot_interval_sec, n_snap_cap))
    return ev, n_events, mid_grid, snap_t, snap_px, snap_sz, n_snaps, counters
