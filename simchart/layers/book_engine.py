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
  l3.order_price: LO ごとに 1 draw (配置距離)。S8 のアイスバーグ判定で
                  板に載る LO ごとに +1 draw (enable_iceberg 時のみ)
  l3.cancel     : 取消ごとに 1 draw (対象選択)
  l3.metaorder  : (S8) 到着ごとに 3 draw (間隔、符号、長さ)、空プール生成は
                  2 draw (符号、長さ)、成行ごとに 1 draw (ψ 混合) +
                  子なら 1 draw (プール選択) / ノイズなら 1 draw (符号)

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
#: S8 で追加。**行種別ごとに意味が変わる** (EV_PRICE/EV_OID と同じ流儀):
#:   MO / TRADE 行 : 攻撃側のメタオーダー記録行番号 (-1 = ノイズトレード)
#:   LIMIT_ADD 行  : アイスバーグの初期表示量 (>0)。非アイスバーグは -1
#:   MODIFY(3) 行  : アイスバーグ補充 (EV_SIZE=補充量, EV_EXEC=補充後の隠れ量)
#:   その他        : -1
EV_META = 12
N_EV_FIELDS = 13


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
# S7: Hawkes thinning の診断
C_H_CANDIDATES = 23  # thinning 候補数 (棄却込み)
C_H_REJECTED = 24  # 棄却数 (受理率 = 1 - rejected/candidates)
C_H_CAP_HITS = 25  # 強度上限ガードの発動数 (指示書 §5.3)
C_H_DAYCAP_HITS = 26  # 日次イベント上限ガードの発動数
C_CX_NOOP = 27  # 励起由来の取消が空板で空振りした数 (n の会計から漏れる分 — 記録)
# S8: メタオーダー・プールとアイスバーグの診断
C_META_CHILD = 28  # メタオーダーの子として出た成行の数
C_META_NOISE = 29  # ノイズトレード (1-ψ 側) の数
C_META_ARRIVALS = 30  # Poisson 到着で生成されたメタオーダー数
C_META_SPAWN_EMPTY = 31  # 空プール時の即時生成数 (供給の調整弁 — 頻度を記録)
C_META_COMPLETED = 32  # 完走したメタオーダー数
C_META_LOG_FULL = 33  # メタオーダー記録の容量到達 (0 であるべき)
C_META_POOL_FULL = 34  # アクティブ・プールの容量到達 (0 であるべき)
C_ICE_ORDERS = 35  # アイスバーグとして出た指値の数
C_ICE_REFILLS = 36  # 補充回数
C_ICE_REFILL_VOL = 37  # 補充量の合計 (隠れ→表示へ移った量)
C_ICE_HIDDEN_IN = 38  # 投入された隠れ量の合計 (台帳の照合用)
N_COUNTERS = 39


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
def uz_transform(mid_ticks, eta):
    """Robert–Rosenbaum 型 uncertainty zones の観測離散化 (S9 §8.2 の fallback)。

    刷り値 P [tick] は、ミッドが P から (0.5+η) tick を超えて離れたときだけ
    1 tick ずつ更新する (ヒステリシス)。更新直後のミッドは新しい P から
    ~(η−0.5) の位置にあり、反転には 2η・継続には 1 tick の移動で足りるため、
    η < 0.5 で交替過多 (負の自己相関) が生じる — R-R の η と同じ向き。
    ★常用しない: queue-reactive の較正で届かない場合の非常口 (README 記録必須)。
    """
    out = np.empty_like(mid_ticks)
    p = np.round(mid_ticks[0])
    thr = 0.5 + eta
    for i in range(mid_ticks.size):
        m = mid_ticks[i]
        while m - p > thr:
            p += 1.0
        while p - m > thr:
            p -= 1.0
        out[i] = p
    return out


@njit(cache=True)
def _meta_spawn(
    rng_meta, meta_alpha, meta_n_min, t, spawned_empty,
    m_sign, m_ntotal, m_nexec, m_t_created, m_spawned_empty,
    act_meta, n_meta, n_act, counters,
):
    """新規メタオーダーを 1 本生成してアクティブ集合へ。戻り値 (n_meta, n_act)。

    長さは N = floor(N_min·(1−u)^{-1/α}) — 離散裾 P(N ≥ n) = (N_min/n)^α が厳密で、
    符号 ACF の減衰指数 γ = α − 1 の理論対応がそのまま成立する (指示書 §4)。
    消費は常に 2 draw (符号 → 長さ)。容量到達時は生成せずカウンタに記録する
    (draw も消費しない — 失敗経路は決定論的に同一)。
    """
    if n_meta >= m_sign.shape[0]:
        counters[C_META_LOG_FULL] += 1.0
        return n_meta, n_act
    if n_act >= act_meta.shape[0]:
        counters[C_META_POOL_FULL] += 1.0
        return n_meta, n_act
    u_sign = rng_meta.random()
    u_len = rng_meta.random()
    mi = n_meta
    m_sign[mi] = 1.0 if u_sign < 0.5 else -1.0
    m_ntotal[mi] = np.floor(meta_n_min * (1.0 - u_len) ** (-1.0 / meta_alpha))
    m_nexec[mi] = 0.0
    m_t_created[mi] = t
    m_spawned_empty[mi] = spawned_empty
    act_meta[n_act] = mi
    return n_meta + 1, n_act + 1


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
    # --- S7: Hawkes (use_hawkes=False なら全て無視され、S6 経路はビット単位で不変) ---
    use_hawkes, rng_hawkes,
    h_a,  # 励起行列 3x3 (型レベル、両サイド合算の子孫数)
    h_beta_day,  # カーネル減衰率 [1/日] (K 成分)
    h_w,  # カーネル重み (合計 1)
    h_mu_mo, h_mu_lo,  # ベースライン [件/日/側]
    h_delta0,  # 取消の各注文独立ハザード [1/日] (ベースライン = φ·δ0·N)
    phi_lam_table,  # φ_λ(u) の格子 (u ∈ [0,1))
    phi_lam_max,
    h_cap,  # 総強度の絶対上限 [1/日]
    h_day_cap,  # 1 日あたり最大イベント数 (ログのみのガード)
    # --- S8: メタオーダー分割 (use_meta=False なら全て無視 — S6/S7 経路は不変) ---
    use_meta, rng_meta,
    meta_psi,  # 成行がメタオーダーの子である確率 (残りはノイズ 50/50)
    meta_lambda_day,  # Poisson 到着率 [件/日] (逐次モードでは 0 を渡す)
    meta_alpha, meta_n_min,  # Pareto 長: N = floor(N_min·u^{-1/α})
    meta_sequential,  # 逐次版 (§3.4 相互検証): 常に 1 本だけを最後まで実行
    meta_log_cap,  # メタオーダー記録の行数上限
    meta_pool_cap,  # 同時アクティブ数の上限
    # --- S8: アイスバーグ (use_ice=False なら無視) ---
    use_ice,
    ice_frac,  # 表示上限超の指値がアイスバーグになる確率
    ice_display,  # 表示量 [ロット]
    ice_refill_tail,  # True: 補充でキュー末尾へ / False: 時間優先を保持
    # --- S9: queue-reactive の意思決定層 (use_qr=False なら無視) ---
    # ★強度には触れない: Hawkes の時刻・種別・サイドを所与として
    # 「どこに置くか」「どれを取り消すか」だけを板の状態から決める (§3.2)。
    use_qr,
    qr_inspread_slope, qr_spread_ref, qr_inspread_cap,  # §5 配置 m(s)
    qr_cx_dist, qr_cx_w_floor, qr_cx_len_pow, qr_cx_back,  # §6 取消重み
    qr_mo_frac,  # 成行サイズのデプス上限 (0 = 無効)
    qr_obi_bias,  # ノイズ成行の OBI 符号バイアス (0 = 無効。§7.2)
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
    ord_hidden = np.zeros(max_orders, dtype=np.float64)  # S8: アイスバーグの隠れ量
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
    for j in range(ev_capacity):
        ev[EV_META, j] = -1.0  # 既定は「該当なし」(0 は正当な記録行番号なので不可)
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

    # S7: Hawkes の励起状態。h_states[y, k] = 励起先 y のカーネル成分 k の現在値
    # [1/日]。イベント (型 x) ごとに h_states[y, k] += a[x,y]·w[k]·β[k] で跳ね、
    # イベント間で e^{-β_k Δt} 減衰する (指数和 = Markov、更新 O(1) — §4.1)。
    n_kern = h_beta_day.shape[0]
    h_states = np.zeros((3, n_kern), dtype=np.float64)
    phi_n = phi_lam_table.shape[0]
    h_day_idx = -1
    h_day_events = 0

    # ------------------------------------------------------------------
    # S8: メタオーダー・プールの状態と記録。
    # 記録行 (m_*) は 1 メタオーダー 1 行で、生成〜最終子注文までの統計を持つ
    # (平方根則の I = 実行スパンのミッド変化, Q = 実行済み子数, V = 同期間の
    # 市場約定量の材料)。アクティブ集合は記録行番号の密配列 + swap-remove。
    m_sign = np.zeros(meta_log_cap, dtype=np.float64)
    m_ntotal = np.zeros(meta_log_cap, dtype=np.float64)
    m_nexec = np.zeros(meta_log_cap, dtype=np.float64)
    m_t_created = np.zeros(meta_log_cap, dtype=np.float64)
    m_t_first = np.full(meta_log_cap, -1.0, dtype=np.float64)
    m_t_last = np.full(meta_log_cap, -1.0, dtype=np.float64)
    m_mid_first = np.zeros(meta_log_cap, dtype=np.float64)  # 初子の**直前**ミッド
    m_mid_last = np.zeros(meta_log_cap, dtype=np.float64)  # 最終子の**直後**ミッド
    m_vol_first = np.zeros(meta_log_cap, dtype=np.float64)  # 初子直前の累積攻撃約定量
    m_vol_last = np.zeros(meta_log_cap, dtype=np.float64)  # 最終子直後の同
    m_own_vol = np.zeros(meta_log_cap, dtype=np.float64)  # 自身の子の約定量合計 (Q)
    m_spawned_empty = np.zeros(meta_log_cap, dtype=np.float64)  # 空プール生成なら 1
    n_meta = 0
    act_meta = np.empty(meta_pool_cap, dtype=np.int64)
    n_act = 0
    # プール占有のグリッド標本 (定常性ゲート用。mid_grid と同じ格子)
    pool_grid = np.zeros(n_grid, dtype=np.float64)
    # S9: 取消の重み付き選択のスクラッチ (生存注文数ぶんの重み)
    qr_w = np.zeros(max_orders, dtype=np.float64)
    # Poisson 到着スケジュール (逐次モードや到着率 0 では発火しない)
    next_meta_t = horizon_sec * 2.0
    if use_meta and (not meta_sequential) and meta_lambda_day > 0.0:
        next_meta_t = -np.log(1.0 - rng_meta.random()) / meta_lambda_day * day_sec

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------
    while True:
        # 次候補時刻の決定。
        # S6: 定数レートの合成 Poisson (厳密)。
        # S7: Ogata thinning の上界 λ̄ = φ_max·ベースライン + 現在の励起
        #     (励起はイベント間で単調減少、φ は大域最大で抑える → 有効な上界)。
        lam_total = 0.0
        lam_bar = 0.0
        if use_hawkes:
            exc_total = 0.0
            for y in range(3):
                for k in range(n_kern):
                    exc_total += h_states[y, k]
            lam_bar = (
                phi_lam_max * (2.0 * h_mu_mo + 2.0 * h_mu_lo + h_delta0 * n_live)
                + exc_total
            )
            if lam_bar > h_cap:
                lam_bar = h_cap
                counters[C_H_CAP_HITS] += 1.0
            u_dt = rng_hawkes.random()
            dt_days = -np.log(1.0 - u_dt) / lam_bar
        else:
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
            pool_grid[next_grid_idx] = n_act
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

        # S8: 期日の来た Poisson 到着を処理 (子の需要が発生する前に必ず追加)。
        # 生成失敗 (容量) でも次回時刻は進める — 詰まって無限ループしないため。
        while use_meta and next_meta_t <= t_now:
            nm2, na2 = _meta_spawn(
                rng_meta, meta_alpha, meta_n_min, next_meta_t, 0.0,
                m_sign, m_ntotal, m_nexec, m_t_created, m_spawned_empty,
                act_meta, n_meta, n_act, counters,
            )
            if nm2 > n_meta:
                counters[C_META_ARRIVALS] += 1.0
            n_meta = nm2
            n_act = na2
            next_meta_t += -np.log(1.0 - rng_meta.random()) / meta_lambda_day * day_sec

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

        # 種別の決定 (kind: 0=MO, 1=LO, 2=CX)
        kind = 0
        side = 1
        if use_hawkes:
            # --- S7: Ogata thinning の受理判定 ---
            # 励起状態を候補時刻まで減衰 (棄却でも時間は進むので必ず先に行う)
            for k in range(n_kern):
                dec = np.exp(-h_beta_day[k] * dt_days)
                for y in range(3):
                    h_states[y, k] *= dec
            # 日内位置 u と φ_λ(u) (季節性はベースラインのみ — §3.3)
            u_day = t_now / day_sec
            u_frac = u_day - int(u_day)
            pi = int(u_frac * phi_n)
            if pi >= phi_n:
                pi = phi_n - 1
            phi_now = phi_lam_table[pi]
            e_mo = 0.0
            e_lo = 0.0
            e_cx = 0.0
            for k in range(n_kern):
                e_mo += h_states[0, k]
                e_lo += h_states[1, k]
                e_cx += h_states[2, k]
            lam_mo_h = phi_now * 2.0 * h_mu_mo + e_mo
            lam_lo_h = phi_now * 2.0 * h_mu_lo + e_lo
            lam_cx_h = phi_now * h_delta0 * n_live + e_cx
            lam_tot_h = lam_mo_h + lam_lo_h + lam_cx_h
            counters[C_H_CANDIDATES] += 1.0
            u_acc = rng_hawkes.random()
            if u_acc * lam_bar >= lam_tot_h:
                counters[C_H_REJECTED] += 1.0
                continue
            # 日次イベント数ガード (記録のみ。記憶保護は ev_capacity が担う)
            di = int(u_day)
            if di != h_day_idx:
                h_day_idx = di
                h_day_events = 0
            h_day_events += 1
            if h_day_events > h_day_cap:
                counters[C_H_DAYCAP_HITS] += 1.0
            # 型レベル 3D + 独立な符号 (符号対称制約 §3.1 の等価表現)。
            # S8 (use_meta): 成行 (kind=0) の符号はメタオーダー機構が決めるので
            # ここでは引かない。指値の符号は従来どおり 50/50。
            u_kind = rng_hawkes.random() * lam_tot_h
            if u_kind < lam_mo_h:
                kind = 0
            elif u_kind < lam_mo_h + lam_lo_h:
                kind = 1
            else:
                kind = 2
            if kind == 1 or (kind == 0 and not use_meta):
                side = 1 if rng_hawkes.random() < 0.5 else -1
        else:
            # --- S6: 定数レート (乱数消費列をビット単位で維持) ---
            u_cat = rng_type.random() * lam_total
            # [0, mu) MO買い / [mu, 2mu) MO売り / [2mu, 2mu+a) LO買い /
            # [.., 2mu+2a) LO売り / 残り CX
            if u_cat < 2.0 * mu_mo:
                kind = 0
                side = 1 if u_cat < mu_mo else -1
            elif u_cat < 2.0 * mu_mo + 2.0 * alpha_lo:
                kind = 1
                side = 1 if u_cat < 2.0 * mu_mo + alpha_lo else -1
            else:
                kind = 2

        # ---------------- S8: 成行の符号決定 (メタオーダー・プール §3) ----------------
        # Hawkes が「いつ」を、ここが「どちらの符号か」**だけ**を決める (§3.1 の
        # 役割分離)。確率 ψ でプールから一様に選び、その符号の子注文を出す。
        # 残りはノイズトレード (iid 50/50)。プールが空なら需要駆動で即時生成する
        # (Poisson 供給 ρ<1 の不足分の調整弁 — 発生数はカウンタで監視)。
        cur_meta = -1
        if use_meta and kind == 0:
            u_mix = rng_meta.random()
            if u_mix < meta_psi:
                if n_act == 0:
                    nm2, na2 = _meta_spawn(
                        rng_meta, meta_alpha, meta_n_min, t_now, 1.0,
                        m_sign, m_ntotal, m_nexec, m_t_created, m_spawned_empty,
                        act_meta, n_meta, n_act, counters,
                    )
                    if na2 > n_act:
                        counters[C_META_SPAWN_EMPTY] += 1.0
                    n_meta = nm2
                    n_act = na2
                if n_act > 0:
                    pick = int(rng_meta.random() * n_act)
                    if pick >= n_act:
                        pick = n_act - 1
                    mi = act_meta[pick]
                    side = 1 if m_sign[mi] > 0.0 else -1
                    cur_meta = mi
                    counters[C_META_CHILD] += 1.0
                    if m_nexec[mi] == 0.0:
                        # 初子: 実行スパンの始点 (ミッドは**約定前**の値)
                        pb = best_bid if best_bid >= 0 else ref_bid
                        pa = best_ask if best_ask >= 0 else ref_ask
                        m_t_first[mi] = t_now
                        m_mid_first[mi] = 0.5 * (pb + pa)
                        m_vol_first[mi] = counters[C_VOL_AGGR]
                    m_nexec[mi] += 1.0
                    if m_nexec[mi] >= m_ntotal[mi]:
                        # 完走 → swap-remove (逐次モードは空になり、次の子需要で生成)
                        counters[C_META_COMPLETED] += 1.0
                        n_act -= 1
                        act_meta[pick] = act_meta[n_act]
                else:
                    # 生成失敗 (容量到達) → ノイズにフォールバック
                    side = 1 if rng_meta.random() < 0.5 else -1
                    counters[C_META_NOISE] += 1.0
            else:
                side = 1 if rng_meta.random() < 0.5 else -1
                counters[C_META_NOISE] += 1.0

        # ---------------- S9: OBI 符号バイアス (§7.2 — 既定は無効) ----------------
        # ★メタオーダーの子には触れない (⑪ の系譜を汚さない)。ノイズ成行のみ、
        # best レベルの不均衡 I に比例した確率で薄い側へ寄せる。使用時は on/off の
        # γ アブレーションが必須 (指示書 §7.2)。
        if use_qr and qr_obi_bias > 0.0 and kind == 0 and cur_meta < 0:
            if best_bid >= 0 and best_ask >= 0:
                qb = lv_vol[best_bid]
                qa = lv_vol[best_ask]
                tot_q = qb + qa
                if tot_q > 0.0:
                    imb = (qb - qa) / tot_q
                    if rng_meta.random() < qr_obi_bias * abs(imb):
                        side = 1 if imb > 0.0 else -1
        ev_row = n_events

        if kind == 0:
            # ---------------- 成行 ----------------
            size = _draw_size(rng_size, w_round, lot_cum, lot_vals, pareto_alpha)
            # S9: 利用可能デプスへのサイズ適応 (§3.2 表 — 既定は無効)。
            # ★有効化するとサイズ分布ゲート (仕様適合) と衝突するため、まず
            # 配置・取消の状態依存だけで届くかを測る方針 (config の注記)。
            if use_qr and qr_mo_frac > 0.0:
                ob_q = best_ask if side == 1 else best_bid
                if ob_q >= 0:
                    dep = _depth_within(
                        lv_vol, ob_q, 1 if side == 1 else -1,
                        depth_ticks, lo_tick, hi_tick,
                    )
                    cap_sz = qr_mo_frac * dep
                    if cap_sz < 1.0:
                        cap_sz = 1.0
                    if size > cap_sz:
                        size = np.floor(cap_sz)
            remaining = size
            counters[C_SUBMITTED_MO] += 1.0
            opp_best = best_ask if side == 1 else best_bid
            ev[EV_T, ev_row] = t_now
            ev[EV_TYPE, ev_row] = 2.0
            ev[EV_SIDE, ev_row] = side
            ev[EV_PRICE, ev_row] = opp_best if opp_best >= 0 else -1
            ev[EV_SIZE, ev_row] = size
            ev[EV_OID, ev_row] = -1
            ev[EV_META, ev_row] = cur_meta
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
                    # TRADE 行 (攻撃側のメタオーダー行番号も刻む — ⑪ の系譜追跡用)
                    tr = n_events
                    if tr < ev_capacity:
                        ev[EV_T, tr] = t_now
                        ev[EV_TYPE, tr] = 4.0
                        ev[EV_SIDE, tr] = side
                        ev[EV_PRICE, tr] = bt
                        ev[EV_SIZE, tr] = take
                        ev[EV_OID, tr] = head
                        ev[EV_META, tr] = cur_meta
                        n_events += 1
                    if ord_rem[head] <= 0.0 and use_ice and ord_hidden[head] > 0.0:
                        # S8: アイスバーグ補充 — 表示が尽きたら隠れ量から回す。
                        # MODIFY(3) 行として記録 (リプレイ検証が再構成に使う)。
                        refill = (
                            ice_display
                            if ord_hidden[head] > ice_display
                            else ord_hidden[head]
                        )
                        ord_hidden[head] -= refill
                        ord_rem[head] = refill
                        lv_vol[bt] += refill
                        counters[C_ICE_REFILLS] += 1.0
                        counters[C_ICE_REFILL_VOL] += refill
                        rr = n_events
                        if rr < ev_capacity:
                            ev[EV_T, rr] = t_now
                            ev[EV_TYPE, rr] = 3.0
                            ev[EV_SIDE, rr] = ord_side[head]
                            ev[EV_PRICE, rr] = bt
                            ev[EV_SIZE, rr] = refill
                            ev[EV_EXEC, rr] = ord_hidden[head]
                            ev[EV_OID, rr] = head
                            n_events += 1
                        if ice_refill_tail:
                            # 時間優先を失いキュー末尾へ (§6.1 の標準ルール)
                            nxt = ord_next[head]
                            lv_head[bt] = nxt
                            if nxt >= 0:
                                ord_prev[nxt] = -1
                            else:
                                lv_tail[bt] = -1
                            ord_next[head] = -1
                            ord_prev[head] = lv_tail[bt]
                            if lv_tail[bt] >= 0:
                                ord_next[lv_tail[bt]] = head
                            else:
                                lv_head[bt] = head
                            lv_tail[bt] = head
                            ord_seq[head] = seq
                            seq += 1
                            head = lv_head[bt]
                        # keep モード: 先頭のまま次の take を受ける
                    elif ord_rem[head] <= 0.0:
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
            # S8: 子注文の実行後スナップショット (最終子の統計は毎回上書きされ、
            # 完走しなかったメタオーダーでも「実行済み部分」の始終点が残る)
            if cur_meta >= 0:
                pb = best_bid if best_bid >= 0 else ref_bid
                pa = best_ask if best_ask >= 0 else ref_ask
                m_t_last[cur_meta] = t_now
                m_mid_last[cur_meta] = 0.5 * (pb + pa)
                m_vol_last[cur_meta] = counters[C_VOL_AGGR]
                m_own_vol[cur_meta] += size - remaining

        elif kind == 1:
            # ---------------- 指値 ----------------
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
            # in-spread 部分の重み和 (小さいので都度合算)。
            # S9 (§5): スプレッド依存の乗法変調 m(s) — スプレッドが広いほど
            # 内側への配置確率が上がる。これが平均回帰の主要な源で、S8 の
            # インパクト赤字を縮める本体。m(s)=1 (use_qr=False) では従来と
            # ビット単位で同一 (×1.0 は恒等)。
            m_s = 1.0
            if use_qr:
                m_s = 1.0 + qr_inspread_slope * (spread - qr_spread_ref)
                if m_s < 1.0:
                    m_s = 1.0
                elif m_s > qr_inspread_cap:
                    m_s = qr_inspread_cap
            wneg_total = 0.0
            for d in range(1, max_impr + 1):
                wneg_total += wneg[d - 1] * m_s
            u_place = rng_price.random() * (wneg_total + place_total_pos)
            if u_place < wneg_total:
                # improvement: best から d ティック内側
                acc = 0.0
                d_sel = 1
                for d in range(1, max_impr + 1):
                    acc += wneg[d - 1] * m_s
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
                    if ord_rem[head] <= 0.0 and use_ice and ord_hidden[head] > 0.0:
                        # S8: アイスバーグ補充 (成行側の補充ブロックと同一の規則)
                        refill = (
                            ice_display
                            if ord_hidden[head] > ice_display
                            else ord_hidden[head]
                        )
                        ord_hidden[head] -= refill
                        ord_rem[head] = refill
                        lv_vol[bt] += refill
                        counters[C_ICE_REFILLS] += 1.0
                        counters[C_ICE_REFILL_VOL] += refill
                        rr = n_events
                        if rr < ev_capacity:
                            ev[EV_T, rr] = t_now
                            ev[EV_TYPE, rr] = 3.0
                            ev[EV_SIDE, rr] = ord_side[head]
                            ev[EV_PRICE, rr] = bt
                            ev[EV_SIZE, rr] = refill
                            ev[EV_EXEC, rr] = ord_hidden[head]
                            ev[EV_OID, rr] = head
                            n_events += 1
                        if ice_refill_tail:
                            nxt = ord_next[head]
                            lv_head[bt] = nxt
                            if nxt >= 0:
                                ord_prev[nxt] = -1
                            else:
                                lv_tail[bt] = -1
                            ord_next[head] = -1
                            ord_prev[head] = lv_tail[bt]
                            if lv_tail[bt] >= 0:
                                ord_next[lv_tail[bt]] = head
                            else:
                                lv_head[bt] = head
                            lv_tail[bt] = head
                            ord_seq[head] = seq
                            seq += 1
                            head = lv_head[bt]
                    elif ord_rem[head] <= 0.0:
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
                # S8: アイスバーグ判定 — 表示上限を超える残量の一部を隠す。
                # 板 (lv_vol・デプス・スナップショット) は**表示量だけ**を見る。
                disp = remaining
                hidden = 0.0
                if (
                    use_ice
                    and remaining > ice_display
                    and rng_price.random() < ice_frac
                ):
                    disp = ice_display
                    hidden = remaining - disp
                    counters[C_ICE_ORDERS] += 1.0
                    counters[C_ICE_HIDDEN_IN] += hidden
                    ev[EV_META, ev_row] = disp  # LIMIT_ADD 行: 初期表示量 (リプレイ用)
                oid = free_stack[n_free - 1]
                n_free -= 1
                ord_price[oid] = tick
                ord_rem[oid] = disp
                ord_hidden[oid] = hidden
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
                lv_vol[tick] += disp
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
                # 生存注文が無ければ空振り (S6: レートが n_live 比例なのでほぼ来ない。
                # S7: 励起項は n_live=0 でも正になり得る → 空振り数を記録し、
                # その子孫は失われる — n の会計上の漏れとして README に明記)
                if use_hawkes:
                    counters[C_CX_NOOP] += 1.0
                continue
            u_pick = rng_cancel.random()
            if use_qr:
                # S9 (§6): 重み付き取消選択。w = exp(−dist·Δ)·L^p·(1 + back·b)。
                # 遠い注文は取り消されにくく、長い列・後方ほど取り消されやすい —
                # デプスのハンプを鋭くする (⑳)。後方度 b は同一レベルの
                # 先頭/末尾 seq からの相対位置で O(1) 近似 (rank 単調)。
                wtot = 0.0
                for j in range(n_live):
                    o = live_ids[j]
                    px_o = ord_price[o]
                    if ord_side[o] == 1:
                        ob2 = best_bid if best_bid >= 0 else ref_bid
                        dist = ob2 - px_o
                    else:
                        oa2 = best_ask if best_ask >= 0 else ref_ask
                        dist = px_o - oa2
                    if dist < 0:
                        dist = 0
                    lq = lv_cnt[px_o]
                    if lq < 1:
                        lq = 1
                    sh = ord_seq[lv_head[px_o]]
                    st = ord_seq[lv_tail[px_o]]
                    denom_b = st - sh + 1
                    b = (ord_seq[o] - sh) / denom_b
                    w = (
                        (qr_cx_w_floor + (1.0 - qr_cx_w_floor) * np.exp(-qr_cx_dist * dist))
                        * lq**qr_cx_len_pow
                        * (1.0 + qr_cx_back * b)
                    )
                    qr_w[j] = w
                    wtot += w
                target = u_pick * wtot
                acc_w = 0.0
                idx = n_live - 1
                for j in range(n_live):
                    acc_w += qr_w[j]
                    if target < acc_w:
                        idx = j
                        break
            else:
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
            # 台帳: アイスバーグの隠れ量も投入 (lo_in) に含めているので、
            # 取消では表示残 + 隠れ残の両方を戻す (数量保存の恒等式)。
            counters[C_VOL_CANCELLED] += rem + ord_hidden[oid]
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

        # ---------------- S7: 励起状態の跳ね (処理済みイベントのみ) ----------------
        # 型 x のイベントは h_states[y,k] += a[x,y]·w_k·β_k を加える。
        # ∫カーネル = a[x,y]·Σw = a[x,y] なので分岐比 n = ρ(a) が厳密に保たれる。
        if use_hawkes:
            for k in range(n_kern):
                jump = h_w[k] * h_beta_day[k]
                for y in range(3):
                    h_states[y, k] += h_a[kind, y] * jump

    # 終了処理
    counters[C_LIVE_ORDERS] = n_live
    total_live_vol = 0.0
    for i in range(n_live):
        total_live_vol += ord_rem[live_ids[i]] + ord_hidden[live_ids[i]]
    counters[C_LIVE_VOL] = total_live_vol
    # 残りのグリッド点を埋める
    cur_bb = best_bid if best_bid >= 0 else ref_bid
    cur_ba = best_ask if best_ask >= 0 else ref_ask
    cur_mid = 0.5 * (cur_bb + cur_ba)
    while next_grid_idx < n_grid:
        mid_grid[next_grid_idx] = cur_mid
        pool_grid[next_grid_idx] = n_act
        next_grid_idx += 1
    n_snaps = int(min(next_snap_t / snapshot_interval_sec, n_snap_cap))
    return (
        ev, n_events, mid_grid, snap_t, snap_px, snap_sz, n_snaps, counters,
        pool_grid,
        m_sign, m_ntotal, m_nexec, m_t_created, m_t_first, m_t_last,
        m_mid_first, m_mid_last, m_vol_first, m_vol_last, m_own_vol,
        m_spawned_empty,
        n_meta,
    )
