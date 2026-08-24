"""L3: 板層 (S0 ではスタブ)。

最終系での役割
--------------
メタオーダー分割 -> 6 次元 Hawkes 注文流 -> queue-reactive 板 -> uncertainty zones
による離散化。**観測価格は板のミッド**であり、p*(t) は外生生成された潜在情報価格に
すぎない。注文流の一部が p* 方向にバイアスを持つ (ハイブリッド方式) ことで、
情報が価格に染み込む速度そのものが内生的に決まる。結合強度が ``kappa`` (S10)。

S0 での実装
-----------
恒等写像。observed price = p*(t) をそのまま返し、イベントも板も空。
"""

from __future__ import annotations

import time

import numpy as np

from ..config import Config
from ..rng import RNGRegistry
from ..types import BookSnapshot, EventLog, EventType, Observation, PriceProcess
from .l0_calendar import ConstantCalendar
from .l1_activity import ConstantActivity

__all__ = ["PassThroughBook", "ZIBook", "build_book_layer"]


class PassThroughBook:
    """板を持たず、潜在情報価格をそのまま観測価格として返す。"""

    name = "l3.passthrough"

    def __init__(self, config: Config, calendar: ConstantCalendar) -> None:
        self._config = config
        self._calendar = calendar

    def observe(
        self,
        price: PriceProcess,
        calendar: ConstantCalendar | None = None,
        activity: ConstantActivity | None = None,
    ) -> tuple[Observation, EventLog, BookSnapshot]:
        """観測系列・イベント列・板スナップショットを返す。

        S0 では観測時刻は L2 のグリッドと同一で、値は log p* そのもの。配列は
        コピーせず共有する (両方とも不変オブジェクトなので安全)。

        S6 以降、観測時刻は L2 のグリッドから切り離され、板イベントの時刻になる。
        そのとき L2 の値は ``price.at(event_times)`` で引く。**呼び出し側が
        「観測 = グリッド」を前提にしないよう、S0 の時点から
        :class:`~simchart.types.Observation` に時刻を明示的に持たせている。**
        """
        del activity
        cal = calendar or self._calendar
        observation = Observation(
            t=price.t,
            log_price=price.log_p_star,
            session_seconds=cal.session_seconds(),
            step_seconds=cal.step_seconds(),
            source="l3.passthrough(p_star)",
        )
        events = EventLog.empty(reason="S0 では注文流を生成しない (S6 で板層を導入)")
        book = BookSnapshot.empty(reason="S0 では板を持たない (S6 で板層を導入)")
        return observation, events, book


class ZIBook:
    """S6: zero-intelligence 板 (Smith et al. 2003 ベースライン、κ=0)。

    観測価格は**板のミッド**になり、L2 の p* は κ=0 で切り離されている。
    この段階の価格に ①③④⑧⑯⑱ は現れない — それが正しい状態 (指示書 §0)。
    p* は各イベントで参照して記録だけする (§10: S10 の結合が 1 行の変更で済み、
    補間参照の性能をここで実測でき、corr(Δmid, Δp*) ≈ 0 が結合判定のベースラインになる)。
    """

    name = "l3.zi_book"

    def __init__(
        self,
        config: Config,
        rng: RNGRegistry,
        calendar: ConstantCalendar,
        activity: ConstantActivity | None = None,
    ) -> None:
        self._config = config
        self._rng = rng
        self._calendar = calendar
        self._activity = activity
        self.last_diagnostics: dict = {}

    # ------------------------------------------------------------------
    def _placement_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """配置べき則の累積表 (板内 Δ>=0) と in-spread 重み (d=1..cap)。

        P(Δ) ∝ (Δ + Δ0)^-(1+μ)。**最大距離で打ち切る** (裾を切らないと遠方に
        無駄な注文が溜まる — 指示書 §6.2)。in-spread 側は同じ式の距離 |Δ| を使い、
        実際の許容幅 (spread-1 と cap の小さい方) はイベント時にカーネル側で絞る。
        """
        cfg = self._config
        delta = np.arange(cfg.book_max_place_ticks + 1, dtype=np.float64)
        w = (delta + cfg.book_place_offset) ** (-(1.0 + cfg.book_mu_place))
        place_cum = np.cumsum(w)
        d = np.arange(1, cfg.book_inspread_cap + 1, dtype=np.float64)
        wneg = (d + cfg.book_place_offset) ** (-(1.0 + cfg.book_mu_place))
        return place_cum, wneg

    def observe(
        self,
        price: PriceProcess,
        calendar: ConstantCalendar | None = None,
        activity: ConstantActivity | None = None,
    ) -> tuple[Observation, EventLog, BookSnapshot]:
        from .book_engine import (
            C_LOG_FULL,
            C_ORDER_POOL_FULL,
            C_WINDOW_OVERFLOW,
            EV_BA,
            EV_BB,
            EV_DASK,
            EV_DBID,
            EV_EXEC,
            EV_META,
            EV_OID,
            EV_PRICE,
            EV_PSTAR,
            EV_SIDE,
            EV_SIZE,
            EV_T,
            EV_TYPE,
            run_zi_book,
        )

        cfg = self._config
        cal = calendar or self._calendar
        session = cal.session_seconds()
        step_sec = cal.step_seconds()

        place_cum, wneg = self._placement_tables()
        lot_probs = np.asarray(cfg.book_lot_probs, dtype=np.float64)
        lot_cum = np.cumsum(lot_probs)
        lot_vals = np.asarray(cfg.book_lot_values, dtype=np.float64)

        p0_tick = cfg.book_window_half_ticks  # 窓の中心 = p0
        # 容量見積: 到着 (MO+LO) + 取消 (定常 N ~ 2α/δ) + TRADE 行 + 余白。
        n_res = 2.0 * cfg.book_alpha_lo / cfg.book_delta_cancel

        # ---------------- S7: Hawkes 引数の構築 ----------------
        use_hawkes = bool(cfg.enable_hawkes)
        if use_hawkes:
            act = activity if activity is not None else self._activity
            if act is None or not hasattr(act, "betas_per_day"):
                raise TypeError(
                    "enable_hawkes には HawkesActivity が必要です"
                    " (build_book_layer 経由で構築すること)。"
                )
            h_a = act.matrix()
            h_beta_day = act.betas_per_day()
            h_w = act.weights()
            h_mu_mo = float(cfg.hawkes_mu_mo)
            h_mu_lo = float(cfg.hawkes_mu_lo)
            h_delta0 = float(cfg.hawkes_delta0)
            # φ_λ(u) の格子 (季節性はベースラインのみ — §3.3)。thinning の上界は
            # カーネル側も同じ格子を引くので、格子の最大値が厳密な上界になる。
            m_phi = 4096
            u_grid = (np.arange(m_phi, dtype=np.float64) + 0.5) / m_phi
            if hasattr(cal, "phi_lambda_of_u"):
                phi_tab = np.asarray(cal.phi_lambda_of_u(u_grid), dtype=np.float64)
            else:
                phi_tab = np.ones(m_phi, dtype=np.float64)
            phi_max = float(phi_tab.max())
            # 定常レートと容量。総数の期待値は ∫φ=1 より季節性で変わらない。
            r_vec = act.stationary_rates()
            rate_per_day = float(r_vec.sum() + 2.5 * r_vec[0])
            # クラスタリングで日単位は過分散になるが全期間の総数の相対 SD は
            # 微小 (Fano ~ (1-n)^-2 でも 500 日合計では ≪1%) — 1.6 倍で足りる。
            ev_capacity = int(cfg.n_days * rate_per_day * 1.6) + 100_000
            # 強度上限 [1/日]: 定常総レートの cap_mult 倍 (§5.3 バーストガード)。
            h_cap = float(cfg.hawkes_intensity_cap_mult * r_vec.sum())
            h_day_cap = int(cfg.hawkes_daily_event_cap)
            rng_hawkes = self._rng.get("l1.hawkes")
        else:
            rate_per_day = (
                2.0 * (cfg.book_mu_mo + cfg.book_alpha_lo)
                + cfg.book_delta_cancel * n_res
                + 2.5 * cfg.book_mu_mo  # TRADE 行 (1 約定 = 平均 ~2 レベル強)
            )
            ev_capacity = int(cfg.n_days * rate_per_day * 1.5) + 100_000
            # ダミー (use_hawkes=False ではカーネルが一切読まない)
            h_a = np.zeros((3, 3), dtype=np.float64)
            h_beta_day = np.ones(1, dtype=np.float64)
            h_w = np.ones(1, dtype=np.float64)
            h_mu_mo = 0.0
            h_mu_lo = 0.0
            h_delta0 = 0.0
            phi_tab = np.ones(8, dtype=np.float64)
            phi_max = 1.0
            h_cap = 1.0
            h_day_cap = 1
            # ★S6 経路のビット単位不変: レジストリのストリームは取得だけでも
            # 名前ハッシュ独立なので他ストリームに影響しないが、紛れを避けるため
            # 使い捨ての Generator を渡す (カーネルは一度も呼ばない)。
            rng_hawkes = np.random.default_rng(0)
        max_orders = int(n_res * 20) + 50_000

        # ---------------- S8: メタオーダーとアイスバーグの引数 ----------------
        use_meta = bool(cfg.enable_metaorder)
        use_ice = bool(cfg.enable_iceberg)
        if use_meta:
            from scipy.special import zeta as _zeta

            # 成行の総レート [件/日]: Hawkes の定常解、または S6 の定数レート
            if use_hawkes:
                lam_mo_total = float(r_vec[0])
            else:
                lam_mo_total = 2.0 * cfg.book_mu_mo
            # E[N] は**実際のサンプラー** N = floor(N_min·(1−u)^{-1/α}) の期待値。
            # ★連続 Pareto の α·N_min/(α−1) を使ってはならない (N_min=1, α=1.6 で
            # 2.67 vs 実サンプラー ζ(1.6)=2.286 — 15% ずれて釣り合いが狂う)。
            # E[N] = Σ_{n≥1} P(N≥n)、P(N≥n) = min(1, (N_min/n)^α)。
            nm = int(cfg.meta_n_min)
            alpha = float(cfg.meta_alpha)
            partial = sum(k ** (-alpha) for k in range(1, nm + 1))
            e_len = nm + (nm**alpha) * (float(_zeta(alpha)) - partial)
            # 子の需要 ψ·λ_MO に対し供給 λ_meta·E[N] を ρ (< 1) 倍で与える。
            # ★ρ=1 の厳密な釣り合いは臨界負荷 (E[N²]=∞) でプール占有が
            # さまよい定常にならない — 不足分は空プール時の即時生成が埋める。
            meta_lambda_day = (
                0.0
                if cfg.meta_sequential
                else float(cfg.meta_supply_ratio) * float(cfg.meta_psi)
                * lam_mo_total / e_len
            )
            meta_log_cap = int(cfg.n_days * cfg.meta_psi * lam_mo_total) + 50_000
            meta_args = dict(
                psi=float(cfg.meta_psi), lam=meta_lambda_day, alpha=alpha,
                n_min=float(nm), sequential=bool(cfg.meta_sequential),
                log_cap=meta_log_cap, pool_cap=int(cfg.meta_pool_cap),
                e_len=e_len, lam_mo_total=lam_mo_total,
            )
            rng_meta = self._rng.get("l3.metaorder")
        else:
            meta_args = dict(
                psi=0.0, lam=0.0, alpha=1.5, n_min=1.0, sequential=False,
                log_cap=8, pool_cap=8, e_len=0.0, lam_mo_total=0.0,
            )
            rng_meta = np.random.default_rng(0)
        if use_ice:
            # MODIFY (補充) 行のぶんイベントログの余白を広げる
            ev_capacity = int(ev_capacity * 1.2)

        # ---------------- S9: queue-reactive (意思決定層のみ — §3.2) ----------------
        use_qr = bool(cfg.enable_queue_reactive)

        # ---------------- S10: κ 結合の引数 ----------------
        # s = σ_t·√τ_meta [log 価格単位]。σ_t は L2 の瞬間ボラ (年率) —
        # 年率 → 「τ_meta 秒あたり」へ √(τ / (252·session)) で換算する。
        tick_f = float(cfg.tick_size)
        base_price_f = float(cfg.p0 - p0_tick * tick_f)
        kappa_f = float(cfg.kappa)
        if kappa_f > 0.0:
            s_scale_grid = np.exp(price.log_vol) * np.sqrt(
                float(cfg.kappa_tau_meta_sec) / (252.0 * float(session))
            )
        else:
            s_scale_grid = np.ones(2, dtype=np.float64)

        # ---------------- S10c: c_vol (緩慢ボラ → 活動度 Z_t) ----------------
        # Z_t = exp(c·V_t − c²/2)、V_t = (MA_w(log σ_obs) − m_V)/s_V。
        # - MA は後方窓 (因果)。窓 = 丸 w 日ぶんなので log φ_σ(u) は日周期の
        #   定数 (セッション平均) に落ち、rough (日未満の記憶) もほぼ消える。
        # - m_V は決定論的定数 (価格層の composition 診断)。全標本平均での
        #   標準化はルックアヘッドなので使わない。s_V も固定定数 (較正実測値)。
        # - Z は 3 日 MA 由来で緩慢 → 独自の粗グリッド (〜60s) で持つ
        #   (p* グリッド解像度で持つと本番 936MB×2 の無駄)。
        use_cvol = float(cfg.c_vol) > 0.0
        if use_cvol:
            spd_full = int(round(session / step_sec))
            stride_z = max(1, int(round(60.0 / step_sec)))
            while spd_full % stride_z != 0:
                stride_z -= 1
            z_step_f = stride_z * float(step_sec)
            spd_z = spd_full // stride_z
            lv_z = np.ascontiguousarray(price.log_vol[::stride_z], dtype=np.float64)
            w_z = max(1, int(round(float(cfg.c_vol_ma_days) * spd_z)))
            csum = np.concatenate(([0.0], np.cumsum(lv_z)))
            n_z = lv_z.size
            hi = np.arange(1, n_z + 1)
            lo = np.maximum(hi - w_z, 0)
            ma_z = (csum[hi] - csum[lo]) / (hi - lo)
            m_v = price.mean_log_vol_deterministic
            if m_v is None:
                raise ValueError(
                    "c_vol > 0 には価格層が mean_log_vol_deterministic を"
                    " 記録している必要があります (l2_price が現行版か確認)"
                )
            v_z = (ma_z - float(m_v)) / float(cfg.c_vol_v_scale)
            cvf = float(cfg.c_vol)
            z_grid = np.exp(cvf * v_z - 0.5 * cvf * cvf)
            # thinning 上界: 4h ブロック max の前方 2 ブロック引き。カーネルは
            # 有効域 (自+次ブロック) を越える候補を打ち切るので厳密な上界。
            z_blk = max(1, int(round(4.0 * 3600.0 / z_step_f)))
            n_blk = (n_z + z_blk - 1) // z_blk
            zpad = np.full(n_blk * z_blk, -np.inf)
            zpad[:n_z] = z_grid
            bmax = zpad.reshape(n_blk, z_blk).max(axis=1)
            z_up_grid = np.repeat(
                np.maximum(bmax, np.concatenate((bmax[1:], bmax[-1:]))), z_blk
            )[:n_z]
            # 容量: レートは Z に比例して膨らむ。Z 経路は実行前に確定している
            # ので、実現平均 Z で見積りを補正する (max は 2 ブロック上界の余裕)。
            z_mean_realized = float(z_grid.mean())
            cap_mult_z = min(4.0, max(1.0, 1.25 * z_mean_realized))
            ev_capacity = int(ev_capacity * cap_mult_z)
            if use_meta:
                meta_args["log_cap"] = int(meta_args["log_cap"] * cap_mult_z)
            cvol_diag = {
                "m_v": float(m_v),
                "v_mean": float(v_z.mean()),
                "v_sd": float(v_z.std()),
                "z_mean": float(z_grid.mean()),
                "z_min": float(z_grid.min()),
                "z_max": float(z_grid.max()),
                "ma_window_days": w_z / spd_z,
                "z_step_sec": z_step_f,
            }
        else:
            z_grid = np.ones(2, dtype=np.float64)
            z_up_grid = np.ones(2, dtype=np.float64)
            z_step_f = 60.0
            z_blk = 240
            cvol_diag = None

        # JIT ウォームアップ (コンパイル / キャッシュロードを計測から外す)。
        # ★使い捨ての Generator を使う — レジストリのストリームを消費すると
        # 決定論が壊れる。出力は捨てる。
        _warm = [np.random.default_rng(i) for i in range(6)]
        run_zi_book(
            _warm[0], _warm[1], _warm[2], _warm[3],
            0.2, float(session),
            float(cfg.book_mu_mo), float(cfg.book_alpha_lo), float(cfg.book_delta_cancel),
            place_cum, wneg, bool(cfg.book_allow_inspread),
            float(cfg.book_size_round_weight), lot_cum, lot_vals,
            float(cfg.book_size_pareto_alpha),
            int(p0_tick), int(cfg.book_init_levels), float(cfg.book_init_size),
            int(cfg.book_window_half_ticks),
            price.log_p_star[:2].copy(), float(step_sec),
            int(cfg.book_depth_ticks), float(cfg.book_snapshot_interval_sec),
            int(cfg.book_snapshot_levels),
            float(step_sec),
            100_000, 100_000,
            False, 50_000,
            use_hawkes, _warm[4],
            h_a, h_beta_day, h_w, h_mu_mo, h_mu_lo, h_delta0,
            phi_tab, phi_max, h_cap, h_day_cap,
            use_meta, _warm[5],
            meta_args["psi"], meta_args["lam"], meta_args["alpha"],
            meta_args["n_min"], meta_args["sequential"],
            10_000, 1_000,
            use_ice, float(cfg.book_iceberg_frac),
            float(cfg.book_iceberg_display_lots), bool(cfg.book_iceberg_refill_tail),
            use_qr,
            float(cfg.qr_inspread_slope), float(cfg.qr_spread_ref),
            float(cfg.qr_inspread_cap), float(cfg.qr_inspread_flat),
            float(cfg.qr_cx_dist_decay), float(cfg.qr_cx_w_floor),
            float(cfg.qr_cx_len_pow), float(cfg.qr_cx_back),
            float(cfg.qr_mo_depth_frac), float(cfg.qr_obi_bias),
            kappa_f, s_scale_grid[:2].copy(), base_price_f, tick_f,
            use_cvol, np.ones(2, dtype=np.float64), np.ones(2, dtype=np.float64),
            60.0, 240,
        )

        started = time.perf_counter()
        (
            ev, n_events, mid_grid, snap_t, snap_px, snap_sz, n_snaps, counters,
            pool_grid,
            m_sign, m_ntotal, m_nexec, m_t_created, m_t_first, m_t_last,
            m_mid_first, m_mid_last, m_vol_first, m_vol_last, m_own_vol,
            m_spawned_empty,
            n_meta,
        ) = run_zi_book(
            self._rng.get("l3.order_type"),
            self._rng.get("l3.order_size"),
            self._rng.get("l3.order_price"),
            self._rng.get("l3.cancel"),
            float(cfg.n_days), float(session),
            float(cfg.book_mu_mo), float(cfg.book_alpha_lo), float(cfg.book_delta_cancel),
            place_cum, wneg, bool(cfg.book_allow_inspread),
            float(cfg.book_size_round_weight), lot_cum, lot_vals,
            float(cfg.book_size_pareto_alpha),
            int(p0_tick), int(cfg.book_init_levels), float(cfg.book_init_size),
            int(cfg.book_window_half_ticks),
            price.log_p_star, float(step_sec),
            int(cfg.book_depth_ticks), float(cfg.book_snapshot_interval_sec),
            int(cfg.book_snapshot_levels),
            float(step_sec),
            int(max_orders), int(ev_capacity),
            bool(cfg.book_debug_invariants), 50_000,
            use_hawkes, rng_hawkes,
            h_a, h_beta_day, h_w, h_mu_mo, h_mu_lo, h_delta0,
            phi_tab, phi_max, h_cap, h_day_cap,
            use_meta, rng_meta,
            meta_args["psi"], meta_args["lam"], meta_args["alpha"],
            meta_args["n_min"], meta_args["sequential"],
            meta_args["log_cap"], meta_args["pool_cap"],
            use_ice, float(cfg.book_iceberg_frac),
            float(cfg.book_iceberg_display_lots), bool(cfg.book_iceberg_refill_tail),
            use_qr,
            float(cfg.qr_inspread_slope), float(cfg.qr_spread_ref),
            float(cfg.qr_inspread_cap), float(cfg.qr_inspread_flat),
            float(cfg.qr_cx_dist_decay), float(cfg.qr_cx_w_floor),
            float(cfg.qr_cx_len_pow), float(cfg.qr_cx_back),
            float(cfg.qr_mo_depth_frac), float(cfg.qr_obi_bias),
            kappa_f, s_scale_grid, base_price_f, tick_f,
            use_cvol, z_grid, z_up_grid, float(z_step_f), int(z_blk),
        )
        engine_runtime = time.perf_counter() - started

        # 容量系の失敗は黙って続けない (結果が静かに欠損する)。
        if counters[C_LOG_FULL] > 0 or counters[C_ORDER_POOL_FULL] > 0:
            raise RuntimeError(
                f"板エンジンの容量が不足しました (log_full={counters[C_LOG_FULL]:.0f},"
                f" pool_full={counters[C_ORDER_POOL_FULL]:.0f})。容量見積を見直すこと。"
            )
        if counters[C_WINDOW_OVERFLOW] > 0:
            raise RuntimeError(
                "ZI ミッドが板の絶対ティック窓から逸脱しました。"
                " book_window_half_ticks を広げるか期間を見直すこと。"
            )
        if use_meta:
            from .book_engine import C_META_LOG_FULL, C_META_POOL_FULL

            if counters[C_META_LOG_FULL] > 0 or counters[C_META_POOL_FULL] > 0:
                raise RuntimeError(
                    f"メタオーダーの容量が不足しました"
                    f" (log_full={counters[C_META_LOG_FULL]:.0f},"
                    f" pool_full={counters[C_META_POOL_FULL]:.0f})。"
                    f" プールが発散していないか (meta_supply_ratio > 1 相当の"
                    f" 釣り合い崩れ) を先に疑うこと。"
                )

        tick = cfg.tick_size
        base_price = cfg.p0 - p0_tick * tick  # 絶対ティック → 価格

        # --- Observation: グリッド上のミッド (対数価格) ---
        obs_source = "l3.zi_book(mid)"
        obs_mid_ticks = mid_grid
        if cfg.enable_uncertainty_zones:
            # S9 §8.2 の fallback。観測系列だけを離散化し、板・イベントログは
            # 生のまま (UZ は「刷り値」の層であって板の動学ではない)。
            from .book_engine import uz_transform

            obs_mid_ticks = uz_transform(
                np.ascontiguousarray(mid_grid, dtype=np.float64), float(cfg.uz_eta)
            )
            obs_source = "l3.zi_book(mid+uz)"
        mid_px = base_price + obs_mid_ticks * tick
        t_grid = np.arange(mid_px.shape[0], dtype=np.float64) * step_sec
        observation = Observation(
            t=t_grid,
            log_price=np.log(mid_px),
            session_seconds=session,
            step_seconds=step_sec,
            source=obs_source,
        )

        # --- EventLog ---
        t_arr = ev[EV_T, :n_events].copy()
        etype = ev[EV_TYPE, :n_events].astype(np.int8)
        # エンジン: 0=LO,1=CX,2=MO,4=TRADE — EventType と同じ値に揃えてある。
        side = ev[EV_SIDE, :n_events].astype(np.int8)
        px_ticks = ev[EV_PRICE, :n_events]
        px = np.where(px_ticks >= 0, base_price + px_ticks * tick, np.nan)
        # S8: EV_META は行種別で意味が変わる (book_engine の定義参照)。
        # agent_id には**成行/約定行のメタオーダー行番号**を移す (ノイズ・非該当 -1)。
        # LIMIT_ADD 行の値 (アイスバーグ表示量) は別キーへ分離する。
        meta_col = ev[EV_META, :n_events]
        is_mo_or_trade = (etype == int(EventType.MARKET)) | (
            etype == int(EventType.TRADE)
        )
        agent = np.where(is_mo_or_trade, meta_col, -1.0).astype(np.int64)
        ice_display_col = np.where(
            etype == int(EventType.LIMIT_ADD), meta_col, -1.0
        )
        events = EventLog(
            t=t_arr,
            event_type=etype,
            side=side,
            price=px,
            size=ev[EV_SIZE, :n_events].copy(),
            order_id=ev[EV_OID, :n_events].astype(np.int64),
            agent_id=agent,
            meta={
                "exec_size": ev[EV_EXEC, :n_events].copy(),
                "best_bid_tick": ev[EV_BB, :n_events].astype(np.int64),
                "best_ask_tick": ev[EV_BA, :n_events].astype(np.int64),
                "depth_bid": ev[EV_DBID, :n_events].copy(),
                "depth_ask": ev[EV_DASK, :n_events].copy(),
                "log_pstar": ev[EV_PSTAR, :n_events].copy(),
                "iceberg_display": ice_display_col,
                "tick_size": tick,
                "base_price": base_price,
            },
        )
        if use_meta:
            events.meta["pool_grid"] = pool_grid
            events.meta["metaorders"] = {
                "sign": m_sign[:n_meta].copy(),
                "n_total": m_ntotal[:n_meta].copy(),
                "n_exec": m_nexec[:n_meta].copy(),
                "t_created": m_t_created[:n_meta].copy(),
                "t_first": m_t_first[:n_meta].copy(),
                "t_last": m_t_last[:n_meta].copy(),
                "mid_first": m_mid_first[:n_meta].copy(),
                "mid_last": m_mid_last[:n_meta].copy(),
                "vol_first": m_vol_first[:n_meta].copy(),
                "vol_last": m_vol_last[:n_meta].copy(),
                "own_vol": m_own_vol[:n_meta].copy(),
                "spawned_empty": m_spawned_empty[:n_meta].copy(),
            }

        # --- BookSnapshot (top-K レベル、-1 = 空)。★n_snaps で必ず切る —
        # 容量いっぱいの配列をそのまま渡すと末尾のゼロ行が「価格 = 窓下端」の
        # 偽スナップショットとして混入する (実際にテストで検出した)。---
        k = cfg.book_snapshot_levels
        snap_px_v = snap_px[:n_snaps]
        snap_sz_v = snap_sz[:n_snaps]
        bid_px_t = snap_px_v[:, :k].astype(np.float64)
        ask_px_t = snap_px_v[:, k:].astype(np.float64)
        book = BookSnapshot(
            t=snap_t[:n_snaps].copy(),
            bid_px=np.where(bid_px_t >= 0, base_price + bid_px_t * tick, np.nan),
            bid_sz=snap_sz_v[:, :k].copy(),
            ask_px=np.where(ask_px_t >= 0, base_price + ask_px_t * tick, np.nan),
            ask_sz=snap_sz_v[:, k:].copy(),
            meta={"tick_size": tick, "levels": k},
        )

        # --- 攻撃注文単位に集約した約定系列 ---
        # ★TRADE 行のまま符号 ACF を測ってはならない: 1 本の成行が複数レベルを
        # 掃くと同符号の行が連続し、機械的な正の自己相関 (実測 +0.38) が出る。
        # これは注文流の性質ではなく記録粒度の人工物。⑪ (符号 ACF) と propagator は
        # **攻撃注文 1 本 = 1 観測** の系列で測る (同一攻撃注文の約定は同時刻なので
        # 時刻の変化で束ねられる)。
        tr_mask = etype == int(EventType.TRADE)
        tr_t = t_arr[tr_mask]
        if tr_t.size:
            starts = np.flatnonzero(np.concatenate([[True], np.diff(tr_t) > 0]))
            tr_sz = events.size[tr_mask]
            tr_px = px[tr_mask]
            agg_size = np.add.reduceat(tr_sz, starts)
            events.meta["agg_trade_t"] = tr_t[starts]
            events.meta["agg_trade_side"] = events.side[tr_mask][starts].astype(np.int8)
            events.meta["agg_trade_size"] = agg_size
            events.meta["agg_trade_log_vwap"] = np.log(
                np.add.reduceat(tr_sz * tr_px, starts) / agg_size
            )
            # S8: 攻撃注文ごとのメタオーダー行番号 (-1 = ノイズ)。⑪ の帰属分解用。
            events.meta["agg_trade_meta"] = agent[tr_mask][starts]
            # propagator 用: 各攻撃注文**直前**のミッド (ティック)。
            # ★EV_BB/BA は「イベント後」の値で、攻撃注文の行 (最初の TRADE の
            # 1 つ前) は既に約定後の板を映している。直前状態は**前のイベント束の
            # 最終行** = 最初の TRADE 行の 2 つ前にある (MO/LO 行が必ず TRADE の
            # 直前に 1 行入る構造による)。
            bb_f = ev[EV_BB, :n_events]
            ba_f = ev[EV_BA, :n_events]
            mid_row = np.where(
                (bb_f >= 0) & (ba_f >= 0), 0.5 * (bb_f + ba_f), np.nan
            )
            trade_idx = np.flatnonzero(tr_mask)
            pre_idx = np.maximum(trade_idx[starts] - 2, 0)
            events.meta["agg_trade_prev_mid_tick"] = mid_row[pre_idx]
        else:
            for key in ("agg_trade_t", "agg_trade_side", "agg_trade_size",
                        "agg_trade_log_vwap", "agg_trade_meta",
                        "agg_trade_prev_mid_tick"):
                events.meta[key] = np.empty(0)

        n_trades = int((etype == int(EventType.TRADE)).sum())
        self.last_diagnostics = {
            "n_events": int(n_events),
            "n_trades": n_trades,
            "engine_runtime_sec": engine_runtime,
            "throughput_events_per_sec": (
                n_events / engine_runtime if engine_runtime > 0 else None
            ),
            "counters": counters.copy(),
            "p0_tick": int(p0_tick),
            "base_price": base_price,
            "ev_capacity": int(ev_capacity),
            "max_orders": int(max_orders),
        }
        if use_hawkes:
            from .book_engine import (
                C_CX_NOOP,
                C_H_CANDIDATES,
                C_H_CAP_HITS,
                C_H_DAYCAP_HITS,
                C_H_REJECTED,
            )

            n_cand = float(counters[C_H_CANDIDATES])
            self.last_diagnostics["hawkes"] = {
                "branching_ratio_design": float(act.branching_ratio()),
                "stationary_rates_target_per_day": [float(v) for v in r_vec],
                "intensity_cap_per_day": h_cap,
                "candidates": int(n_cand),
                "rejected": int(counters[C_H_REJECTED]),
                "acceptance_rate": (
                    1.0 - counters[C_H_REJECTED] / n_cand if n_cand > 0 else None
                ),
                "cap_hits": int(counters[C_H_CAP_HITS]),
                "cap_hit_rate": (
                    counters[C_H_CAP_HITS] / n_cand if n_cand > 0 else None
                ),
                "daycap_hits": int(counters[C_H_DAYCAP_HITS]),
                "cx_noop": int(counters[C_CX_NOOP]),
                "phi_lambda_max": phi_max,
            }
        if use_cvol:
            from .book_engine import C_CVOL_TRUNC

            cvol_diag["truncations"] = int(counters[C_CVOL_TRUNC])
            self.last_diagnostics["cvol"] = cvol_diag
        if use_meta:
            from .book_engine import (
                C_META_ARRIVALS,
                C_META_CHILD,
                C_META_COMPLETED,
                C_META_NOISE,
                C_META_SPAWN_EMPTY,
            )

            n_child = float(counters[C_META_CHILD])
            n_noise = float(counters[C_META_NOISE])
            n_mo_meta = n_child + n_noise
            self.last_diagnostics["meta"] = {
                "n_metaorders": int(n_meta),
                "arrivals": int(counters[C_META_ARRIVALS]),
                "spawned_on_empty": int(counters[C_META_SPAWN_EMPTY]),
                "completed": int(counters[C_META_COMPLETED]),
                "children": int(n_child),
                "noise_trades": int(n_noise),
                "child_fraction": (n_child / n_mo_meta) if n_mo_meta > 0 else None,
                "psi_config": meta_args["psi"],
                "lambda_meta_per_day": meta_args["lam"],
                "e_len_theoretical": meta_args["e_len"],
                "lam_mo_total_target": meta_args["lam_mo_total"],
                "supply_ratio_config": float(cfg.meta_supply_ratio),
                "sequential": bool(cfg.meta_sequential),
                "pool_mean": float(pool_grid.mean()),
                "pool_median": float(np.median(pool_grid)),
                "pool_max": float(pool_grid.max()),
            }
        if use_ice:
            from .book_engine import (
                C_ICE_HIDDEN_IN,
                C_ICE_ORDERS,
                C_ICE_REFILLS,
                C_ICE_REFILL_VOL,
                C_SUBMITTED_LO,
            )

            self.last_diagnostics["iceberg"] = {
                "n_iceberg_orders": int(counters[C_ICE_ORDERS]),
                "refills": int(counters[C_ICE_REFILLS]),
                "refill_volume": float(counters[C_ICE_REFILL_VOL]),
                "hidden_volume_in": float(counters[C_ICE_HIDDEN_IN]),
                "iceberg_share_of_lo": (
                    float(counters[C_ICE_ORDERS])
                    / max(float(counters[C_SUBMITTED_LO]), 1.0)
                ),
            }
        return observation, events, book


def build_book_layer(
    config: Config,
    rng: RNGRegistry,
    calendar: ConstantCalendar,
    activity: ConstantActivity,
) -> PassThroughBook | ZIBook:
    if config.enable_book:
        # S7+: Hawkes の仕様 (行列・カーネル・ベースライン) は L1 が持ち、
        # L3 はそれを消費する。S6 (enable_hawkes=False) では使わない。
        return ZIBook(config, rng, calendar, activity)
    del rng, activity  # S0〜S5 の L3 は乱数も活動度も使わない
    return PassThroughBook(config, calendar)
