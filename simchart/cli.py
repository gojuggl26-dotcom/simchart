"""コマンドラインインターフェース。

    python -m simchart.cli run --config configs/s0.yaml --stage S0
    python -m simchart.cli validate --stage S0
    python -m simchart.cli compare --stages S0 S1

``run`` は「実行 -> 検証 -> ゲート判定 -> 永続化 -> 書き出しの確認 -> ゲート再判定」
という順で動く。最後の 2 段は ``artifacts_written`` ゲートのためで、書いたつもりで
終わらせないために metrics.json を読み直してから最終版を書く。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import Config
from .pipeline import (
    BASELINE_STAGE,
    baseline_invariance_check,
    determinism_check,
    rng_diffusion_check,
    rng_stability_check,
    run as run_pipeline,
    scale_invariance_check,
)
from .report import (
    compare_stages,
    load_metrics,
    make_plots,
    verify_metrics_file,
    write_metrics,
)
from .validation import evaluate, gates_for, summarize
from .validation.base import jsonable
from .validation.suite import collect_errors, run_all

__all__ = ["main"]


# ---------------------------------------------------------------------------
def _build_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config) if args.config else Config()
    overrides: dict[str, Any] = {}
    if args.stage:
        overrides["stage"] = args.stage
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.n_days is not None:
        overrides["n_days"] = args.n_days
    if args.steps_per_day is not None:
        overrides["steps_per_day"] = args.steps_per_day
    return config.replace(**overrides) if overrides else config


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.4e}"
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        # ★jsonable を通す。検証関数が numpy 配列を返す枝があり (S4 のスペクトル
        # 検定の ratios など)、生の dict を json.dumps すると TypeError で
        # **ゲート表示の途中で実行が落ちる**。表示のために解析を落とすのは本末転倒。
        return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))[:60]
    if isinstance(value, np.ndarray):
        return f"<配列 {value.shape}>"
    return str(value)


def _intraday_gph_pair(result, config: Config) -> tuple[float | None, float | None]:
    """日内バー |r| の GPH d を (生, 真値 φ 除去後) の組で返す (S4)。

    ★真値 φ で除く。推定 φ̂ ではなく真値を使うのは、ここで測りたいのが
    「季節性が推定を汚す量」であって「推定器の性能」ではないから。φ̂ の性能は
    別の枝 (seasonality.deseasonalization.recovery) が測る。
    """
    import numpy as np

    from .layers.l0_calendar import build_calendar
    from .rng import RNGRegistry
    from .validation.memory import gph_estimator
    from .validation.seasonality import deseasonalize, true_phi_bars

    obs = result.observation
    bars = obs.to_bars(config.validation.primary_bar_sec)
    r_2d = bars.returns_2d()
    calendar = build_calendar(config, RNGRegistry(config.seed))
    steps_per_day = (
        int(round(obs.session_seconds / obs.step_seconds)) if obs.step_seconds else None
    )
    truth = true_phi_bars(calendar, r_2d.shape[1], steps_per_day=steps_per_day)
    bwe = config.validation.gph_bandwidth_exponent
    raw = gph_estimator(np.abs(r_2d).ravel(), bandwidth_exponent=bwe).get("d")
    if truth["status"] != "ok":
        return raw, None
    dsn = gph_estimator(
        np.abs(deseasonalize(r_2d, np.asarray(truth["value"]))).ravel(),
        bandwidth_exponent=bwe,
    ).get("d")
    return raw, dsn


def _dilution_correlations(r5, r4) -> dict[str, float]:
    """レバレッジ希釈の相関ベース 3 計器 (S5 — **記録のみ**)。

    A. corr(r_t, RV_{t+1})     — 現行 multiseed の計器 (レベル領域、RV ノイズ入り)
    B. corr(r_t, IV_{t+1})     — 真値積分分散 (レベル領域)
    C. corr(r_t, log IV_{t+1}) — log 領域 (希釈式が形式的に当てはまる)

    ★ゲートには使わない (2026-08-21 裁定)。|L| ~ 0.02〜0.05 (S3 裁定の水準) に
    対し相関推定の SE ~0.014 が信号の 30〜40% あり、シード別の比は [−0.13, +3.04]
    と無統制になる。判定は log σ の経路 SD 比 (推定ノイズなし) が行い、こちらは
    「指示書の字義の計器ではどう見えるか」の記録。
    """
    import numpy as np

    out = {}
    for tag, r in (("5", r5), ("4", r4)):
        obs = r.observation
        rd = obs.to_bars(obs.session_seconds).returns()
        spd = int(round(obs.session_seconds / obs.step_seconds))
        n_days = rd.shape[0]
        dt_y = 1.0 / (252.0 * spd)
        # 本番グリッドは 1 配列 936MB。exp/diff の一時配列を丸ごと作らず
        # 日ブロックで畳む (250 日 ≈ 47MB/チャンク)。
        rv = np.empty(n_days, dtype=np.float64)
        iv = np.empty(n_days, dtype=np.float64)
        lp = obs.log_price
        lv = r.price.log_vol
        for d0 in range(0, n_days, 250):
            d1 = min(d0 + 250, n_days)
            seg = lp[d0 * spd : d1 * spd + 1]
            rv[d0:d1] = (np.diff(seg) ** 2).reshape(d1 - d0, spd).sum(axis=1)
            block = lv[d0 * spd : d1 * spd]
            iv[d0:d1] = (np.exp(2.0 * block) * dt_y).reshape(d1 - d0, spd).sum(axis=1)
        out[f"rv{tag}"] = float(np.corrcoef(rd[:-1], rv[1:])[0, 1])
        out[f"iv{tag}"] = float(np.corrcoef(rd[:-1], iv[1:])[0, 1])
        out[f"logiv{tag}"] = float(np.corrcoef(rd[:-1], np.log(iv[1:]))[0, 1])
    return {
        key: (out[f"{key}5"] / out[f"{key}4"] if out[f"{key}4"] != 0 else float("nan"))
        for key in ("rv", "iv", "logiv")
    }


def _book_seed_stats(result, config: Config) -> dict[str, float | None]:
    """S6 のシード別板統計 (multiseed の中央値記録用)。

    ゲート判定は seed 42 の単一実行で行う (板統計は L2 統計より遥かに速く収束する
    — 指示書 §4)。ここでの多シード値はシード間ばらつきの記録。
    """
    import math

    import numpy as np

    from .types import EventType

    ev = result.events
    meta = ev.meta if isinstance(ev.meta, dict) else {}
    burn = config.book_burn_in_days * result.observation.session_seconds
    out: dict[str, float | None] = {
        "book_throughput": (result.meta.get("l3") or {}).get("throughput_events_per_sec"),
    }
    bb = np.asarray(meta.get("best_bid_tick", np.empty(0)))
    ba = np.asarray(meta.get("best_ask_tick", np.empty(0)))
    if bb.size:
        m = (bb >= 0) & (ba >= 0) & (ev.t >= burn)
        out["book_spread_median"] = float(np.median(ba[m] - bb[m])) if m.any() else None
    sgn = np.asarray(meta.get("agg_trade_side", np.empty(0)), dtype=np.float64)
    tt = np.asarray(meta.get("agg_trade_t", np.empty(0)))
    if sgn.size > 5000:
        s = sgn[tt >= burn]
        d = s - s.mean()
        denom = float(d @ d)
        mx = max(
            abs(float(d[:-k] @ d[k:]) / denom) for k in range(1, 201)
        )
        out["book_sign_acf_max_z"] = mx * math.sqrt(s.size)
    is_order = (ev.event_type == int(EventType.LIMIT_ADD)) | (
        ev.event_type == int(EventType.MARKET)
    )
    ta = ev.t[is_order & (ev.t >= burn)]
    if ta.size > 1000:
        dt = np.diff(ta)
        dt = dt[dt > 0]
        out["book_interevent_cv2"] = float(dt.var() / dt.mean() ** 2)
    lp_star = meta.get("log_pstar")
    if lp_star is not None and bb.size:
        okm = (bb >= 0) & (ba >= 0) & (ev.t >= burn)
        stride = 100
        mid_v = 0.5 * (bb[okm] + ba[okm]).astype(np.float64)
        dm = np.diff(mid_v[::stride])
        dp = np.diff(np.asarray(lp_star)[okm][::stride])
        good = (dm != 0) | (dp != 0)
        if good.sum() > 100:
            out["book_corr_mid_pstar"] = float(np.corrcoef(dm[good], dp[good])[0, 1])
    return out


def _hawkes_seed_stats(result, config: Config) -> dict[str, float | None]:
    """S7 のシード別 Hawkes 統計 (multiseed の中央値記録用)。

    ゲート判定は seed 42 の単一実行 (500 日 = 297 万イベントで n̂ の 50 日ブロック
    SD が 0.003)。ここではシード間のばらつきを記録する。raw 経路の再推定は
    シードごとには回さない (計算が支配的になる割に情報が薄い — 単一シードの
    +0.066 と多シードの n̂_true の安定で足りる)。
    """
    import numpy as np

    from .validation.hawkes import hawkes_mle, marks_from_eventlog

    times, marks = marks_from_eventlog(result.events)
    session = float(result.observation.session_seconds)
    t_end = config.n_days * session
    betas = 1.0 / np.asarray(config.hawkes_tau_seconds, dtype=np.float64)
    w = np.asarray(config.hawkes_weights, dtype=np.float64)

    phi_table = None
    if config.enable_seasonality:
        from .layers.l0_calendar import build_calendar
        from .rng import RNGRegistry

        cal = build_calendar(config, RNGRegistry(config.seed))
        u = (np.arange(4096, dtype=np.float64) + 0.5) / 4096
        phi_table = np.asarray(cal.phi_lambda_of_u(u))

    # S10c: ベースラインは φ·Z — Z を渡さないと n̂ が Z のクラスタリング分だけ
    # 上振れする (suite 側と同じ補償。z はエンジン公開値)。
    ev_meta = result.events.meta if isinstance(result.events.meta, dict) else {}
    zkw: dict = {}
    if ev_meta.get("cvol_z_grid") is not None:
        zkw = {"z_grid": np.asarray(ev_meta["cvol_z_grid"], dtype=np.float64),
               "z_step_sec": float(ev_meta["cvol_z_step_sec"])}
    fit = hawkes_mle(
        times, marks, t_end, betas, w,
        phi_table=phi_table, session_seconds=session if phi_table is not None else None,
        **zkw,
    )
    burn = config.book_burn_in_days * session
    t_b = times[times >= burn]
    edges = np.arange(burn, t_end + 60.0, 60.0)
    c, _ = np.histogram(t_b, bins=edges)
    h_diag = (result.meta.get("l3") or {}).get("hawkes") or {}
    return {
        "hawkes_n_hat_true_phi": float(fit["n_hat"]) if fit["converged"] else None,
        "hawkes_fano_60s": float(c.var() / c.mean()) if c.mean() > 0 else None,
        "hawkes_acceptance": h_diag.get("acceptance_rate"),
        "hawkes_cap_hits": float(h_diag.get("cap_hits", -1)),
    }


def _meta_seed_stats(result, config: Config) -> dict[str, float | None]:
    """S8 のシード別メタオーダー統計 (multiseed の中央値判定用)。

    γ・VR・β は whale (α<2 の裾) に支配され単一シードで大きく散らばる
    (γ̂: 250 日で [0.52, 0.66]、日次 VR: {0.97, 1.9, 14.7} を実測)。
    量的ゲートは全てここからの多シード中央値で判定する (S3 §8 のインフラ)。
    """
    from .validation.suite import _meta_metrics

    m = _meta_metrics(result, config)
    g = m["sign_acf_gamma"]
    dfc = m["impact_deficit"]
    return {
        "meta_gamma": g.get("gamma"),
        "meta_c1": g.get("c1"),
        "meta_acf_r2": g.get("r2"),
        "meta_beta": dfc.get("beta_measured"),
        "meta_beta_deficit": dfc.get("beta_deficit"),
        "meta_vr_trade_1000": dfc.get("vr_s8_trade_1000"),
        "meta_vr_daily_max": dfc.get("vr_s8_daily_max"),
        "meta_sqrt_slope": dfc.get("sqrt_law_exponent"),
        "meta_sqrt_qv": dfc.get("sqrt_law_exponent_qv"),
        "meta_pool_rel_diff": m["pool"].get("rel_diff"),
    }


def _qr_seed_stats(result, config: Config) -> dict[str, float | None]:
    """S9 のシード別 queue-reactive 統計 (multiseed の中央値判定用)。"""
    from .validation.suite import _qr_metrics

    m = _qr_metrics(result, config)
    return {
        "qr_eta_trade": m["eta_trade"].get("eta"),
        "qr_eta_mid": m["eta_mid"].get("eta"),
        "qr_change_sign_corr": m["mid_return_acf"].get("change_sign_corr_event"),
        "qr_obi_h1": m["obi"].get("corr_h1"),
        "qr_reversion_frac": m["reversion"].get("reversion_frac"),
        "qr_depth_peak_tick": m["depth_tick_profile"].get("peak_tick_distance"),
    }


def _coupling_seed_stats(result, config: Config) -> dict[str, float | None]:
    """S10 のシード別結合統計 (multiseed の中央値判定用)。

    ⑪ の保存判定は**残差符号の γ** (raw の C(ℓ) には κ 追跡の情報チャネルが
    重畳する — S10a の解剖)。T_daily・乖離半減期・⑦ も whale/エポックで
    シード間に散らばるため中央値で判定する。
    """
    from .validation.suite import _coupling_metrics

    m = _coupling_metrics(result, config)
    tr = m["transmission"].get("T") or {}
    return {
        "cpl_gamma_resid": m["residual_sign_acf"].get("gamma_resid"),
        "cpl_gamma_resid_tail": m["residual_sign_acf"].get("gamma_resid_tail"),
        "cpl_c1_resid": m["residual_sign_acf"].get("c1_resid"),
        "cpl_gamma_raw": m["residual_sign_acf"].get("gamma_raw"),
        "cpl_c1_raw": m["residual_sign_acf"].get("c1_raw"),
        "cpl_gap_halflife_min": m["gap"].get("halflife_min"),
        "cpl_gap_sd_bp": m["gap"].get("sd_bp"),
        "cpl_T_daily": m["transmission"].get("T_daily"),
        "cpl_T_5d": tr.get("h117000s"),
        "cpl_corr_daily_level": m["tracking"].get("corr_daily_level"),
        "cpl_corr_daily_return": m["tracking"].get("corr_daily_return"),
        "cpl_rv_volume_log": m["vol_activity"].get("corr_rv_volume_log"),
        "cpl_rv_spread": m["vol_activity"].get("corr_rv_spread"),
    }


def _nt_series(result, config: Config):
    """n_t の時系列 (u と χ₃ から決定論的に再構成、バーンイン後、時間格子)。"""
    import numpy as np

    meta = result.events.meta if isinstance(result.events.meta, dict) else {}
    u = np.asarray(meta.get("fb_u_grid", np.empty(0)), dtype=np.float64)
    if not u.size:
        return None
    step = float(meta.get("fb_u_step_sec", 60.0))
    th = np.tanh(u / float(config.fb_u_scale))
    arg = float(config.fb_b_n) * th
    if config.enable_chaos_branching and float(config.chi3_b) > 0.0:
        from .chaos import chi_window

        t3_days, chi3_norm, _ = chi_window(config, float(config.n_days) + 1.0, "chi3")
        t_days = np.arange(u.size, dtype=np.float64) * step / 23400.0
        arg = arg + float(config.chi3_b) * np.interp(t_days, t3_days, chi3_norm)
    sig = 1.0 / (1.0 + np.exp(-arg))
    nt = float(config.fb_n_min) + (float(config.fb_n_max) - float(config.fb_n_min)) * sig
    start = int(config.book_burn_in_days * 23400.0 / step)
    return nt[start:]


def _nt_window_means(result, config: Config, window_days: float):
    """窓 (5 日) ごとの平均 n_t (§8.1 の「脆弱性の窓」の直接検証素材)。"""
    meta = result.events.meta if isinstance(result.events.meta, dict) else {}
    step = float(meta.get("fb_u_step_sec", 60.0))
    nt_b = _nt_series(result, config)
    if nt_b is None:
        return None
    spw = int(round(window_days * 23400.0 / step))
    n_w = nt_b.size // spw
    return nt_b[: n_w * spw].reshape(n_w, spw).mean(axis=1)


def _u_time_mean(result, config: Config) -> float | None:
    """時間加重の u 平均 (スナップショット格子、バーンイン後)。"""
    import numpy as np

    meta = result.events.meta if isinstance(result.events.meta, dict) else {}
    u = np.asarray(meta.get("fb_u_grid", np.empty(0)), dtype=np.float64)
    if not u.size:
        return None
    step = float(meta.get("fb_u_step_sec", 60.0))
    burn = int(config.book_burn_in_days * 23400.0 / step)
    return float(u[burn:].mean()) if u.size > burn else None


def _feedback_solo_stats(result, config: Config) -> dict[str, float | None]:
    """S11/S13 のシード別フィードバック統計のうち **off 対を要しない**部分。

    S13 の多資産ループは参照資産についてこれだけを収集する (ペア量 g・発散・
    T_off・深さ CV 比は S12 から繰り越し — ループ機構は n1 回帰がビット単位で
    同一と保証しており、§11 の「参照資産の単変量性質は S12 の結果を流用」の実装)。
    """
    from .validation import feedback as fbv

    import numpy as np

    det = fbv.crisis_detect(result, config)
    ana = fbv.crisis_anatomy(
        result, config, detection=det if det.get("status") == "ok" else None
    )
    fb = (result.meta.get("l3") or {}).get("feedback") or {}

    # ⑧ の判定は**危機日除外の Hill α** (指示書 §6.2 の分解そのもの)。全体 α は
    # 増幅が whale 日に集中するため構造的に低下する (フロンティア実測 — 記録)。
    def _hill_split():
        from .validation.tails import hill_estimator

        det_ = fbv.crisis_detect(result, config)
        obs = result.observation
        spd = int(round(23400.0 / obs.step_seconds))
        r_d = np.diff(np.asarray(obs.log_price)[::spd])
        h_all = hill_estimator(r_d, 0.05, "both").get("alpha")
        mask = np.ones(r_d.size, dtype=bool)
        step_snap = det_.get("step_sec") or 60.0
        for a, b in det_.get("episodes") or []:
            d0 = int(a * step_snap / 23400.0)
            d1 = int(b * step_snap / 23400.0)
            mask[max(d0 - 1, 0): min(d1 + 2, r_d.size)] = False
        h_ex = hill_estimator(r_d[mask], 0.05, "both").get("alpha")
        return h_all, h_ex

    hill_all, hill_ex = _hill_split()

    # ③ の判定は危機日を**対でマスク**した gph_d 差 — 危機スパイクは日次 |r| の
    # GPH を白色希釈する (S3 で解剖済みの機構)。同じ日を両系列から除くので
    # 「観測は潜在の記憶を保存するか」を共通サポートで問える。
    def _masked_gph_diff() -> float | None:
        from .validation.memory import gph_estimator

        det_ = fbv.crisis_detect(result, config)
        obs = result.observation
        spd = int(round(23400.0 / obs.step_seconds))
        r_obs = np.diff(np.asarray(obs.log_price)[::spd])
        # S12: χ₁ (4.7 日) は日次 GPH の推定帯の内側にあり obs 側の d を
        # ~−0.02 傾ける (設計変調 — R² 劣化と同機構)。既知の決定論係数
        # e^{a₁χ₁/2} (RV ∝ 活動度 → |r| ∝ √活動度) で除去してから測る —
        # φ 脱季節化 (S4) と方法論的に同一の「既知変調の除去」。
        day_centers = np.arange(r_obs.size, dtype=np.float64) + 0.5
        if config.enable_chaos_lambda:
            from .chaos import chi_window

            t1d_, x1_, _ = chi_window(config, float(config.n_days) + 1.0, "chi1")
            a1_ = float(config.c_vol) * float(
                np.sqrt(config.chi1_var_share / (1.0 - config.chi1_var_share))
            )
            chi1_day = np.interp(day_centers, t1d_, x1_)
            r_obs = r_obs / np.exp(0.5 * a1_ * chi1_day)
        if config.enable_chaos_branching and float(config.chi3_b) > 0.0:
            # χ₃ の**決定論部分** (u=0 の n_t^det) が作る活動係数も同様に除去 —
            # 13 日の近臨界窓は週スケールの振幅変調として GPH 帯内に低周波
            # パワーを足す (危機日マスクでは滑らかな変調は取れない)。
            # u 由来のシード固有部分は除去しない (それは実ダイナミクス)。
            from .chaos import chi_window as _cw3

            t3d_, x3_, _ = _cw3(config, float(config.n_days) + 1.0, "chi3")
            chi3_day = np.interp(day_centers, t3d_, x3_)
            sig3 = 1.0 / (1.0 + np.exp(-float(config.chi3_b) * chi3_day))
            nt_det = float(config.fb_n_min) + (
                float(config.fb_n_max) - float(config.fb_n_min)
            ) * sig3
            n_design_ = float(np.max(np.abs(np.linalg.eigvals(
                np.asarray(config.hawkes_a, dtype=np.float64)))))
            f_det = (1.0 - n_design_) / (1.0 - nt_det)
            r_obs = r_obs / np.sqrt(f_det)
        ps = np.asarray(result.price.log_p_star)
        spd_g = int(round(23400.0 / float(result.price.t[1] - result.price.t[0])))
        r_lat = np.diff(ps[::spd_g])
        n = min(r_obs.size, r_lat.size)
        mask = np.ones(n, dtype=bool)
        step_snap = det_.get("step_sec") or 60.0
        for a, b in det_.get("episodes") or []:
            d0 = int(a * step_snap / 23400.0)
            d1 = int(b * step_snap / 23400.0)
            mask[max(d0 - 1, 0): min(d1 + 2, n)] = False
        bw = config.validation.daily_gph_bandwidth_exponent
        d_o = gph_estimator(np.abs(r_obs[:n][mask]), bw).get("d")
        d_l = gph_estimator(np.abs(r_lat[:n][mask]), bw).get("d")
        return (d_o - d_l) if (d_o is not None and d_l is not None) else None

    return {
        "fb_crises_per_year": det.get("per_year"),
        "fb_recovery30_dislocation": ana.get("recovery_30min_dislocation"),
        "fb_recovery1d_dislocation": ana.get("recovery_1day_dislocation"),
        "fb_recovery1d_catchup": ana.get("recovery_1day_catchup"),
        "fb_n_dislocation": ana.get("n_dislocation"),
        "fb_crisis_duration_min": ana.get("duration_min_median"),
        "fb_crisis_spread_ratio": ana.get("max_spread_ratio_median"),
        "fb_crisis_depth_ratio": ana.get("min_depth_ratio_median"),
        "fb_nt_mean": fb.get("nt_mean"),
        "fb_nt_mean_time": (
            float(np.mean(nt_s)) if (nt_s := _nt_series(result, config)) is not None
            else None
        ),
        "fb_nt_max": fb.get("nt_max"),
        # ★§2.1 の定常性は**時間加重**平均で判定する。イベント加重 (エンジンの
        # カウンタ) はイベントが高 u 状態に集積するため +1 前後になる —
        # それ自体は「活動は驚きに集中する」という情報なので別名で記録。
        "fb_u_mean_time": _u_time_mean(result, config),
        "fb_u_mean_event": fb.get("u_mean"),
        "fb_gph_d_diff_masked": _masked_gph_diff(),
        "fb_hill_all": hill_all,
        "fb_hill_ex_crisis": hill_ex,
    }


def _feedback_seed_stats(result, config: Config, result_off) -> dict[str, float | None]:
    """S11 のシード別フィードバック統計 (multiseed の中央値判定用)。

    ペア量 (ループゲイン g・発散) は同一シードの off 対で測る (§4.1 — L2 経路が
    同一なので L2 起因が厳密に相殺する)。ソロ部分は :func:`_feedback_solo_stats`。
    """
    from .validation import feedback as fbv

    import numpy as np

    out = _feedback_solo_stats(result, config)
    g = fbv.loop_gain_estimate(result, result_off, config)
    div = fbv.divergence_monitor(result, config, result_off=result_off)

    # ⑭ デプス変動の増大 — 同一シード off 対との CV 比 (基準値不要のペア計器)
    def _depth_cv(res) -> float | None:
        bk = res.book
        d = np.asarray(bk.bid_sz, dtype=np.float64).sum(axis=1) + np.asarray(
            bk.ask_sz, dtype=np.float64
        ).sum(axis=1)
        d = d[d > 0]
        return float(d.std() / d.mean()) if d.size > 100 and d.mean() > 0 else None

    cv_on = _depth_cv(result)
    cv_off = _depth_cv(result_off)

    # 結合忠実度 (T_daily) は **off 対で判定** — g ∈ [0.3,0.6] は日次分散の増幅を
    # 強制するので、on 側の T ±0.07 は指示書内部で矛盾する (S11e 実測 T_on ~1.9)。
    # κ/σ̄ はフィードバックが触らないため off 対がその検証。on 側は超過として記録
    # (幾何/算術の分解: 典型日 +8% / 平均分散 ×2 = 裾駆動 — 危機の物理そのもの)。
    from .validation.coupling import transmission

    t_off = transmission(result_off, config)
    from .validation.feedback import _log_rv_series

    lon = _log_rv_series(result.observation, config, 23400.0)
    loff = _log_rv_series(result_off.observation, config, 23400.0)
    n_c = min(lon.size, loff.size)
    out.update({
        "fb_g_30min": g.get("g_30min"),
        "fb_g_daily": g.get("g_daily"),
        "fb_divergences": div.get("n_divergences"),
        "fb_crises_per_year_off": fbv.crisis_detect(result_off, config).get("per_year"),
        "fb_depth_cv_ratio": (
            cv_on / cv_off if (cv_on is not None and cv_off) else None
        ),
        "fb_T_daily_off": t_off.get("T_daily"),
        "fb_rv_excess_geo": float(np.exp(lon[:n_c].mean() - loff[:n_c].mean())),
        "fb_rv_excess_ari": float(np.exp(lon[:n_c]).mean() / np.exp(loff[:n_c]).mean()),
    })
    return out


def _run_multiseed(config: Config, n_seeds: int) -> dict[str, Any]:
    """ノイズの大きい指標をシードを変えて測り、中央値・IQR を返す (S3 指示書 §8)。

    Hill α は 5000 日でも上位 5% が 250 観測しかなく単一シードで ±0.5 ばらつく。
    対象は hill_alpha / leverage_corr / jv_share / skewness の 4 つ (経路統計)。
    分散予算・補償・相関実測・決定性は単一シードで判定できるので対象外。
    追加シードでは対象指標だけを測る (フル検証スイートは回さない)。
    """
    import numpy as np
    from scipy import stats as sp_stats

    from .validation.memory import leverage_function
    from .validation.scaling import realized_variance
    from .validation.tails import bns_jump_test, hill_by_scale, hill_estimator

    per_seed: dict[str, list[float]] = {
        "hill_alpha": [], "leverage_corr": [], "jv_share": [], "skewness_daily": [],
        "hill_scale_slope": [], "gph_d": [],
        # S4: 季節性が**日内**バーの長期記憶推定を汚す量。1 経路では GPH の
        # SE 0.013 に埋もれる (実測バイアス +0.017) ため中央値で判定する。
        # 日次 gph_d は φ_σ の二乗正規化により汚染を受けない (別枝で確認)。
        "gph_d_intraday_raw": [], "gph_d_intraday_deseason": [], "gph_bias_intraday": [],
        # S5: レバレッジ希釈の SD 比 (2026-08-21 裁定の判定計器) と、
        # 相関ベース 3 計器の比 (記録 — |L| ~ 0.02 の水準では判定不能)。
        "dilution_sd_ratio": [],
        "dilution_corr_rv": [], "dilution_corr_iv": [], "dilution_corr_logiv": [],
    }
    # S5: シード横断相関 (指示書 §8 — S5 の中核ゲート) 用に φ 除去済み log σ の
    # サブサンプルを保持する。1 分間引きで 1 シード ~16MB、10 シードで 156MB。
    cross_seed_paths: list[np.ndarray] = []
    chi_hashes: list[str] = []
    fb_window_vecs: list = []
    fb_nt_window_vecs: list = []
    seeds = [config.seed + i for i in range(n_seeds)]
    skipped_seeds: list[dict[str, str]] = []
    for i, seed in enumerate(seeds):
        seed_config = config.replace(seed=seed)
        # ★S8+: 超拡散ミッドは重い裾のトレンドを引き、稀に板窓 (価格正値性で
        # 上限あり) から逸脱して RuntimeError で止まる。多シード判定では該当
        # シードを**記録の上でスキップ**し、残りの中央値で判定する (欠落は
        # 最大トレンド側の打ち切りなので中央値への影響は片側・軽微 — README)。
        try:
            result = run_pipeline(seed_config)
        except RuntimeError as exc:
            skipped_seeds.append({"seed": str(seed), "leg": "main", "error": str(exc)})
            print(f"      シード {seed} ({i + 1}/{n_seeds}) スキップ: {exc}", flush=True)
            continue
        obs = result.observation
        r_daily = obs.to_bars(obs.session_seconds).returns()
        steps_per_day = int(round(obs.session_seconds / obs.step_seconds))
        step_r = np.diff(obs.log_price)
        rv_daily = realized_variance(step_r, steps_per_day)

        hill = hill_estimator(r_daily, 0.05, "both")
        per_seed["hill_alpha"].append(hill.get("alpha"))
        lev = leverage_function(r_daily, rv_daily, horizons=(0, 1))
        per_seed["leverage_corr"].append(lev.get("corr_r_rv_h1"))
        bns = bns_jump_test(step_r, steps_per_day)
        per_seed["jv_share"].append(bns.get("jv_share"))
        per_seed["skewness_daily"].append(float(sp_stats.skew(r_daily, bias=False)))
        hbs = hill_by_scale(r_daily)
        per_seed["hill_scale_slope"].append(hbs.get("slope_vs_log_scale"))
        from .validation.memory import gph_estimator

        per_seed["gph_d"].append(
            gph_estimator(np.abs(r_daily), config.validation.daily_gph_bandwidth_exponent).get("d")
        )
        # S10 (κ>0): ③ の判定計器 — 同一シードの潜在 log p* で同じ gph_d を測り
        # **per-seed 差**で判定する (S5 基準値は 5000 日測定で、1000 日の観測値と
        # 有限標本バイアスが異なる。同一ラン・同一視野の差なら相殺する)。
        # ⑧ の JV は 1 秒 BNS だとバウンスをジャンプと誤検出する (S10b) ので
        # バウンス頑健な 5 分サンプリング版も測る。
        if config.kappa > 0.0:
            ps = result.price.log_p_star
            step_g = float(result.price.t[1] - result.price.t[0])
            spd_g = int(round(obs.session_seconds / step_g))
            r_daily_lat = np.diff(np.asarray(ps)[::spd_g])
            d_lat = gph_estimator(
                np.abs(r_daily_lat), config.validation.daily_gph_bandwidth_exponent
            ).get("d")
            d_obs = per_seed["gph_d"][-1]
            per_seed.setdefault("gph_d_latent", []).append(d_lat)
            per_seed.setdefault("gph_d_obs_minus_latent", []).append(
                d_obs - d_lat if (d_obs is not None and d_lat is not None) else None
            )
            stride5 = max(1, int(round(300.0 / obs.step_seconds)))
            r5 = np.diff(obs.log_price[::stride5])
            per_seed.setdefault("jv_share_5min", []).append(
                bns_jump_test(r5, steps_per_day // stride5).get("jv_share")
            )
            # ① の歪度も同一シード対で記録 (計器のペア差 SD 2.7 実測 — 検定力が
            # 無いので記録のみ。ゲートは張らない)
            sk_lat = float(sp_stats.skew(r_daily_lat, bias=False))
            per_seed.setdefault("skew_daily_latent", []).append(sk_lat)
            sk_obs = per_seed["skewness_daily"][-1]
            per_seed.setdefault("skew_obs_minus_latent", []).append(
                sk_obs - sk_lat if sk_obs is not None else None
            )
        # S6 (κ=0 の板): 観測は ZI ミッドなので、観測ベースの季節性ペアと
        # 希釈の相関ペア実行 (どちらも観測を測る) はスキップする。潜在側
        # (SD 比・シード横断・chi ハッシュ) は板と無関係なので継続する。
        obs_is_book = config.enable_book and config.kappa == 0.0
        if config.enable_seasonality and not obs_is_book:
            raw, dsn = _intraday_gph_pair(result, seed_config)
            per_seed["gph_d_intraday_raw"].append(raw)
            per_seed["gph_d_intraday_deseason"].append(dsn)
            per_seed["gph_bias_intraday"].append(
                raw - dsn if raw is not None and dsn is not None else None
            )
        if config.enable_chaos_vol:
            sub = result.meta["l2"]["vol_subsample"]
            lv_with = np.asarray(sub["log_vol"]) - np.asarray(sub["log_phi_sigma"])
            lv_without = lv_with - np.asarray(sub["chi_term"]) + float(sub["c_chi"])
            v_with, v_without = float(lv_with.var()), float(lv_without.var())
            per_seed["dilution_sd_ratio"].append(
                float(np.sqrt(v_without / v_with)) if v_with > 0 else None
            )
            if not obs_is_book:
                # 相関ベース 3 計器: 同一シードで chi を厳密に除いた S4 相当ペアを
                # 回す (log σ は引き算で厳密復元できるが、価格はジャンプ抽選が
                # λ(σ) 経由で変わるため再実行が必要)。
                # ★S10 (κ>0): アブレーション側は p* 経路が変わるため、本走が
                # 完走しても窓逸脱で落ちうる — 記録の上スキップ (ice-off と同じ)。
                try:
                    r4 = run_pipeline(seed_config.replace(enable_chaos_vol=False))
                except RuntimeError as exc:
                    skipped_seeds.append(
                        {"seed": str(seed), "leg": "chaos_off", "error": str(exc)}
                    )
                    print(f"      シード {seed} (chaos-off) スキップ: {exc}", flush=True)
                else:
                    dil = _dilution_correlations(result, r4)
                    for key_ in ("rv", "iv", "logiv"):
                        per_seed[f"dilution_corr_{key_}"].append(dil[key_])
                    del r4
            # シード横断相関 (5 分に間引いてメモリを 1/5 に)。
            cross_seed_paths.append(lv_with[::5].astype(np.float64))
            chi_hashes.append(result.meta["l2"]["chaos"]["sha256"])
            del lv_with, lv_without
        if config.enable_book:
            stats = _book_seed_stats(result, config)
            for key_, val_ in stats.items():
                per_seed.setdefault(key_, []).append(val_)
        if config.enable_hawkes:
            hstats = _hawkes_seed_stats(result, config)
            for key_, val_ in hstats.items():
                per_seed.setdefault(key_, []).append(val_)
        if config.enable_metaorder:
            mstats = _meta_seed_stats(result, seed_config)
            for key_, val_ in mstats.items():
                per_seed.setdefault(key_, []).append(val_)
        if config.enable_queue_reactive:
            qstats = _qr_seed_stats(result, seed_config)
            for key_, val_ in qstats.items():
                per_seed.setdefault(key_, []).append(val_)
        if config.kappa > 0.0 or config.c_vol > 0.0:
            cstats = _coupling_seed_stats(result, seed_config)
            for key_, val_ in cstats.items():
                per_seed.setdefault(key_, []).append(val_)
        if config.enable_feedback:
            # S11: ループゲイン・発散 (ペア判定 §4.1) — 同一シードの off 対を回す。
            import dataclasses as _dc2

            defaults2 = {f.name: f.default for f in _dc2.fields(type(config))}
            # ★off 対は χ₃ も外す (χ₃ は n_t 機構に乗るため単独では立てられない)。
            # χ₁ は両脚に残す — 決定論変調は比で相殺し、ループだけが測れる。
            cfg_off = seed_config.replace(
                enable_feedback=False, enable_chaos_branching=False,
                **{n: defaults2[n] for n in type(config)._S11_FB_PARAMS},
                **{n: defaults2[n] for n in type(config)._S12_CHI3_PARAMS},
            )
            try:
                r_off = run_pipeline(cfg_off)
            except RuntimeError as exc:
                skipped_seeds.append(
                    {"seed": str(seed), "leg": "feedback_off", "error": str(exc)}
                )
                print(f"      シード {seed} (fb-off) スキップ: {exc}", flush=True)
            else:
                fstats = _feedback_seed_stats(result, seed_config, r_off)
                for key_, val_ in fstats.items():
                    per_seed.setdefault(key_, []).append(val_)
                del r_off
                # S12 §8.2: 窓あたり危機件数と窓平均 n_t のベクトルを収集
                # (シード横断相関はループ後に一括計算)
                if config.enable_chaos_branching:
                    from .validation import feedback as fbv2

                    det_w = fbv2.crisis_detect(result, seed_config)
                    fb_window_vecs.append(
                        fbv2.crisis_window_counts(
                            result, seed_config, 5.0, detection=det_w
                        )
                    )
                    fb_nt_window_vecs.append(
                        _nt_window_means(result, seed_config, 5.0)
                    )
            if config.enable_iceberg:
                # §6.3 アブレーション: 同一シードで iceberg off。単一シードの
                # on/off 差は経路分岐で SD ~0.06 になるため、中央値同士で判定する。
                import dataclasses as _dc

                defaults_ = {f.name: f.default for f in _dc.fields(type(config))}
                resets_ = {
                    n: defaults_[n] for n in type(config)._S8_ICEBERG_PARAMS
                }
                cfg_off = seed_config.replace(enable_iceberg=False, **resets_)
                try:
                    r_off = run_pipeline(cfg_off)
                except RuntimeError as exc:
                    skipped_seeds.append(
                        {"seed": str(seed), "leg": "iceberg_off", "error": str(exc)}
                    )
                    print(f"      シード {seed} (ice-off) スキップ: {exc}", flush=True)
                else:
                    g_off = _meta_seed_stats(r_off, cfg_off)
                    per_seed.setdefault("meta_gamma_ice_off", []).append(
                        g_off["meta_gamma"]
                    )
                    # ★主計器は C(1): γ̂ は whale 支配でシード中央値同士の差にも
                    # SD ~0.035 のノイズが残る (構造的にゼロ効果でも ±0.05 は
                    # 4 割の確率で偽陽性)。C(1) は SD ~0.002 で 20 倍鋭い。
                    per_seed.setdefault("meta_c1_ice_off", []).append(
                        g_off["meta_c1"]
                    )
                    del r_off
        del result, obs, step_r, rv_daily
        print(f"      シード {seed} ({i + 1}/{n_seeds}) 完了", flush=True)

    out: dict[str, Any] = {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "skipped_seeds": skipped_seeds,
        "n_completed": n_seeds - sum(1 for s in skipped_seeds if s["leg"] == "main"),
    }
    if config.enable_chaos_vol and cross_seed_paths:
        from .validation.scaling import cross_seed_correlation

        out["cross_seed_corr"] = cross_seed_correlation(cross_seed_paths)
        out["chi_hash_all_equal"] = bool(len(set(chi_hashes)) == 1)
        out["chi_hashes"] = chi_hashes
        del cross_seed_paths
    # S12 §8: 窓再現性の二層計器 (シード横断)。
    #  - fb_nt_window_corr: 脆弱変数 n_t 自体の窓再現 (χ₃ 支配 → 高相関が正解)
    #  - fb_window_repro: 危機**発火**の窓相関 (存在が鯨供給のため天井 ~0.15 —
    #    ICC 分解込みで記録。「窓は再現・発火は確率的」§8.1 の定量形)
    if fb_window_vecs:
        from .validation.feedback import crisis_window_reproducibility

        out["fb_window_repro"] = crisis_window_reproducibility(fb_window_vecs)
        nt_ok = [v for v in fb_nt_window_vecs if v is not None]
        out["fb_nt_window_corr"] = crisis_window_reproducibility(nt_ok)
        if len(fb_window_vecs) >= 3:
            n_c = min(v.size for v in fb_window_vecs)
            M = np.stack([np.asarray(v)[:n_c] for v in fb_window_vecs])
            var_b = float(M.mean(axis=0).var())
            var_w = float(M.var(axis=0).mean())
            var_chi = max(var_b - var_w / M.shape[0], 0.0)
            out["fb_window_repro"]["icc_chi3_share"] = (
                var_chi / (var_chi + var_w) if (var_chi + var_w) > 0 else None
            )

    for name, values in per_seed.items():
        clean = [v for v in values if v is not None]
        if not clean:
            out[name] = {"median": None, "values": values}
            continue
        arr = np.array(clean, dtype=np.float64)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        out[name] = {
            "median": float(np.median(arr)),
            "iqr": q3 - q1,
            "q1": q1,
            "q3": q3,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": values,
        }
    return out


def _path_seed_stats_s13(result, config: Config) -> dict[str, float | None]:
    """S13 用: 参照資産のシード別経路統計 (S12 の _run_multiseed 内の
    インラインブロックと同じキー・同じ計算)。

    ★S12 ループはタグ済みの回帰資産なので触らない — ここは意図的な小さな
    複製で、キー名の一致は compare S12 S13 の前提。
    """
    import numpy as np
    from scipy import stats as sp_stats

    from .validation.memory import gph_estimator, leverage_function
    from .validation.scaling import realized_variance
    from .validation.tails import bns_jump_test, hill_by_scale, hill_estimator

    obs = result.observation
    r_daily = obs.to_bars(obs.session_seconds).returns()
    steps_per_day = int(round(obs.session_seconds / obs.step_seconds))
    step_r = np.diff(obs.log_price)
    rv_daily = realized_variance(step_r, steps_per_day)
    out: dict[str, float | None] = {}
    out["hill_alpha"] = hill_estimator(r_daily, 0.05, "both").get("alpha")
    out["leverage_corr"] = leverage_function(r_daily, rv_daily, horizons=(0, 1)).get(
        "corr_r_rv_h1"
    )
    out["jv_share"] = bns_jump_test(step_r, steps_per_day).get("jv_share")
    out["skewness_daily"] = float(sp_stats.skew(r_daily, bias=False))
    out["hill_scale_slope"] = hill_by_scale(r_daily).get("slope_vs_log_scale")
    bw = config.validation.daily_gph_bandwidth_exponent
    out["gph_d"] = gph_estimator(np.abs(r_daily), bw).get("d")
    ps = result.price.log_p_star
    step_g = float(result.price.t[1] - result.price.t[0])
    spd_g = int(round(obs.session_seconds / step_g))
    r_daily_lat = np.diff(np.asarray(ps)[::spd_g])
    d_lat = gph_estimator(np.abs(r_daily_lat), bw).get("d")
    out["gph_d_latent"] = d_lat
    out["gph_d_obs_minus_latent"] = (
        out["gph_d"] - d_lat if (out["gph_d"] is not None and d_lat is not None) else None
    )
    stride5 = max(1, int(round(300.0 / obs.step_seconds)))
    out["jv_share_5min"] = bns_jump_test(
        np.diff(obs.log_price[::stride5]), steps_per_day // stride5
    ).get("jv_share")
    sk_lat = float(sp_stats.skew(r_daily_lat, bias=False))
    out["skew_daily_latent"] = sk_lat
    out["skew_obs_minus_latent"] = out["skewness_daily"] - sk_lat
    return out


def _run_multiseed_s13(config: Config, n_seeds: int) -> dict[str, Any]:
    """S13 の多シード判定: 各シードで run_multi し、参照資産のソロ統計と
    クロス資産統計を収集する。

    ペア量 (g・発散・T_off・深さ CV・iceberg) は収集しない — S12 から繰り越す
    (§11「参照資産の単変量性質は S12 の結果を流用」、繰り越しの妥当性は
    n1 回帰のビット単位一致が担保する。metrics.s12_carryover を参照)。
    危機時相関はシード横断で日次系列をプールして測る (§11 の 2000 日以上の
    要求を 1000 日 × n シードのプールで満たす — 単一メカニズムの定常標本)。
    """
    import numpy as np

    from .pipeline import run_multi
    from .validation import feedback as fbv2
    from .validation.cross import conditional_correlation, cross_asset_metrics

    per_seed: dict[str, list] = {}
    skipped_seeds: list[dict[str, str]] = []
    chi_hashes: list[str] = []
    cross_seed_paths: list[np.ndarray] = []
    fb_window_vecs: list = []
    fb_nt_window_vecs: list = []
    pooled_pairs: dict[str, dict[str, list]] = {}
    seeds = [config.seed + i for i in range(n_seeds)]
    for i, seed in enumerate(seeds):
        seed_config = config.replace(seed=seed)
        try:
            multi = run_multi(seed_config)
        except RuntimeError as exc:
            skipped_seeds.append({"seed": str(seed), "leg": "main", "error": str(exc)})
            print(f"      シード {seed} ({i + 1}/{n_seeds}) スキップ: {exc}", flush=True)
            continue
        result = multi.asset0

        for key_, val_ in _path_seed_stats_s13(result, seed_config).items():
            per_seed.setdefault(key_, []).append(val_)
        for helper in (_book_seed_stats, _hawkes_seed_stats, _meta_seed_stats,
                       _qr_seed_stats, _coupling_seed_stats, _feedback_solo_stats):
            for key_, val_ in helper(result, seed_config).items():
                per_seed.setdefault(key_, []).append(val_)

        # χ₂ の決定性とレバレッジ希釈 (ソロ — S12 ループと同じ計器)
        sub = result.meta["l2"]["vol_subsample"]
        lv_with = np.asarray(sub["log_vol"]) - np.asarray(sub["log_phi_sigma"])
        lv_without = lv_with - np.asarray(sub["chi_term"]) + float(sub["c_chi"])
        v_w, v_wo = float(lv_with.var()), float(lv_without.var())
        per_seed.setdefault("dilution_sd_ratio", []).append(
            float(np.sqrt(v_wo / v_w)) if v_w > 0 else None
        )
        cross_seed_paths.append(lv_with[::5].astype(np.float64))
        chi_hashes.append(result.meta["l2"]["chaos"]["sha256"])
        del lv_with, lv_without

        # 窓再現性 (参照資産、S12 §8 と同じ計器)
        det_w = fbv2.crisis_detect(result, seed_config)
        fb_window_vecs.append(
            fbv2.crisis_window_counts(result, seed_config, 5.0, detection=det_w)
        )
        fb_nt_window_vecs.append(_nt_window_means(result, seed_config, 5.0))

        # クロス資産統計 (このシード)
        cm = cross_asset_metrics(multi, seed_config)
        sm = cm.get("summary") or {}
        for key_src, key_dst in (
            ("hy_max_abs_err", "x_hy_max_abs_err"),
            ("epps_ratio_median", "x_epps_ratio"),
            ("daily_corr_latent_max_abs_err", "x_daily_corr_err"),
            ("vol_corr_min", "x_vol_corr_min"),
            ("vol_corr_max", "x_vol_corr_max"),
            ("crisis_corr_increase_median", "x_crisis_corr_increase_seed"),
        ):
            per_seed.setdefault(key_dst, []).append(sm.get(key_src))
        per_seed.setdefault("x_epps_monotone", []).append(
            1.0 if sm.get("epps_monotone_all") else 0.0
        )
        per_seed.setdefault("x_vol_horizon_increasing", []).append(
            1.0 if sm.get("vol_corr_horizon_increasing_all") else 0.0
        )
        for pk, pv in (cm.get("pairs") or {}).items():
            ll = pv.get("lead_lag") or {}
            if ll.get("status") == "ok":
                per_seed.setdefault(f"x_leadlag_peak_{pk}", []).append(
                    float(ll["peak_lag"])
                )
                per_seed.setdefault(f"x_leadlag_asym_{pk}", []).append(
                    ll["asymmetry_i_leads"]
                )
            per_seed.setdefault(f"x_vol_corr_{pk}", []).append(pv.get("vol_corr_latent"))
            per_seed.setdefault(f"x_daily_corr_latent_{pk}", []).append(
                pv.get("daily_corr_latent")
            )
            per_seed.setdefault(f"x_daily_corr_obs_{pk}", []).append(
                pv.get("daily_corr_obs")
            )
        for pa in cm.get("per_asset") or []:
            ai = pa["asset_index"]
            per_seed.setdefault(f"x_throughput_a{ai}", []).append(
                pa.get("throughput_events_per_sec")
            )
            per_seed.setdefault(f"x_spread_a{ai}", []).append(
                pa.get("spread_median_ticks")
            )
            per_seed.setdefault(f"x_trades_a{ai}", []).append(float(pa["n_trades"]))

        # 危機時相関のプール素材 (§11: 2000 日以上 → シード横断プール)。
        # 3 条件付け: 和集合 (指示書の字義) / ブレッドス (≥2 資産同時 = 観測可能な
        # 市場危機日) / 潜在 big|z_F| (§7.1 の機構実在)。
        from .validation.cross import crisis_day_mask

        pl = multi.payloads
        n_dm = min(min(p.daily_ret_obs.size, p.n_days) for p in pl)
        masks_all = [
            crisis_day_mask(p.crisis_episodes, p.crisis_step_sec, n_dm) for p in pl
        ]
        breadth = np.sum(np.stack(masks_all), axis=0) >= 2
        fd = np.abs(np.asarray(multi.factor_daily)[:n_dm])
        bigf = fd > np.quantile(fd, 0.9)
        for a in range(len(pl)):
            for b in range(a + 1, len(pl)):
                key = f"{a}-{b}"
                n_dd = n_dm
                mask = masks_all[a][:n_dd] | masks_all[b][:n_dd]
                slot = pooled_pairs.setdefault(
                    key, {"di": [], "dj": [], "m": [], "mb": [],
                          "li": [], "lj": [], "mf": []}
                )
                slot["di"].append(pl[a].daily_ret_obs[:n_dd])
                slot["dj"].append(pl[b].daily_ret_obs[:n_dd])
                slot["m"].append(mask)
                slot["mb"].append(breadth[:n_dd])
                slot["li"].append(pl[a].daily_ret_latent[:n_dd])
                slot["lj"].append(pl[b].daily_ret_latent[:n_dd])
                slot["mf"].append(bigf[:n_dd])
        del multi, result
        print(f"      シード {seed} ({i + 1}/{n_seeds}) 完了", flush=True)

    out: dict[str, Any] = {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "skipped_seeds": skipped_seeds,
        "n_completed": n_seeds - sum(1 for s in skipped_seeds if s["leg"] == "main"),
    }
    if cross_seed_paths:
        from .validation.scaling import cross_seed_correlation

        out["cross_seed_corr"] = cross_seed_correlation(cross_seed_paths)
        out["chi_hash_all_equal"] = bool(len(set(chi_hashes)) == 1)
        out["chi_hashes"] = chi_hashes
        del cross_seed_paths
    if fb_window_vecs:
        from .validation.feedback import crisis_window_reproducibility

        out["fb_window_repro"] = crisis_window_reproducibility(fb_window_vecs)
        nt_ok = [v for v in fb_nt_window_vecs if v is not None]
        out["fb_nt_window_corr"] = crisis_window_reproducibility(nt_ok)
        if len(fb_window_vecs) >= 3:
            n_c = min(v.size for v in fb_window_vecs)
            M = np.stack([np.asarray(v)[:n_c] for v in fb_window_vecs])
            var_b = float(M.mean(axis=0).var())
            var_w = float(M.var(axis=0).mean())
            var_chi = max(var_b - var_w / M.shape[0], 0.0)
            out["fb_window_repro"]["icc_chi3_share"] = (
                var_chi / (var_chi + var_w) if (var_chi + var_w) > 0 else None
            )
    # 危機時相関 (プール): ペアごとに全シードの日次系列を連結して条件付き相関
    pooled_out: dict[str, Any] = {}
    increases: list[float] = []
    inc_breadth: list[float] = []
    inc_bigf: list[float] = []
    for key, slot in pooled_pairs.items():
        di = np.concatenate(slot["di"])
        dj = np.concatenate(slot["dj"])
        cc = conditional_correlation(di, dj, np.concatenate(slot["m"]))
        cc_b = conditional_correlation(di, dj, np.concatenate(slot["mb"]))
        cc_f = conditional_correlation(
            np.concatenate(slot["li"]), np.concatenate(slot["lj"]),
            np.concatenate(slot["mf"]),
        )
        pooled_out[key] = {"union": cc, "breadth": cc_b, "latent_bigf": cc_f}
        if cc.get("status") == "ok":
            increases.append(cc["increase"])
        if cc_b.get("status") == "ok":
            inc_breadth.append(cc_b["increase"])
        if cc_f.get("status") == "ok":
            inc_bigf.append(cc_f["increase"])
    out["x_crisis_corr_pooled"] = pooled_out
    out["x_crisis_corr_increase_pooled_median"] = (
        float(np.median(increases)) if increases else None
    )
    out["x_crisis_corr_increase_breadth_pooled_median"] = (
        float(np.median(inc_breadth)) if inc_breadth else None
    )
    out["x_crisis_corr_increase_latent_bigf_pooled_median"] = (
        float(np.median(inc_bigf)) if inc_bigf else None
    )
    out["x_crisis_pooled_days"] = (
        int(sum(v.size for v in next(iter(pooled_pairs.values()))["di"]))
        if pooled_pairs else 0
    )

    for name, values in per_seed.items():
        clean = [v for v in values if v is not None]
        if not clean:
            out[name] = {"median": None, "values": values}
            continue
        arr = np.array(clean, dtype=np.float64)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        out[name] = {
            "median": float(np.median(arr)),
            "iqr": q3 - q1, "q1": q1, "q3": q3,
            "min": float(arr.min()), "max": float(arr.max()),
            "values": values,
        }
    return out


def _s12_carryover(results_dir: str | None) -> dict[str, Any]:
    """S12 本番の multiseed 中央値からペア計器の値を繰り越す (§11)。

    繰り越しの妥当性は n1_regression (ビット単位一致) が担保する: ループ機構の
    コードパスは S12 と同一で、多資産の結合 (因子合成・χ 共有) はフィードバック
    経路に入らない。値は「参照資産の単変量性質」として S12 の実測をそのまま使う。
    """
    from .report import load_metrics

    keys = (
        "fb_g_30min", "fb_g_daily", "fb_divergences", "fb_T_daily_off",
        "fb_depth_cv_ratio", "fb_rv_excess_ari", "fb_rv_excess_geo",
        "fb_crises_per_year_off", "meta_gamma_ice_off", "meta_c1_ice_off",
        "meta_c1", "meta_gamma",
        # ③ と Hill: 500 日では計器が検定力不足 (③ IQR 0.185 / hill k=18 点で
        # IQR 0.78 — 事前測定 #3 実測)。1000 日 × 30 の S12 実測を繰り越す。
        "fb_gph_d_diff_masked", "fb_hill_ex_crisis", "fb_hill_all",
    )
    try:
        base = load_metrics("S12", root=results_dir)
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    ms = (base.get("metrics") or {}).get("multiseed") or {}
    out: dict[str, Any] = {"available": True, "source": "results/S12/metrics.json",
                           "source_git": base.get("git_commit")}
    for k in keys:
        info = ms.get(k) or {}
        out[k] = {kk: info.get(kk) for kk in ("median", "q1", "q3", "min", "max")}
    return out


def _run_s13(args: argparse.Namespace, config: Config) -> int:
    """S13 (多資産) の run フロー。単一資産の cmd_run と同じ段取りで、
    実行を run_multi に、構造検査を n1/退化/資産追加のビット単位検査に替える。"""
    import numpy as np

    from .pipeline import (
        asset_addition_check,
        factor_degeneracy_check,
        n1_regression_check,
        run as run_pipeline_single,
        run_multi,
    )
    from .validation.cross import cross_asset_metrics

    started = time.perf_counter()
    stage = config.stage
    from .report import git_info

    git_at_start = git_info()
    print(f"[1/6] 多資産実行 stage={stage} seed={config.seed} n_assets={config.n_assets} "
          f"n_days={config.n_days} betas={config.factor_betas}")
    multi = run_multi(config)
    result = multi.asset0
    print(f"      完了 ({multi.runtime_sec:.2f} 秒, 資産 {config.n_assets} 本)")

    print("[2/6] 決定性の確認 (run_multi を再実行し全資産のダイジェスト照合)")
    multi2 = run_multi(config)
    det_ok = bool(multi.digests == multi2.digests)
    determinism = {
        "bitwise_identical": det_ok,
        "digests_match": det_ok,
        "per_asset": {str(k): bool(multi.digests[k] == multi2.digests[k])
                      for k in multi.digests},
    }
    del multi2
    print(f"      ビット単位一致: {det_ok}")

    print("[3/6] RNG 安定性 + 多資産の構造検査 (§8)")
    rng_stability = rng_stability_check(config)
    rng_diffusion = rng_diffusion_check(config, result)
    print(f"      既存ストリーム不変: {rng_stability['unchanged']} / "
          f"l2.diffusion 一致 (資産0): {rng_diffusion['match']}")
    cfg_n1 = config.n1_config()
    r_n1 = run_pipeline_single(cfg_n1)
    n1 = n1_regression_check(config, results_root=args.results_dir, result=r_n1)
    print(f"      n1 回帰 (S12 ダイジェスト照合): {n1.get('match')}")
    degen = factor_degeneracy_check(config)
    print(f"      因子経路の退化 (β=0 ビット単位): {degen['match']}")
    addition = asset_addition_check(config)
    print(f"      資産追加の不変性 (N={config.n_assets}→{config.n_assets + 1}): "
          f"{addition['bitwise']}")
    # L2 凍結 (板は L2 を読むが書かない) — 板 off の多資産ランと潜在側を照合。
    # ★流動性オーバーライドも外す: 許可キーは全て L3/L1 のパラメータで L2 には
    # 一切入らない (config の _S13_OVERRIDE_KEYS 検証がそれを保証している) ため、
    # 外しても L2 側の比較対象は変わらない — むしろ「オーバーライドが L2 に
    # 漏れていない」ことの検証を兼ねる。
    multi_off = run_multi(config.without_book().replace(asset_overrides=()))
    l2_frozen = {
        "passed": True, "per_asset": {},
        "basis": "板 off の run_multi との潜在 (日次リターン + log σ サブサンプル) 照合",
    }
    for i_, p_ in enumerate(multi.payloads):
        of_p = multi_off.payloads[i_]
        same = bool(
            np.array_equal(p_.daily_ret_latent, of_p.daily_ret_latent)
            and np.array_equal(p_.log_vol_sub, of_p.log_vol_sub)
        )
        l2_frozen["per_asset"][str(i_)] = same
        l2_frozen["passed"] = l2_frozen["passed"] and same
    del multi_off
    print(f"      L2 凍結 (全資産): {l2_frozen['passed']}")

    # 時間スケール不変性は n1 退化脚で判定する (L2 生成器の物理時間定義は
    # 因子分割後も同一の generator — 共通/固有とも simulate_msm_path /
    # simulate_ou_path を通る。多資産での再測定は板実現の独立性で S10 と
    # 同じ理由により検定にならない)。
    try:
        scale_invariance = scale_invariance_check(cfg_n1, r_n1)
    except RuntimeError as exc:
        scale_invariance = {
            "passed": None, "skipped": f"対照解像度ランが失敗: {exc}", "checks": {},
        }
    del r_n1

    print("[4/6] 検証スイート (参照資産) + クロス資産測定")
    validation_started = time.perf_counter()
    # §9: suite の cross.hayashi_yoshida を初めて有効化する (p* を各資産の
    # 実約定時刻でサンプルした系列 — 推定量と非同期性の検定対象)
    result.meta["assets"] = [
        (p.trade_t, p.pstar_at_trades) for p in multi.payloads[:2]
    ]
    metrics = run_all(result, config)
    errors = collect_errors(metrics)
    metrics["cross_assets"] = cross_asset_metrics(multi, config)
    metrics["s12_carryover"] = _s12_carryover(args.results_dir)

    multi_runtime_sec = multi.runtime_sec
    asset_digests = {str(k): v for k, v in multi.digests.items()}
    multiseed = None
    if args.seeds and args.seeds > 1:
        print(f"[4c/6] 多シード判定 ({args.seeds} シード — run_multi ループ)")
        del multi  # メモリ: ループ中はメインの結果を保持しない (metrics に抽出済み)
        multiseed = _run_multiseed_s13(config, args.seeds)
        for name in (
            "x_hy_max_abs_err", "x_epps_ratio", "x_daily_corr_err",
            "x_vol_corr_min", "x_crisis_corr_increase_seed",
            "hill_alpha", "fb_hill_ex_crisis", "fb_gph_d_diff_masked",
            "fb_nt_mean_time", "fb_crises_per_year", "qr_eta_trade",
        ):
            info = multiseed.get(name) or {}
            if isinstance(info, dict) and info.get("median") is not None:
                print(f"      {name}: median={info['median']:+.4f}  IQR={info['iqr']:.4f}")
        pooled = multiseed.get("x_crisis_corr_increase_pooled_median")
        if pooled is not None:
            print(f"      crisis_corr_increase (プール): {pooled:+.4f}")
        metrics["multiseed"] = multiseed

    # クロスのゲート値: 多シードがあればその中央値/プール、無ければ主シード値
    cs = metrics["cross_assets"].get("summary") or {}
    ms = multiseed or {}
    def med(key: str):
        info = ms.get(key) or {}
        return info.get("median") if isinstance(info, dict) else None
    metrics["cross_assets"]["gatevals"] = {
        "hy_max_abs_err": (
            (ms.get("x_hy_max_abs_err") or {}).get("max")
            if multiseed else cs.get("hy_max_abs_err")
        ),
        "epps_ratio": med("x_epps_ratio") if multiseed else cs.get("epps_ratio_median"),
        "epps_monotone": (
            bool((med("x_epps_monotone") or 0) >= 1.0)
            if multiseed else cs.get("epps_monotone_all")
        ),
        # ★per-seed の max 誤差はサンプリング雑音 (1000 日で SE≈0.032、3 ペア
        # max の期待値 ~0.05) がゲート幅 ±0.05 を食い潰す — シード中央値の
        # 相関 vs 理論で判定する (5 シード中央値の SE ≈ 0.018、ペア max ~0.04)。
        "daily_corr_err": (
            max(
                (
                    abs(med(f"x_daily_corr_latent_{pk}") - tv)
                    for pk, tv in (
                        (metrics["cross_assets"].get("theory") or {}).get("pairs") or {}
                    ).items()
                    if med(f"x_daily_corr_latent_{pk}") is not None and tv is not None
                ),
                default=None,
            )
            if multiseed
            else cs.get("daily_corr_latent_max_abs_err")
        ),
        "vol_corr_lo": (
            med("x_vol_corr_min") if multiseed else cs.get("vol_corr_min")
        ),
        "vol_corr_hi": (
            med("x_vol_corr_max") if multiseed else cs.get("vol_corr_max")
        ),
        # ペア中央値の中央値 (vol_corr ゲートの判定値 — min はペア間の
        # エポックゆらぎで下振れしやすく、500 日窓の検定にならない)
        "vol_corr_med": (
            (lambda vals: float(np.median(vals)) if vals else None)(
                [
                    med(k) for k in ms
                    if isinstance(k, str) and k.startswith("x_vol_corr_")
                    and k.count("-") == 1 and med(k) is not None
                ]
            )
            if multiseed
            else cs.get("vol_corr_min")
        ),
        "vol_corr_horizon_increasing": (
            bool((med("x_vol_horizon_increasing") or 0) >= 1.0)
            if multiseed else cs.get("vol_corr_horizon_increasing_all")
        ),
        "crisis_corr_increase": (
            ms.get("x_crisis_corr_increase_pooled_median")
            if multiseed else cs.get("crisis_corr_increase_median")
        ),
        "crisis_corr_increase_breadth": (
            ms.get("x_crisis_corr_increase_breadth_pooled_median")
            if multiseed else cs.get("crisis_corr_increase_breadth_median")
        ),
        "crisis_corr_increase_latent_bigf": (
            ms.get("x_crisis_corr_increase_latent_bigf_pooled_median")
            if multiseed else cs.get("crisis_corr_increase_latent_bigf_median")
        ),
        "leadlag_asym_1_2": med("x_leadlag_asym_1-2") if multiseed else (
            ((metrics["cross_assets"].get("pairs") or {}).get("1-2") or {})
            .get("lead_lag", {}).get("asymmetry_i_leads")
        ),
        "leadlag_peak_1_2": med("x_leadlag_peak_1-2") if multiseed else (
            ((metrics["cross_assets"].get("pairs") or {}).get("1-2") or {})
            .get("lead_lag", {}).get("peak_lag")
        ),
    }

    metrics["runtime"] = {
        "pipeline": {
            "completed": True,
            "runtime_sec": multi_runtime_sec,
            "driver": "multi_asset",
            "layers": result.meta.get("layers"),
            "grid": result.meta.get("grid"),
            "rng_streams_used": result.meta.get("rng_streams_used"),
            "environment": result.meta.get("environment"),
            "result_digest": result.digest(),
            "asset_digests": asset_digests,
        },
        "determinism": determinism,
        "rng_stability": rng_stability,
        "rng_diffusion": rng_diffusion,
        **({"scale_invariance": scale_invariance} if scale_invariance is not None else {}),
        "multi_asset_checks": {
            "n1_regression": n1,
            "factor_degeneracy": degen,
            "asset_addition": addition,
            "l2_frozen_multi": l2_frozen,
        },
        "validation": {
            "all_callable": not errors,
            "n_errors": len(errors),
            "errors": errors,
            "runtime_sec": time.perf_counter() - validation_started,
        },
        "artifacts": {"metrics_json_ok": False, "reason": "書き出し前"},
        "rng_fingerprint": result.rng_fingerprint,
    }
    print(f"      指標の算出完了 ({metrics['runtime']['validation']['runtime_sec']:.2f} 秒, "
          f"エラー {len(errors)} 件)")

    gates = gates_for(stage)
    gate_results = evaluate(gates, metrics)
    summary = summarize(gate_results)

    print("[5/6] 結果の書き出しと読み直し")
    total_runtime = time.perf_counter() - started
    path = write_metrics(
        stage, config, metrics, gate_results, summary, total_runtime,
        root=args.results_dir, git=git_at_start,
    )
    verification = verify_metrics_file(path)
    metrics["runtime"]["artifacts"] = verification
    gate_results = evaluate(gates, metrics)
    summary = summarize(gate_results)
    total_runtime = time.perf_counter() - started
    path = write_metrics(
        stage, config, metrics, gate_results, summary, total_runtime,
        root=args.results_dir, git=git_at_start,
    )
    print(f"      {path} ({verification.get('size_bytes', 0):,} バイト)")

    if not args.no_plots:
        print("[6/6] プロット")
        plots = make_plots(metrics, stage, result=result, root=args.results_dir)
        for plot_path in plots:
            print(f"      {plot_path.name}")
    else:
        print("[6/6] プロットは --no-plots によりスキップ")

    _print_key_metrics(metrics)
    _print_na(metrics)
    _print_gates(gate_results, summary)
    print(f"\n所要時間 {total_runtime:.1f} 秒")
    return 0 if summary["all_critical_passed"] else 1


def _print_gates(gate_results, summary: Mapping[str, Any]) -> None:
    name_width = max(len(g.name) for g in gate_results)
    print()
    print("ゲート判定")
    print("-" * (name_width + 58))
    for g in gate_results:
        mark = "PASS" if g.passed else ("FAIL" if g.critical else "WARN")
        tag = "" if g.critical else " (warning)"
        line = f"  [{mark}] {g.name.ljust(name_width)}  {_fmt(g.value)}"
        print(line)
        if not g.passed:
            print(f"         期待: {g.threshold}{tag}")
            if g.error:
                print(f"         理由: {g.error}")
    print("-" * (name_width + 58))
    print(
        f"  合格 {summary['n_passed']}/{summary['n_gates']}"
        f" / critical {summary['n_critical']} 件中 "
        f"{'全て合格' if summary['all_critical_passed'] else '不合格あり: ' + ', '.join(summary['failed_critical'])}"
    )


def _print_key_metrics(metrics: Mapping[str, Any]) -> None:
    def get(path: str) -> Any:
        node: Any = metrics
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return None
            node = node[part]
        return node

    rows = [
        ("基準粒度 (秒)", get("series.primary_bar_sec")),
        ("リターン本数", get("series.n_primary_returns")),
        ("尖度", get("tails.moments.kurtosis")),
        ("歪度", get("tails.moments.skewness")),
        ("Hill alpha (k=5%)", get("tails.hill.alpha")),
        ("Hill 不安定度", get("tails.hill_profile.instability")),
        ("QQ の R^2", get("tails.qq_normal.r2")),
        ("ACF(r) ラグ1", get("memory.acf_r.lag1")),
        ("ACF(|r|) ラグ1", get("memory.acf_abs_r.lag1")),
        ("Ljung-Box p (ラグ20)", get("memory.ljung_box_r.pvalue_primary")),
        ("GPH d (|r|)", get("memory.gph_abs_r.d")),
        ("local Whittle d (|r|)", get("memory.local_whittle_abs_r.d")),
        ("分散比 max|VR-1|", get("scaling.variance_ratio.max_abs_dev")),
        ("尖度のスケール依存 max|k-3|", get("scaling.kurtosis_by_scale.max_abs_dev_from_3_gated")),
        ("zeta_q 直線性 R^2", get("scaling.zeta_q.r2")),
        ("zeta_q の傾き", get("scaling.zeta_q.slope")),
        ("signature plot 最大乖離", get("scaling.signature_plot.max_rel_dev")),
        ("ADF p (log P)", get("scaling.adf.log_price_pvalue")),
        ("ADF p (r)", get("scaling.adf.returns_pvalue")),
        ("日次 尖度", get("daily.moments.kurtosis")),
        ("日次 ACF(|r|) ラグ1", get("daily.acf_abs_r.lag1")),
        ("日次 GPH d (|r|)", get("daily.gph_abs_r.d")),
        ("日次 |r| ACF べき則 R^2", get("daily.acf_abs_r_powerlaw.r2")),
        ("日次 zeta_q 直線性 R^2", get("daily.zeta_q.r2")),
        ("尖度の減衰傾き (日次→複数日)", get("daily.kurtosis_decay.decay_slope")),
        ("Var(log σ) 断面", get("vol.ensemble.var_log_sigma")),
        ("予算シェア 断面 (分母 0.25)", get("vol.ensemble.shares_of_budget")),
        ("予算使用率 断面", get("vol.ensemble.budget_used_fraction")),
        ("E[σ²]/σ̄² 断面", get("vol.ensemble.e_sigma2_ratio")),
        ("粗さ H (潜在, 5分〜4時間)", get("rough.h_latent.h")),
        ("粗さ ζ_q 線形性 R²", get("rough.h_latent.linearity_r2")),
        ("粗さ H (RV 側, 記録のみ)", get("rough.h_rv.h")),
        ("ボラ増分 ACF(1) (60秒)", get("rough.increment_acf.lag1")),
        ("ラフ予算シェア (経路)", get("rough.share_of_budget_path.value")),
    ]
    width = max(len(label) for label, _ in rows)
    print()
    print("主要指標")
    print("-" * (width + 24))
    for label, value in rows:
        print(f"  {label.ljust(width)}  {_fmt(value)}")


def _print_na(metrics: Mapping[str, Any]) -> None:
    """該当なしの指標を一覧する (「測っていない」ことを可視化する)。"""
    entries: list[str] = []
    for group in ("micro", "cross"):
        for name, node in metrics.get(group, {}).items():
            if isinstance(node, Mapping) and node.get("status") == "not_applicable":
                entries.append(f"  {group}.{name}: {node.get('reason')}")
    if entries:
        print()
        print("該当なし (not_applicable) の指標")
        print("-" * 70)
        for line in entries:
            print(line)


# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = _build_config(args)
    if config.n_assets > 1:
        return _run_s13(args, config)
    stage = config.stage
    # ★コード版数は**実行開始時点**で確定する。書き出し時に取ると自分の出力
    # (metrics.json) が status に映って dirty が常に True になる (report.py 参照)。
    from .report import git_info

    git_at_start = git_info()
    print(f"[1/6] 実行 stage={stage} seed={config.seed} "
          f"n_days={config.n_days} steps_per_day={config.steps_per_day}")

    result = run_pipeline(config)
    print(f"      完了 ({result.runtime_sec:.2f} 秒, グリッド {result.price.n_points:,} 点)")

    print("[2/6] 決定性の確認 (同一シードで再実行)")
    determinism = determinism_check(config, first=result)
    print(f"      ビット単位一致: {determinism['bitwise_identical']}")

    print("[3/6] RNG ストリーム安定性の確認")
    rng_stability = rng_stability_check(config)
    print(f"      既存ストリーム不変: {rng_stability['unchanged']} / "
          f"相互に別系列: {rng_stability['streams_distinct']}")

    rng_diffusion = rng_diffusion_check(config, result)
    print(f"      l2.diffusion 消費列の一致: {rng_diffusion['match']}")

    scale_invariance = None
    if stage != "S0":
        low_steps = config.validation.scale_invariance_steps_per_day
        print(f"[3b/6] 時間スケール不変性 (steps_per_day={low_steps} で対照実行)")
        # ★S10 (κ>0): 対照解像度は独立な板実現なので、本走が完走しても
        # 窓逸脱で落ちうる。落ちたら記録の上スキップ (潜在側の判定は
        # 本番 30 シードの multiseed が別途担う)。
        try:
            scale_invariance = scale_invariance_check(config, result)
        except RuntimeError as exc:
            scale_invariance = {
                "passed": None,
                "skipped": f"対照解像度ランが失敗: {exc}",
                "checks": {},
            }
            print(f"      対照実行が失敗 — 記録の上スキップ: {exc}")
        else:
            print(f"      日次統計の一致: {scale_invariance['passed']}")
        for name, chk in scale_invariance["checks"].items():
            if not chk["passed"]:
                print(f"        不一致: {name}  hi={chk.get('hi')}  lo={chk.get('lo')}")

    print("[4/6] 検証スイート")
    validation_started = time.perf_counter()
    metrics = run_all(result, config)
    errors = collect_errors(metrics)

    baseline_inv = None
    if stage in BASELINE_STAGE:
        base_stage = BASELINE_STAGE[stage]
        print(f"[4b/6] {base_stage} からの不変性照合 (results/{base_stage}/metrics.json)")
        baseline_inv = baseline_invariance_check(
            config, metrics, base_stage, results_root=args.results_dir, result=result
        )
        if baseline_inv.get("error"):
            print(f"      基準が読めません: {baseline_inv['error']}")
        else:
            print(f"      不変性: {baseline_inv['passed']}")
            for name, chk in baseline_inv["checks"].items():
                if not chk.get("passed"):
                    print(f"        不一致: {name}  {chk}")

    multiseed = None
    if args.seeds and args.seeds > 1:
        print(f"[4c/6] 多シード判定 ({args.seeds} シード — hill/leverage/JV/skew)")
        multiseed = _run_multiseed(config, args.seeds)
        for name in (
            "hill_alpha", "leverage_corr", "jv_share", "skewness_daily",
            "dilution_sd_ratio", "dilution_corr_logiv",
            "hawkes_n_hat_true_phi", "hawkes_fano_60s",
            "meta_gamma", "meta_c1", "meta_vr_trade_1000", "meta_beta_deficit",
            "qr_eta_trade", "qr_obi_h1", "qr_change_sign_corr",
        ):
            info = multiseed.get(name) or {}
            if info.get("median") is not None:
                print(f"      {name}: median={info['median']:+.4f}  IQR={info['iqr']:.4f}")
        csc = multiseed.get("cross_seed_corr")
        if isinstance(csc, dict) and csc.get("mean") is not None:
            print(
                f"      cross_seed_corr: mean={csc['mean']:.4f} "
                f"[{csc['min']:.4f}, {csc['max']:.4f}] ({csc['n_pairs']} 対)"
            )
        metrics["multiseed"] = multiseed

    metrics["runtime"] = {
        "pipeline": {
            "completed": True,
            "runtime_sec": result.runtime_sec,
            "driver": result.meta.get("driver"),
            "layers": result.meta.get("layers"),
            "grid": result.meta.get("grid"),
            "rng_streams_used": result.meta.get("rng_streams_used"),
            "environment": result.meta.get("environment"),
            "result_digest": result.digest(),
        },
        "determinism": determinism,
        "rng_stability": rng_stability,
        "rng_diffusion": rng_diffusion,
        **({"scale_invariance": scale_invariance} if scale_invariance is not None else {}),
        **({"baseline_invariance": baseline_inv} if baseline_inv is not None else {}),
        "validation": {
            "all_callable": not errors,
            "n_errors": len(errors),
            "errors": errors,
            "runtime_sec": time.perf_counter() - validation_started,
        },
        "artifacts": {"metrics_json_ok": False, "reason": "書き出し前"},
        "rng_fingerprint": result.rng_fingerprint,
    }
    print(f"      指標の算出完了 ({metrics['runtime']['validation']['runtime_sec']:.2f} 秒, "
          f"エラー {len(errors)} 件)")

    gates = gates_for(stage)
    gate_results = evaluate(gates, metrics)
    summary = summarize(gate_results)

    print("[5/6] 結果の書き出しと読み直し")
    total_runtime = time.perf_counter() - started
    path = write_metrics(
        stage, config, metrics, gate_results, summary, total_runtime,
        root=args.results_dir, git=git_at_start,
    )
    verification = verify_metrics_file(path)
    metrics["runtime"]["artifacts"] = verification
    gate_results = evaluate(gates, metrics)
    summary = summarize(gate_results)
    total_runtime = time.perf_counter() - started
    path = write_metrics(
        stage, config, metrics, gate_results, summary, total_runtime,
        root=args.results_dir, git=git_at_start,
    )
    print(f"      {path} ({verification.get('size_bytes', 0):,} バイト)")

    if not args.no_plots:
        print("[6/6] プロット")
        plots = make_plots(metrics, stage, result=result, root=args.results_dir)
        for plot_path in plots:
            print(f"      {plot_path.name}")
    else:
        print("[6/6] プロットは --no-plots によりスキップ")

    _print_key_metrics(metrics)
    _print_na(metrics)
    _print_gates(gate_results, summary)
    print(f"\n所要時間 {total_runtime:.1f} 秒")
    return 0 if summary["all_critical_passed"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    stage = args.stage
    data = load_metrics(stage, root=args.results_dir)
    metrics = data.get("metrics", {})

    # 保存済みの artifacts 判定を鵜呑みにせず、いま実際にファイルがあるかで上書きする。
    path = Path(args.results_dir or (Path(__file__).resolve().parent.parent / "results"))
    metrics.setdefault("runtime", {})["artifacts"] = verify_metrics_file(
        path / stage / "metrics.json"
    )

    gate_results = evaluate(gates_for(stage), metrics)
    summary = summarize(gate_results)
    print(f"stage={stage}  git_commit={data.get('git_commit')}  "
          f"config_hash={(data.get('config_hash') or '')[:12]}")
    print(f"作成日時 {data.get('created_at')}  実行時間 {data.get('runtime_sec')}")
    _print_key_metrics(metrics)
    _print_gates(gate_results, summary)
    if summary["all_critical_passed"] != data.get("all_critical_passed"):
        print("\n注意: 保存時の判定と再判定の結果が食い違っています "
              f"(保存時 {data.get('all_critical_passed')} / 再判定 {summary['all_critical_passed']})。"
              " ゲート定義が変わった可能性があります。")
    return 0 if summary["all_critical_passed"] else 1


def cmd_compare(args: argparse.Namespace) -> int:
    stages: Sequence[str] = args.stages
    if len(stages) < 2:
        print("compare には 2 つ以上の段階が必要です", file=sys.stderr)
        return 2
    diff = compare_stages(stages, root=args.results_dir, only_changed=args.only_changed)

    print("段階間比較: " + " vs ".join(stages))
    print()
    print("  " + "指標".ljust(52) + "  " + "  ".join(s.rjust(14) for s in stages)
          + ("      差分" if len(stages) == 2 else ""))
    print("-" * (54 + 16 * len(stages) + 12))
    for row in diff["metrics"]:
        cells = "  ".join(_fmt(row["values"][s]).rjust(14) for s in stages)
        delta = f"  {_fmt(row['delta']).rjust(12)}" if len(stages) == 2 else ""
        print(f"  {row['metric'][:52].ljust(52)}  {cells}{delta}")

    print()
    print("  " + "ゲート".ljust(52) + "  " + "  ".join(s.rjust(14) for s in stages))
    print("-" * (54 + 16 * len(stages)))
    for row in diff["gates"]:
        cells = "  ".join(_fmt(row.get(s)).rjust(14) for s in stages)
        print(f"  {row['gate'][:52].ljust(52)}  {cells}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(diff, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"\n{args.json} に書き出しました")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simchart.cli",
        description="段階構築式マイクロ構造シミュレータ (S0 骨格層)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="シミュレーション・検証・ゲート判定・永続化")
    run_parser.add_argument("--config", type=str, default=None, help="設定 YAML / JSON")
    run_parser.add_argument("--stage", type=str, default=None, help="段階名 (既定は設定ファイルの値)")
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--n-days", type=int, default=None)
    run_parser.add_argument("--steps-per-day", type=int, default=None)
    run_parser.add_argument("--results-dir", type=str, default=None, help="results/ の位置")
    run_parser.add_argument("--no-plots", action="store_true")
    run_parser.add_argument(
        "--seeds", type=int, default=None,
        help="ノイズの大きい指標 (hill/leverage/JV/skew) を N シードの中央値で判定する (S3+)",
    )
    run_parser.set_defaults(func=cmd_run)

    validate_parser = sub.add_parser("validate", help="保存済み結果のゲート再判定")
    validate_parser.add_argument("--stage", type=str, required=True)
    validate_parser.add_argument("--results-dir", type=str, default=None)
    validate_parser.set_defaults(func=cmd_validate)

    compare_parser = sub.add_parser("compare", help="段階間の指標差分")
    compare_parser.add_argument("--stages", type=str, nargs="+", required=True)
    compare_parser.add_argument("--results-dir", type=str, default=None)
    compare_parser.add_argument("--only-changed", action="store_true", help="差が 0 の指標を省く")
    compare_parser.add_argument("--json", type=str, default=None, help="比較結果の書き出し先")
    compare_parser.set_defaults(func=cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NotImplementedError as exc:
        print(f"\n未実装のため停止しました:\n  {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
