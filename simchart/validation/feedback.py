"""S11: フィードバックと内生的危機の測定群 (指示書 §9)。

- **loop_gain_estimate**: 同一シード on/off の log RV 分散比から g を推定 (§4.1)。
  ★ループが増幅するのは**短期帯域** (RV_long が追随する緩慢帯域は u に現れず
  増幅されない) ので、日次版 (指示書の式そのまま) と短期帯域版の両方を出す。
- **crisis_detect / crisis_anatomy**: 3 条件 (価格・スプレッド・デプス) の同時
  成立をエピソード化し、頻度・継続・深さ・回復率を測る (§6)。
- **divergence_monitor**: RV の発散検出 (§10 no_divergence)。
- **nt_distribution**: u 系列から n_t の実現分布を再構成 (§8.3)。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import na, num, ok

__all__ = [
    "loop_gain_estimate",
    "crisis_detect",
    "crisis_anatomy",
    "divergence_monitor",
    "nt_distribution",
]

_SESSION = 23400.0


def _log_rv_series(obs, cfg, window_sec: float) -> np.ndarray:
    """窓ごとの log RV (脱季節化なし — on/off 比なので季節性は相殺する)。"""
    step = float(obs.step_seconds)
    burn = int(cfg.book_burn_in_days * _SESSION / step)
    r = np.diff(np.asarray(obs.log_price)[burn:])
    w = max(1, int(round(window_sec / step)))
    n_w = r.size // w
    rv = (r[: n_w * w] ** 2).reshape(n_w, w).sum(axis=1)
    return np.log(np.maximum(rv, 1e-18))


def loop_gain_estimate(result_on, result_off, cfg) -> dict[str, Any]:
    """g = 1 − sqrt(Var_off(log RV)/Var_on(log RV)) (指示書 §4.1)。

    日次 (指示書そのまま) と 30 分帯域 (ループが実際に増幅する側) の両方。
    同一シード・同一 L2 経路で呼ぶこと (呼び出し側の責任)。
    """
    rows: dict[str, Any] = {}
    for label, win in (("daily", _SESSION), ("30min", 1800.0)):
        lon = _log_rv_series(result_on.observation, cfg, win)
        loff = _log_rv_series(result_off.observation, cfg, win)
        v_on = float(lon.var())
        v_off = float(loff.var())
        if v_on <= 0:
            return na("Var_on が非正です")
        g = 1.0 - float(np.sqrt(v_off / v_on))
        rows[f"g_{label}"] = num(g)
        rows[f"var_on_{label}"] = num(v_on)
        rows[f"var_off_{label}"] = num(v_off)
    return ok(rows["g_30min"], **rows)


def _ewma(x: np.ndarray, lam: float) -> np.ndarray:
    """バイアス補正つき EWMA (先頭から因果)。NaN は**更新をスキップ**する
    (定数で埋めると単位依存の汚染になる — 実測でスプレッドはドル単位)。"""
    out = np.empty_like(x)
    acc = 0.0
    w = 0.0
    prev = np.nan
    for i in range(x.size):
        xi = x[i]
        if np.isfinite(xi):
            acc = lam * acc + (1.0 - lam) * xi
            w = lam * w + (1.0 - lam)
        out[i] = acc / w if w > 0 else (xi if np.isfinite(xi) else prev)
        prev = out[i]
    return out


def crisis_detect(result, cfg) -> dict[str, Any]:
    """3 条件 (§6.1) の同時成立をスナップショット格子 (60s) 上でエピソード化。

    「通常」はどの条件も**長期 EWMA (RV_long と同じ半減期)** を基準にする —
    市場参加者が直近履歴から通常水準を推定するという §2.2 と同じ解釈。
    """
    obs = result.observation
    book = result.book
    step_snap = float(book.t[1] - book.t[0]) if book.t.size > 1 else 60.0
    burn = int(cfg.book_burn_in_days * _SESSION / step_snap)
    bb = np.asarray(book.bid_px[:, 0], dtype=np.float64)
    ba = np.asarray(book.ask_px[:, 0], dtype=np.float64)
    valid = (bb >= 0) & (ba >= 0)
    spread = np.where(valid, ba - bb, np.nan)
    depth = np.asarray(book.bid_sz, dtype=np.float64).sum(axis=1) + np.asarray(
        book.ask_sz, dtype=np.float64
    ).sum(axis=1)

    # 5 分リターン (スナップショット格子に整列)
    step_o = float(obs.step_seconds)
    stride5 = max(1, int(round(300.0 / step_o)))
    lp = np.asarray(obs.log_price)
    idx_snap = np.minimum(
        (np.asarray(book.t) / step_o).astype(np.int64), lp.size - 1
    )
    lp_snap = lp[idx_snap]
    k5 = max(1, int(round(300.0 / step_snap)))
    r5 = np.zeros_like(lp_snap)
    r5[k5:] = lp_snap[k5:] - lp_snap[:-k5]

    lam = float(0.5 ** (step_snap / (cfg.fb_rv_long_halflife_days * _SESSION)))
    sig5 = np.sqrt(np.maximum(_ewma(r5**2, lam), 1e-18))
    sp_norm = _ewma(spread, lam)
    dp_norm = np.maximum(_ewma(depth, lam), 1e-12)

    k = float(cfg.crisis_k_sigma)
    m = float(cfg.crisis_spread_mult)
    cond_price = np.abs(r5) > k * sig5
    cond_spread = spread > m * sp_norm
    cond_depth = depth < dp_norm / m
    hit = cond_price & np.nan_to_num(cond_spread, nan=False) & cond_depth
    hit[:burn] = False

    # エピソード化 (5 分未満の隙間は連結)
    gap_join = max(1, int(round(300.0 / step_snap)))
    episodes: list[tuple[int, int]] = []
    i = 0
    n = hit.size
    while i < n:
        if hit[i]:
            j = i
            last = i
            while j < n:
                if hit[j]:
                    last = j
                    j += 1
                elif j - last <= gap_join:
                    j += 1
                else:
                    break
            episodes.append((i, last))
            i = j
        else:
            i += 1

    years = (cfg.n_days - cfg.book_burn_in_days) / 252.0
    return ok(
        num(len(episodes) / years if years > 0 else None),
        n_episodes=int(len(episodes)),
        per_year=num(len(episodes) / years if years > 0 else None),
        episodes=[[int(a), int(b)] for a, b in episodes],
        cond_rates={
            "price": num(float(cond_price[burn:].mean())),
            "spread": num(float(np.nan_to_num(cond_spread, nan=False)[burn:].mean())),
            "depth": num(float(cond_depth[burn:].mean())),
            "joint": num(float(hit[burn:].mean())),
        },
        step_sec=num(step_snap),
    )


def crisis_anatomy(result, cfg, detection: dict[str, Any] | None = None) -> dict[str, Any]:
    """エピソードごとの継続時間・深さ (スプレッド倍率/デプス比)・回復率 (§6.2)。

    回復率 = (30 分後の価格 − 極値) / (開始 − 極値)。1 = 全戻し、0 = 戻らず。
    フラッシュ・クラッシュは大きく戻すのが特徴 — 戻らないならそれはトレンドで、
    フィードバックが平均回帰 (S9/S10) を壊している疑い (§6.3)。
    """
    det = detection if detection is not None else crisis_detect(result, cfg)
    if det.get("status") != "ok" or not det.get("episodes"):
        return na("エピソードがありません")
    obs = result.observation
    book = result.book
    step_snap = det["step_sec"]
    step_o = float(obs.step_seconds)
    lp = np.asarray(obs.log_price)
    idx_snap = np.minimum((np.asarray(book.t) / step_o).astype(np.int64), lp.size - 1)
    lp_snap = lp[idx_snap]
    bb = np.asarray(book.bid_px[:, 0], dtype=np.float64)
    ba = np.asarray(book.ask_px[:, 0], dtype=np.float64)
    spread = np.where((bb >= 0) & (ba >= 0), ba - bb, np.nan)
    depth = np.asarray(book.bid_sz, dtype=np.float64).sum(axis=1) + np.asarray(
        book.ask_sz, dtype=np.float64
    ).sum(axis=1)
    lam = float(0.5 ** (step_snap / (cfg.fb_rv_long_halflife_days * _SESSION)))
    sp_norm = _ewma(spread, lam)
    dp_norm = np.maximum(_ewma(depth, lam), 1e-12)

    # 情報性の分類 (★この分割が回復率の解釈を決める):
    #   catch-up (|d| 縮小) = κ ハーディングの追いつきカスケード — ミッドが
    #     効率価格 p* **へ**動いた事件。恒久で、回復**しない**のが正しい。
    #   dislocation (|d| 拡大) = 無情報スイープ — p* から**離れた**事件。
    #     κ が引き戻す = フラッシュ・クラッシュ型。回復ゲートはこちらに課す。
    ps = np.asarray(result.price.log_p_star)
    idx_ps = np.minimum(
        (np.asarray(book.t) / float(result.price.t[1] - result.price.t[0])).astype(np.int64),
        ps.size - 1,
    )
    d_snap = ps[idx_ps] - lp_snap

    k30 = max(1, int(round(1800.0 / step_snap)))
    k1d = max(1, int(round(_SESSION / step_snap)))
    rows = []
    for a, b in det["episodes"]:
        seg = lp_snap[a: b + 1]
        start = lp_snap[max(a - 1, 0)]
        ext_i = int(np.argmax(np.abs(seg - start)))
        extreme = float(seg[ext_i])
        move = extreme - float(start)
        rec30 = None
        rec1d = None
        if abs(move) > 1e-12:
            i30 = min(b + k30, lp_snap.size - 1)
            rec30 = float((lp_snap[i30] - extreme) / (start - extreme))
            i1d = min(b + k1d, lp_snap.size - 1)
            rec1d = float((lp_snap[i1d] - extreme) / (start - extreme))
        sp_ratio = float(np.nanmax(spread[a: b + 1] / sp_norm[a: b + 1]))
        dp_ratio = float(np.nanmin(depth[a: b + 1] / dp_norm[a: b + 1]))
        d0 = abs(float(d_snap[max(a - 1, 0)]))
        d1 = abs(float(d_snap[b]))
        rows.append({
            "duration_min": num((b - a + 1) * step_snap / 60.0),
            "move": num(move),
            "down": bool(move < 0),
            "dislocation": bool(d1 > d0),
            "d_change": num(d1 - d0),
            "max_spread_ratio": num(sp_ratio),
            "min_depth_ratio": num(dp_ratio),
            "recovery_30min": num(rec30),
            "recovery_1day": num(rec1d),
        })

    def _med(vals):
        vv = [v for v in vals if v is not None]
        return num(float(np.median(vv))) if vv else None

    disl = [r for r in rows if r["dislocation"]]
    catch = [r for r in rows if not r["dislocation"]]
    rec_all = _med([r["recovery_30min"] for r in rows])
    return ok(
        rec_all,
        n=len(rows),
        n_dislocation=len(disl),
        n_catchup=len(catch),
        duration_min_median=_med([r["duration_min"] for r in rows]),
        max_spread_ratio_median=_med([r["max_spread_ratio"] for r in rows]),
        min_depth_ratio_median=_med([r["min_depth_ratio"] for r in rows]),
        recovery_30min_median=rec_all,
        recovery_1day_median=_med([r["recovery_1day"] for r in rows]),
        recovery_30min_dislocation=_med([r["recovery_30min"] for r in disl]),
        recovery_1day_dislocation=_med([r["recovery_1day"] for r in disl]),
        recovery_1day_catchup=_med([r["recovery_1day"] for r in catch]),
        down_fraction=num(float(np.mean([r["down"] for r in rows]))),
        episodes=rows[:200],
    )


def divergence_monitor(
    result, cfg, threshold: float = 10.0, duration_days: float = 5.0,
    result_off=None,
) -> dict[str, Any]:
    """発散検出 (§10 no_divergence)。

    ★単独ランの「RV > 閾値×中央値の持続」は **L2 のボラエポック (MSM 高状態) で
    偽陽性を出す** (S11a 実測: b_δ=1.5 で 3 件 — フィードバック無しでも起きる水準)。
    `result_off` (同一シードのフィードバック off 対) を与えると判定は
    **log(RV_on/RV_off) の持続** に切り替わり、L2 起因が厳密に相殺される —
    残るのはループ自身の寄与だけ。ゲートはペア版で判定すること。
    """
    lrv = _log_rv_series(result.observation, cfg, _SESSION)
    if lrv.size < 30:
        return na(f"日数が足りません (n={lrv.size})")
    if result_off is not None:
        loff = _log_rv_series(result_off.observation, cfg, _SESSION)
        n = min(lrv.size, loff.size)
        daily = lrv[:n] - loff[:n]
        # ★日次のペア差は whale の**出方の不一致**で ±3〜5 揺れる (パスが
        # 脱相関した後、同じ鯨が片側にしか現れない日がある — S11e 実測)。
        # 真の発散 (g ≥ 1) の署名は**持続的な**天井増幅なので、30 日移動平均が
        # log(threshold) を超えることを判定に使う (鯨タイミング差は多週で相殺)。
        w = 30
        if daily.size < w + 5:
            return na(f"日数が足りません (n={daily.size})")
        csum = np.concatenate(([0.0], np.cumsum(daily)))
        series = (csum[w:] - csum[:-w]) / w
        basis = "paired-smoothed (30 日移動平均の log RV_on − log RV_off)"
        ref = 0.0
        thr = np.log(threshold)
    else:
        series = lrv
        basis = "single (L2 エポックの偽陽性あり — 記録用)"
        ref = float(np.median(lrv))
        thr = np.log(threshold)
    above = series > ref + thr
    n_div = 0
    run = 0
    for a in above:
        run = run + 1 if a else 0
        if run == int(duration_days):
            n_div += 1
    return ok(
        num(n_div),
        n_divergences=int(n_div),
        basis=basis,
        threshold=num(threshold),
        duration_days=num(duration_days),
        max_excess=num(float(series.max() - ref)),
    )


def nt_distribution(result, cfg) -> dict[str, Any]:
    """n_t の実現分布 (§8.3)。u 系列から決定論的に再構成 + エンジン実測カウンタ。"""
    fb = (result.meta.get("l3") or {}).get("feedback") or {}
    meta = result.events.meta if isinstance(result.events.meta, dict) else {}
    u = np.asarray(meta.get("fb_u_grid", np.empty(0)), dtype=np.float64)
    rows: dict[str, Any] = {
        "nt_mean_engine": num(fb.get("nt_mean")),
        "nt_sd_engine": num(fb.get("nt_sd")),
        "nt_max_engine": num(fb.get("nt_max")),
        "u_mean_engine": num(fb.get("u_mean")),
        "u_sd_engine": num(fb.get("u_sd")),
    }
    if u.size and float(cfg.fb_b_n) > 0.0:
        th = np.tanh(u / float(cfg.fb_u_scale))
        sig = 1.0 / (1.0 + np.exp(-float(cfg.fb_b_n) * th))
        nt = float(cfg.fb_n_min) + (float(cfg.fb_n_max) - float(cfg.fb_n_min)) * sig
        rows.update({
            "nt_p05": num(float(np.quantile(nt, 0.05))),
            "nt_p50": num(float(np.quantile(nt, 0.50))),
            "nt_p95": num(float(np.quantile(nt, 0.95))),
        })
    return ok(rows.get("nt_max_engine"), **rows)
