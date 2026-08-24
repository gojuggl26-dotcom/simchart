# -*- coding: utf-8 -*-
"""S10c: c_vol スイープ — L2 緩慢ボラ → L1 活動度リンクの較正 (指示書 §7)。

c_vol ∈ {0 (対照), 0.3, 0.5, 0.8} × 5 シード × 500 日 (κ=0.2, σ̄=0.2217 固定)。

測るもの:
- corr(日次 RV(p_obs), 日次出来高) — 主計器は log-log Pearson
  (日次 RV は裾が重く (Hill≈3.4)、レベル相関は少数の鯨日に支配される —
  S8/S9 で繰り返し確認した計器ノイズと同型)。レベル相関も記録する。
- corr(日次 RV, 日次平均スプレッド) (§7.3 の打ち消し検査、> 0.3)
- 日内スプレッド曲線 (§7.4 — 記録。φ_λ の調整は必要になった場合のみ)
- 伝達率 T(h) — σ̄=0.2217 (S10b、c_vol=0 で較正) が c_vol 有効後も
  T_daily ∈ 1.00±0.05 に留まるかの検証 (Z→追跡速度の相互作用チェック)
- z_mean / cap_hit_rate / 打ち切り回数 (エンジン健全性)

実行: uv run python scripts/s10c_cvol_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simchart import Config, run
from simchart.validation.base import jsonable

S = 23400.0
SEEDS = (42, 43, 44, 45, 46)
KAPPA = 0.2
SIGMA_BAR = 0.2217
N_DAYS = 500
CVOLS = tuple(
    float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else
                       ("0.0", "0.3", "0.5", "0.8"))
)
N_INTRADAY_BINS = 26  # 15 分ビン


def measure_seed(cfg, r) -> dict:
    from simchart.types import EventType

    obs = r.observation
    step = float(obs.step_seconds)
    spd = int(round(S / step))
    burn_d = int(cfg.book_burn_in_days)
    n_days = int(cfg.n_days)

    # --- 日次 RV (p_obs, 1 分リターン) ---
    lp = obs.log_price
    stride = int(round(60.0 / step))
    r1m = np.diff(lp[::stride])
    day_of_r = (np.arange(r1m.size) * stride + stride) // spd
    rv = np.bincount(
        day_of_r.astype(np.int64), weights=r1m**2, minlength=n_days
    )[:n_days]

    # --- 日次出来高 (TRADE 行の size 合計) と日次イベント数 ---
    t_ev = np.asarray(r.events.t)
    etype = np.asarray(r.events.event_type)
    size = np.asarray(r.events.size)
    day_ev = (t_ev / S).astype(np.int64)
    is_tr = etype == int(EventType.TRADE)
    vol = np.bincount(day_ev[is_tr], weights=size[is_tr], minlength=n_days)[:n_days]
    n_ev = np.bincount(day_ev, minlength=n_days)[:n_days].astype(np.float64)

    # --- スプレッド (イベント時点、日次平均と日内曲線) ---
    bb = np.asarray(r.events.meta["best_bid_tick"], dtype=np.float64)
    ba = np.asarray(r.events.meta["best_ask_tick"], dtype=np.float64)
    ok_sp = (bb >= 0) & (ba >= 0)
    sp = (ba - bb)[ok_sp]
    d_sp = day_ev[ok_sp]
    sp_day = np.bincount(d_sp, weights=sp, minlength=n_days)[:n_days]
    sp_cnt = np.bincount(d_sp, minlength=n_days)[:n_days]
    sp_daily = np.where(sp_cnt > 0, sp_day / np.maximum(sp_cnt, 1), np.nan)
    u_bin = ((t_ev[ok_sp] % S) / S * N_INTRADAY_BINS).astype(np.int64)
    u_bin = np.clip(u_bin, 0, N_INTRADAY_BINS - 1)
    keep_b = d_sp >= burn_d
    sp_curve = np.bincount(
        u_bin[keep_b], weights=sp[keep_b], minlength=N_INTRADAY_BINS
    ) / np.maximum(np.bincount(u_bin[keep_b], minlength=N_INTRADAY_BINS), 1)

    # --- 相関 (バーンイン後の日次系列) ---
    k = slice(burn_d, n_days)
    rv_k, vol_k, nev_k, spd_k = rv[k], vol[k], n_ev[k], sp_daily[k]
    good = (rv_k > 0) & (vol_k > 0) & np.isfinite(spd_k)

    def _corr(a, b):
        if good.sum() < 30:
            return None
        return float(np.corrcoef(a[good], b[good])[0, 1])

    out = {
        "corr_rv_volume_log": _corr(np.log(rv_k), np.log(vol_k)),
        "corr_rv_volume_level": _corr(rv_k, vol_k),
        "corr_rv_nevents_log": _corr(np.log(rv_k), np.log(nev_k)),
        "corr_rv_spread": _corr(np.log(rv_k), spd_k),
        "spread_mean_ticks": float(np.nanmean(spd_k)),
        "spread_intraday_curve": [float(x) for x in sp_curve],
        "volume_mean_daily": float(vol_k.mean()),
        "n_events_mean_daily": float(nev_k.mean()),
    }

    # --- 伝達率 T(h) (σ̄ 再検証) ---
    burn = burn_d * spd
    lp_b = lp[burn:]
    ps_b = r.price.log_p_star[burn:]
    for label, st in (("60", stride), ("600", int(round(600 / step))),
                      ("1d", spd), ("5d", 5 * spd)):
        do = np.diff(lp_b[::st])
        dp = np.diff(ps_b[::st])
        out[f"T_{label}"] = float(do.var() / dp.var())

    # --- エンジン診断 ---
    hk = r.meta["l3"].get("hawkes", {})
    out["cap_hit_rate"] = hk.get("cap_hit_rate")
    out["acceptance_rate"] = hk.get("acceptance_rate")
    cv = r.meta["l3"].get("cvol")
    if cv:
        out["z_mean"] = cv["z_mean"]
        out["z_max"] = cv["z_max"]
        out["v_sd"] = cv["v_sd"]
        out["truncations"] = cv["truncations"]
    return out


def main() -> int:
    out_dir = ROOT / "results" / "S10c"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cvol in CVOLS:
        per_seed = []
        t0 = time.perf_counter()
        for seed in SEEDS:
            cfg = Config.load(ROOT / "configs" / "s9.yaml").replace(
                stage="S10", seed=seed, n_days=N_DAYS, kappa=KAPPA,
                sigma_bar=SIGMA_BAR,
                **({"c_vol": cvol} if cvol > 0 else {}),
            )
            per_seed.append(measure_seed(cfg, run(cfg)))
        med = {
            k: float(np.median([s[k] for s in per_seed]))
            for k in ("corr_rv_volume_log", "corr_rv_volume_level",
                      "corr_rv_nevents_log", "corr_rv_spread",
                      "spread_mean_ticks", "T_60", "T_600", "T_1d", "T_5d")
            if all(s.get(k) is not None for s in per_seed)
        }
        row = {"c_vol": cvol, "median": med, "seeds": per_seed}
        rows.append(row)
        zs = [s.get("z_mean") for s in per_seed if s.get("z_mean") is not None]
        print(
            f"c_vol={cvol}: rv-vol(log)={med.get('corr_rv_volume_log', float('nan')):+.3f} "
            f"(level {med.get('corr_rv_volume_level', float('nan')):+.3f}) "
            f"rv-spread={med.get('corr_rv_spread', float('nan')):+.3f} "
            f"T_1d={med.get('T_1d', float('nan')):.3f} "
            f"spread={med.get('spread_mean_ticks', float('nan')):.2f}t "
            f"z_mean={np.median(zs) if zs else 1.0:.3f} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    suffix = "" if len(sys.argv) <= 1 else "_refine"
    (out_dir / f"cvol_sweep{suffix}.json").write_text(
        json.dumps(jsonable({
            "kappa": KAPPA, "sigma_bar": SIGMA_BAR, "n_days": N_DAYS,
            "seeds": SEEDS, "target_rv_volume": [0.5, 0.7],
            "rows": rows,
        }), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"saved {out_dir / ('cvol_sweep' + suffix + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
