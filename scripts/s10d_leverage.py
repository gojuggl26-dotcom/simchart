# -*- coding: utf-8 -*-
"""S10d: レバレッジ実測 + T_daily 再ピン + 5000 日実行可能性。

結合後の観測価格 p_obs で corr(r_d, RV_{d+1}) を 1000 日 × 5 シードで測る。
- 帯: 実市場 [−0.28, −0.16] 程度。< −0.40 (過強) の場合のみ ρ_rough を
  弱める (設計要件が明示的に許可する L2 調整。ρ_slow は触らない)。
- τ プロファイル corr(r_d, RV_{d+τ}) (τ=0..10) と p* 側も記録 (経路の同定)。
- T(h) を 1000 日で再測定 — S10c で判明した σ̄=0.2217 の帯端問題
  (500 日推定は ±0.05 揺れる) を高精度で決着させる。
- 5000 日 × 10 の実行可能性: 実測レートからイベントログ所要メモリを外挿し、
  1000 日 1 本の全スイート時間を測って S10e の形 (5000×10 vs 1000×N) を決める。

実行: uv run python scripts/s10d_leverage.py
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
# 1000 日は絶対ティック窓 (±9900) に対し p* の逸脱が ~1.6σ で、シードの
# 1 割前後が窓超過で落ちる (seed 46 実測)。S8 と同じ skip + カバレッジ方式。
SEEDS = tuple(range(42, 54))
KAPPA = 0.2
C_VOL = 0.65
SIGMA_BAR = 0.2217
N_DAYS = 1000
TAUS = tuple(range(0, 11))


def _daily(lp: np.ndarray, step: float, n_days: int):
    spd = int(round(S / step))
    r_d = np.diff(lp[::spd])[:n_days - 1]
    stride = max(1, int(round(60.0 / step)))
    r1m = np.diff(lp[::stride])
    day_of_r = (np.arange(r1m.size) * stride + stride) // spd
    rv = np.bincount(day_of_r.astype(np.int64), weights=r1m**2, minlength=n_days)[:n_days]
    return r_d, rv


def measure_seed(cfg, r) -> dict:
    obs = r.observation
    step = float(obs.step_seconds)
    burn_d = int(cfg.book_burn_in_days)
    n_days = int(cfg.n_days)
    out: dict = {}
    for name, lp in (("obs", obs.log_price), ("latent", r.price.log_p_star)):
        r_d, rv = _daily(np.asarray(lp), step, n_days)
        prof = {}
        for tau in TAUS:
            # 対 (r_d[i] = 日 i のリターン, rv[i+τ] = 日 i+τ の RV)
            i0, i1 = burn_d, min(n_days - 1, n_days - tau)
            a = r_d[i0:i1]
            b = rv[i0 + tau:i1 + tau]
            prof[str(tau)] = float(np.corrcoef(a, b)[0, 1])
        out[f"lev_{name}_tau1"] = prof["1"]
        out[f"lev_{name}_profile"] = prof
    spd = int(round(S / step))
    burn = burn_d * spd
    lp_b = np.asarray(obs.log_price)[burn:]
    ps_b = r.price.log_p_star[burn:]
    for label, st in (("60", int(round(60 / step))), ("600", int(round(600 / step))),
                      ("1d", spd), ("5d", 5 * spd)):
        do = np.diff(lp_b[::st])
        dp = np.diff(ps_b[::st])
        out[f"T_{label}"] = float(do.var() / dp.var())
    out["n_events"] = int(np.asarray(r.events.t).size)
    cv = r.meta["l3"].get("cvol") or {}
    out["z_mean"] = cv.get("z_mean")
    # 窓リスクの定量化: ミッドの p0 からの最大逸脱 [tick] (窓 half=9900)
    px = np.exp(np.asarray(obs.log_price))
    tick = float(cfg.tick_size)
    out["max_abs_dev_ticks"] = float(np.max(np.abs(px - float(cfg.p0))) / tick)
    return out


def main() -> int:
    out_dir = ROOT / "results" / "S10d"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    skipped = []
    t_all = time.perf_counter()
    suite_time_1000d = None
    for seed in SEEDS:
        cfg = Config.load(ROOT / "configs" / "s9.yaml").replace(
            stage="S10", seed=seed, n_days=N_DAYS, kappa=KAPPA,
            sigma_bar=SIGMA_BAR, c_vol=C_VOL,
        )
        t0 = time.perf_counter()
        try:
            r = run(cfg)
        except RuntimeError as e:
            skipped.append({"seed": seed, "reason": str(e)[:120]})
            print(f"seed {seed}: SKIP ({str(e)[:60]}...)", flush=True)
            continue
        sim_sec = time.perf_counter() - t0
        row = measure_seed(cfg, r)
        row["seed"] = seed
        row["sim_sec"] = sim_sec
        if suite_time_1000d is None:
            from simchart.validation import run_all

            t1 = time.perf_counter()
            _m = run_all(r, cfg)
            suite_time_1000d = time.perf_counter() - t1
            del _m
        per_seed.append(row)
        print(
            f"seed {seed}: lev_obs(1)={row['lev_obs_tau1']:+.3f} "
            f"lev_latent(1)={row['lev_latent_tau1']:+.3f} "
            f"T_1d={row['T_1d']:.3f} T_5d={row['T_5d']:.3f} "
            f"dev={row['max_abs_dev_ticks']:.0f}t "
            f"ev={row['n_events']/1e6:.1f}M sim={sim_sec:.0f}s",
            flush=True,
        )
    med = {
        k: float(np.median([s[k] for s in per_seed]))
        for k in ("lev_obs_tau1", "lev_latent_tau1", "T_60", "T_600", "T_1d", "T_5d")
    }
    ev_per_day = float(np.median([s["n_events"] for s in per_seed])) / N_DAYS
    n_ev_fields = 14
    est_5000d_gb = 5000 * ev_per_day * 1.6 * 1.2 * n_ev_fields * 8 / 1e9
    feas = {
        "events_per_day": ev_per_day,
        "event_log_5000d_est_gb": est_5000d_gb,
        "suite_time_1000d_sec": suite_time_1000d,
        "sim_time_1000d_sec_median": float(np.median([s["sim_sec"] for s in per_seed])),
    }
    print(
        f"\nmedian: lev_obs(1)={med['lev_obs_tau1']:+.3f} "
        f"lev_latent(1)={med['lev_latent_tau1']:+.3f}\n"
        f"T: 1m={med['T_60']:.3f} 10m={med['T_600']:.3f} "
        f"1d={med['T_1d']:.3f} 5d={med['T_5d']:.3f}\n"
        f"feasibility: {ev_per_day:.0f} ev/day -> 5000d log ≈ {est_5000d_gb:.1f} GB, "
        f"suite(1000d) = {suite_time_1000d:.0f}s"
    )
    (out_dir / "leverage.json").write_text(
        json.dumps(jsonable({
            "kappa": KAPPA, "c_vol": C_VOL, "sigma_bar": SIGMA_BAR,
            "n_days": N_DAYS, "seeds": SEEDS,
            "median": med, "seeds_detail": per_seed, "skipped": skipped,
            "n_completed": len(per_seed), "feasibility": feas,
            "total_sec": time.perf_counter() - t_all,
        }), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'leverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
