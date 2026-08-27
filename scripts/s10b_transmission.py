# -*- coding: utf-8 -*-
"""S10b: 伝達率 T(h) の測定と σ̄ の再較正 (設計要件 — 方向は測ってから決める)。

T(h) = Var[Δ_h log p_obs] / Var[Δ_h log p*] を h = 1分/10分/1日/5日 で測り、
σ̄_new = σ̄_old / √T_daily を T_daily = 1.00 ± 0.05 まで反復 (通常 2 回)。
あわせて §6 のジャンプ平滑化 (JV share・Hill α) を各反復で記録する。
結果は results/S10b/ へ。

実行: uv run python scripts/s10b_transmission.py
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
from simchart.validation.scaling import realized_variance
from simchart.validation.tails import bns_jump_test, hill_estimator

S = 23400.0
SEEDS = (42, 43, 44, 45, 46)
KAPPA = 0.2
N_DAYS = 500


def measure(sigma_bar: float) -> dict:
    t_rows = {"60": [], "600": [], "1d": [], "5d": []}
    jv, hill, kurt_1m = [], [], []
    for seed in SEEDS:
        cfg = Config.load(ROOT / "configs" / "s9.yaml").replace(
            stage="S10", seed=seed, n_days=N_DAYS, kappa=KAPPA,
            sigma_bar=sigma_bar,
        )
        r = run(cfg)
        obs = r.observation
        step = obs.step_seconds
        burn = int(cfg.book_burn_in_days * S / step)
        lp = obs.log_price[burn:]
        ps = r.price.log_p_star[burn:]
        spd = int(round(S / step))
        for label, stride in (("60", int(round(60 / step))),
                              ("600", int(round(600 / step))),
                              ("1d", spd), ("5d", 5 * spd)):
            do = np.diff(lp[::stride])
            dp = np.diff(ps[::stride])
            t_rows[label].append(float(do.var() / dp.var()))
        r_daily = np.diff(lp[::spd])
        step_r = np.diff(lp)
        jv.append(bns_jump_test(step_r, spd).get("jv_share"))
        hill.append(hill_estimator(r_daily, 0.05, "both").get("alpha"))
        r1m = np.diff(lp[:: int(round(60 / step))])
        kurt_1m.append(float(3.0 + (np.mean(r1m**4) / np.mean(r1m**2) ** 2 - 3.0)))
    out = {
        "sigma_bar": sigma_bar,
        "T": {k: float(np.median(v)) for k, v in t_rows.items()},
        "T_values": t_rows,
        "jv_share_median": float(np.median([x for x in jv if x is not None])),
        "hill_median": float(np.median([x for x in hill if x is not None])),
        "kurt_1min_median": float(np.median(kurt_1m)),
    }
    return out


def main() -> int:
    out_dir = ROOT / "results" / "S10b"
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = 0.20
    iters = []
    for it in range(4):
        t0 = time.perf_counter()
        m = measure(sigma)
        m["iteration"] = it
        iters.append(m)
        td = m["T"]["1d"]
        print(f"iter {it}: sigma_bar={sigma:.4f}  "
              f"T(1m)={m['T']['60']:.3f} T(10m)={m['T']['600']:.3f} "
              f"T(1d)={td:.3f} T(5d)={m['T']['5d']:.3f}  "
              f"JV={m['jv_share_median']:.3f} Hill={m['hill_median']:.2f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        if abs(td - 1.0) <= 0.05:
            print(f"収束: T_daily = {td:.3f} ∈ 1.00 ± 0.05")
            break
        sigma = sigma / np.sqrt(td)
        print(f"  -> sigma_bar を {sigma:.4f} へ (σ̄/√T_daily)")
    (out_dir / "transmission.json").write_text(
        json.dumps(jsonable({"kappa": KAPPA, "n_days": N_DAYS, "seeds": SEEDS,
                             "iterations": iters,
                             "sigma_bar_final": sigma}),
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'transmission.json'}  sigma_final={sigma:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
