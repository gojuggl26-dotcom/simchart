# -*- coding: utf-8 -*-
"""S12c: b_χ 較正 — 危機窓のシード横断再現性 (§8.2) + g 再測定 (§6.2)。

n_max = 0.95 (S12 拡張) で b_χ を振り、
- 窓 (5 日) あたり危機件数のシード横断相関 → 目標 0.3〜0.6
- g (同一シード on/off 対 — off はフィードバック+χ₃ を外し χ₁ は両脚に残す
  = ループだけを測る)
- hill_all / hill_ex (裾ガード)・発散 (30 日平滑ペア)・nt_max
を測る。1000 日 × 6 シード。

実行: uv run python scripts/s12c_window_sweep.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simchart import Config, run
from simchart.validation.base import jsonable
from simchart.validation.feedback import (
    crisis_detect,
    crisis_window_counts,
    crisis_window_reproducibility,
    divergence_monitor,
    loop_gain_estimate,
)
from simchart.validation.tails import hill_estimator

S = 23400.0
SEEDS = (42, 43, 44, 45, 47, 48)
N_DAYS = 1000
# 格子は (b_chi, n_max) 対。引数 1 = "1.0:0.95,2.0:0.93,..."、引数 2 = 出力名、
# 引数 3 = 検出器基準の半減期 [日] (既定 2.0 = S11 挙動)。
BCHIS = ((0.5, 0.95), (1.0, 0.95), (1.5, 0.95), (2.0, 0.95))
OUT_NAME = "window_sweep"
NORM_HL = 2.0
if len(sys.argv) > 2:
    BCHIS = tuple(
        tuple(float(x) for x in p.split(":")) for p in sys.argv[1].split(",")
    )
    OUT_NAME = sys.argv[2]
    if len(sys.argv) > 3:
        NORM_HL = float(sys.argv[3])
DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}


def base_cfg(seed: int, b_chi: float, n_max: float = 0.95) -> Config:
    return Config.load(ROOT / "configs" / "s11.yaml").replace(
        stage="S12", seed=seed, n_days=N_DAYS,
        fb_n_max=n_max, crisis_norm_halflife_days=NORM_HL,
        enable_chaos_lambda=True, enable_chaos_branching=True, chi3_b=b_chi,
    )


def off_cfg(cfg: Config) -> Config:
    return cfg.replace(
        enable_feedback=False, enable_chaos_branching=False,
        **{p: DEFAULTS[p] for p in Config._S11_FB_PARAMS},
        **{p: DEFAULTS[p] for p in Config._S12_CHI3_PARAMS},
    )


def hill_pair(r, cfg, det):
    obs = r.observation
    spd = int(round(S / obs.step_seconds))
    r_d = np.diff(np.asarray(obs.log_price)[::spd])
    h_all = hill_estimator(r_d, 0.05, "both").get("alpha")
    mask = np.ones(r_d.size, dtype=bool)
    step_snap = det.get("step_sec") or 60.0
    for a, b in det.get("episodes") or []:
        d0 = int(a * step_snap / S)
        d1 = int(b * step_snap / S)
        mask[max(d0 - 1, 0): min(d1 + 2, r_d.size)] = False
    return h_all, hill_estimator(r_d[mask], 0.05, "both").get("alpha")


def main() -> int:
    out_dir = ROOT / "results" / "S12c"
    out_dir.mkdir(parents=True, exist_ok=True)
    offs = {}
    for seed in SEEDS:
        try:
            offs[seed] = run(off_cfg(base_cfg(seed, 1.0)))
        except RuntimeError as e:
            print(f"seed {seed}: off SKIP ({str(e)[:50]})", flush=True)
    rows = []
    for b_chi, n_max in BCHIS:
        per, vecs = [], []
        t0 = time.perf_counter()
        for seed in SEEDS:
            if seed not in offs:
                continue
            cfg = base_cfg(seed, b_chi, n_max)
            try:
                r_on = run(cfg)
            except RuntimeError as e:
                per.append({"seed": seed, "skip": str(e)[:60]})
                continue
            det = crisis_detect(r_on, cfg)
            vecs.append(crisis_window_counts(r_on, cfg, 5.0, detection=det))
            g = loop_gain_estimate(r_on, offs[seed], cfg)
            div = divergence_monitor(r_on, cfg, result_off=offs[seed])
            h_all, h_ex = hill_pair(r_on, cfg, det)
            fb = r_on.meta["l3"]["feedback"]
            per.append({
                "seed": seed, "g_30min": g.get("g_30min"),
                "n_div": div.get("n_divergences"),
                "crises_per_year": det.get("per_year"),
                "hill_all": h_all, "hill_ex": h_ex,
                "nt_max": fb["nt_max"], "nt_mean": fb["nt_mean"],
            })
        okr = [p for p in per if "skip" not in p]
        rep = crisis_window_reproducibility(vecs)

        def med(k):
            vals = [p[k] for p in okr if p.get(k) is not None]
            return float(np.median(vals)) if vals else float("nan")

        rows.append({"b_chi": b_chi, "n_max": n_max, "per_seed": per,
                     "window_reproducibility": rep,
                     "median": {k: med(k) for k in
                                ("g_30min", "crises_per_year", "hill_all",
                                 "hill_ex", "nt_max", "nt_mean")}})
        print(
            f"b_chi={b_chi} n_max={n_max}: win_corr={rep.get('median'):+.3f} "
            f"[{rep.get('q1'):+.2f},{rep.get('q3'):+.2f}] "
            f"g30={med('g_30min'):+.3f} crises/y={med('crises_per_year'):.1f} "
            f"hill={med('hill_all'):.2f}/{med('hill_ex'):.2f} "
            f"nt_max={med('nt_max'):.3f} "
            f"div={sum(p.get('n_div', 0) or 0 for p in okr)} "
            f"({time.perf_counter()-t0:.0f}s)",
            flush=True,
        )
    (out_dir / f"{OUT_NAME}.json").write_text(
        json.dumps(jsonable({"n_days": N_DAYS, "seeds": SEEDS,
                             "norm_halflife_days": NORM_HL,
                             "rows": rows}), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"saved {out_dir / (OUT_NAME + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
