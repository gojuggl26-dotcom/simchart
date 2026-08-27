# -*- coding: utf-8 -*-
"""S11a: L3 チャネル (b_δ, b_Δ) 単独のループゲイン測定 (設計要件/§5)。

同一シード・同一 L2 経路の on/off 対で g = 1 − √(Var_off/Var_on) を推定。
日次 (設計要件の式) と 30 分帯域 (ループが実際に増幅する側) の両方を出す。
発散検出・スプレッド・u 統計も記録。500 日 × 3 シード。

実行: uv run python scripts/s11a_gain_sweep.py
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
from simchart.validation.feedback import (
    crisis_anatomy,
    crisis_detect,
    divergence_monitor,
    loop_gain_estimate,
)

SEEDS = (42, 43, 44)
N_DAYS = 500
# 格子は (b_δ, b_Δ, b_n)。引数で上書き可: "0.6:0.4:0,0:0:2.0,..." と
# 出力サブディレクトリ名 (S11a/S11b/S11c)。
GRID = (
    (0.3, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0),
    (0.0, 0.3, 0.0), (0.0, 0.6, 0.0), (0.0, 1.0, 0.0),
    (0.6, 0.4, 0.0), (1.0, 0.6, 0.0),
)
OUT_SUB = "S11a"
if len(sys.argv) > 2:
    GRID = tuple(
        tuple(float(x) for x in p.split(":")) for p in sys.argv[1].split(",")
    )
    OUT_SUB = sys.argv[2]


def base_cfg(seed: int) -> Config:
    return Config.load(ROOT / "configs" / "s10.yaml").replace(
        stage="S11", seed=seed, n_days=N_DAYS,
    )


def main() -> int:
    out_dir = ROOT / "results" / OUT_SUB
    out_dir.mkdir(parents=True, exist_ok=True)
    offs = {}
    off_crisis = {}
    for seed in SEEDS:
        try:
            offs[seed] = run(base_cfg(seed))
        except RuntimeError as e:
            print(f"seed {seed}: off SKIP ({str(e)[:50]})", flush=True)
            continue
        det0 = crisis_detect(offs[seed], base_cfg(seed))
        off_crisis[seed] = det0.get("per_year")
    if off_crisis:
        print(
            "off 基線 (フィードバック無しの検出率): "
            + ", ".join(f"seed{s}={v:.1f}/y" for s, v in off_crisis.items()),
            flush=True,
        )
    rows = []
    for b_d, b_p, b_n in GRID:
        per = []
        t0 = time.perf_counter()
        for seed in SEEDS:
            if seed not in offs:
                continue
            cfg = base_cfg(seed).replace(
                enable_feedback=True, fb_b_delta=b_d, fb_b_place=b_p, fb_b_n=b_n,
            )
            try:
                r_on = run(cfg)
            except RuntimeError as e:
                per.append({"seed": seed, "skip": str(e)[:60]})
                continue
            g = loop_gain_estimate(r_on, offs[seed], cfg)
            div = divergence_monitor(r_on, cfg, result_off=offs[seed])
            det = crisis_detect(r_on, cfg)
            ana = crisis_anatomy(r_on, cfg, detection=det)
            fb = r_on.meta["l3"]["feedback"]
            bk = r_on.book
            bb = np.asarray(bk.bid_px[:, 0])
            ba = np.asarray(bk.ask_px[:, 0])
            sp = (ba - bb)[(bb >= 0) & (ba >= 0)]
            per.append({
                "seed": seed,
                "g_daily": g.get("g_daily"), "g_30min": g.get("g_30min"),
                "n_div": div.get("n_divergences"),
                "crises_per_year": det.get("per_year"),
                "u_mean": fb["u_mean"], "u_sd": fb["u_sd"],
                "nt_mean": fb["nt_mean"], "nt_max": fb["nt_max"],
                "spread_median": float(np.median(sp)),
                "spread_p99": float(np.quantile(sp, 0.99)),
                "recovery_30min": ana.get("recovery_30min_median"),
                "crisis_duration_min": ana.get("duration_min_median"),
                "crisis_spread_ratio": ana.get("max_spread_ratio_median"),
                "crisis_depth_ratio": ana.get("min_depth_ratio_median"),
                "crisis_down_frac": ana.get("down_fraction"),
            })
        ok_rows = [p for p in per if "skip" not in p]

        def med(k):
            vals = [p[k] for p in ok_rows if p.get(k) is not None]
            return float(np.median(vals)) if vals else float("nan")

        rows.append({"b_delta": b_d, "b_place": b_p, "b_n": b_n, "per_seed": per,
                     "median": {k: med(k) for k in
                                ("g_daily", "g_30min", "crises_per_year",
                                 "u_sd", "nt_mean", "nt_max",
                                 "spread_median", "spread_p99",
                                 "recovery_30min", "crisis_duration_min",
                                 "crisis_spread_ratio", "crisis_depth_ratio",
                                 "crisis_down_frac")}})
        print(
            f"b_δ={b_d} b_Δ={b_p} b_n={b_n}: g_30m={med('g_30min'):+.3f} "
            f"g_daily={med('g_daily'):+.3f} crises/y={med('crises_per_year'):.1f} "
            f"rec30={med('recovery_30min'):+.2f} dur={med('crisis_duration_min'):.0f}m "
            f"nt_max={med('nt_max'):.3f} "
            f"sp_p99={med('spread_p99') * 100:.0f}t "
            f"div={sum(p.get('n_div', 0) or 0 for p in ok_rows)} "
            f"({time.perf_counter()-t0:.0f}s)",
            flush=True,
        )
    (out_dir / "gain_sweep.json").write_text(
        json.dumps(jsonable({"n_days": N_DAYS, "seeds": SEEDS, "rows": rows}),
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'gain_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
