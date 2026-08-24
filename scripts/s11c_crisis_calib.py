# -*- coding: utf-8 -*-
"""S11c: 危機検出器の閾値較正 (§6.1 — 閾値は config、帯は実測で確定)。

off 基線 (フィードバック無し) の検出率を ~0 に落とす (k, m) を探す —
現行 (k=5, m=3) は通常のマイクロ構造 + L2 ジャンプで ~120/年 発火しており、
「危機」ではなく通常変動を数えている (回復率 ≈ 0 が傍証 — ジャンプは戻らない)。
シミュレーションは 3 シード × (off, on@作業点) の 6 本だけ回し、
検出は cfg.replace で閾値を振って再計算する (検出は純後処理)。

実行: uv run python scripts/s11c_crisis_calib.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simchart import Config, run
from simchart.validation.base import jsonable
from simchart.validation.feedback import crisis_anatomy, crisis_detect

SEEDS = (42, 43, 44)
N_DAYS = 500
B = (1.0, 1.0, 2.0)


def main() -> int:
    out_dir = ROOT / "results" / "S11c"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in SEEDS:
        base = Config.load(ROOT / "configs" / "s10.yaml").replace(
            stage="S11", seed=seed, n_days=N_DAYS,
        )
        runs[seed] = {
            "off": run(base),
            "on": run(base.replace(
                enable_feedback=True,
                fb_b_delta=B[0], fb_b_place=B[1], fb_b_n=B[2],
            )),
            "cfg": base,
        }
        print(f"seed {seed}: sims done", flush=True)

    rows = []
    for k in (5.0, 6.0, 7.0, 8.0):
        for m in (3.0, 4.0, 5.0):
            offr, onr, rec, dur, spr, dpr = [], [], [], [], [], []
            for seed in SEEDS:
                cfg_t = runs[seed]["cfg"].replace(
                    enable_feedback=True, fb_b_delta=B[0], fb_b_place=B[1],
                    fb_b_n=B[2], crisis_k_sigma=k, crisis_spread_mult=m,
                )
                d_off = crisis_detect(runs[seed]["off"], cfg_t)
                d_on = crisis_detect(runs[seed]["on"], cfg_t)
                offr.append(d_off.get("per_year") or 0.0)
                onr.append(d_on.get("per_year") or 0.0)
                if d_on.get("episodes"):
                    a = crisis_anatomy(runs[seed]["on"], cfg_t, detection=d_on)
                    if a.get("status") == "ok":
                        rec.append(a.get("recovery_30min_median"))
                        dur.append(a.get("duration_min_median"))
                        spr.append(a.get("max_spread_ratio_median"))
                        dpr.append(a.get("min_depth_ratio_median"))

            def med(v):
                vv = [x for x in v if x is not None]
                return float(np.median(vv)) if vv else None

            row = {"k": k, "m": m, "off_per_year": med(offr), "on_per_year": med(onr),
                   "recovery_30min": med(rec), "duration_min": med(dur),
                   "spread_ratio": med(spr), "depth_ratio": med(dpr)}
            rows.append(row)
            print(
                f"k={k} m={m}: off={row['off_per_year']:.1f}/y on={row['on_per_year']:.1f}/y "
                f"rec30={row['recovery_30min'] if row['recovery_30min'] is not None else float('nan'):+.2f} "
                f"dur={row['duration_min'] or 0:.0f}m sp_ratio={row['spread_ratio'] or 0:.1f} "
                f"dp_ratio={row['depth_ratio'] or 1:.2f}",
                flush=True,
            )
    (out_dir / "crisis_calib.json").write_text(
        json.dumps(jsonable({"b": B, "n_days": N_DAYS, "seeds": SEEDS, "rows": rows}),
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'crisis_calib.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
