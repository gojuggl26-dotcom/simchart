# -*- coding: utf-8 -*-
"""S11: 危機頻度の長期標本 (設計要件 — 50,000 日相当)。

危機は稀なので頻度推定には長期標本が必須。5000 日 × 10 は実行不能 (S10d) の
ため、1000 日 × 50 シードで同一の総標本を取る。フルスイートは回さず
危機検出・解剖だけを測る軽量ラン。窓逸脱シードは記録の上スキップ。

実行: uv run python scripts/s11_crisis_longsample.py
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
from simchart.validation.feedback import crisis_anatomy, crisis_detect

SEEDS = tuple(range(42, 42 + 62))  # 逸脱 ~20% 見込みで 62 → 完走 ~50
TARGET_DAYS = 50_000
N_DAYS = 1000


def main() -> int:
    out_dir = ROOT / "results" / "S11"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    days_done = 0
    t0 = time.perf_counter()
    for seed in SEEDS:
        if days_done >= TARGET_DAYS:
            break
        cfg = Config.load(ROOT / "configs" / "s11.yaml").replace(
            seed=seed, n_days=N_DAYS
        )
        try:
            r = run(cfg)
        except RuntimeError as e:
            skipped.append({"seed": seed, "err": str(e)[:60]})
            continue
        det = crisis_detect(r, cfg)
        ana = crisis_anatomy(r, cfg, detection=det)
        rows.append({
            "seed": seed,
            "per_year": det.get("per_year"),
            "n_episodes": det.get("n_episodes"),
            "n_dislocation": ana.get("n_dislocation"),
            "n_catchup": ana.get("n_catchup"),
            "duration_min_median": ana.get("duration_min_median"),
            "max_spread_ratio_median": ana.get("max_spread_ratio_median"),
            "min_depth_ratio_median": ana.get("min_depth_ratio_median"),
            "recovery_30min_dislocation": ana.get("recovery_30min_dislocation"),
            "recovery_1day_dislocation": ana.get("recovery_1day_dislocation"),
            "recovery_1day_catchup": ana.get("recovery_1day_catchup"),
        })
        days_done += N_DAYS - int(cfg.book_burn_in_days)
        if len(rows) % 10 == 0:
            print(f"{len(rows)} 完走 / {days_done} 日 ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    def med(k):
        vals = [x[k] for x in rows if x.get(k) is not None]
        return float(np.median(vals)) if vals else None

    summary = {
        "n_runs": len(rows), "n_skipped": len(skipped),
        "total_days": days_done,
        "per_year_median": med("per_year"),
        "per_year_q1q3": [
            float(np.percentile([x["per_year"] for x in rows], q)) for q in (25, 75)
        ] if rows else None,
        "dislocation_fraction": (
            float(sum(x["n_dislocation"] or 0 for x in rows))
            / max(sum(x["n_episodes"] or 0 for x in rows), 1)
        ),
        "duration_min_median": med("duration_min_median"),
        "max_spread_ratio_median": med("max_spread_ratio_median"),
        "min_depth_ratio_median": med("min_depth_ratio_median"),
        "recovery_30min_dislocation_median": med("recovery_30min_dislocation"),
        "recovery_1day_dislocation_median": med("recovery_1day_dislocation"),
        "recovery_1day_catchup_median": med("recovery_1day_catchup"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (out_dir / "crisis_longsample.json").write_text(
        json.dumps(jsonable({"summary": summary, "rows": rows, "skipped": skipped}),
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'crisis_longsample.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
