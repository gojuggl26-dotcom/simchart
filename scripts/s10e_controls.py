# -*- coding: utf-8 -*-
"""S10e: 同一ホライズン対照 — ゲート不合格の「ホライズン効果 vs 結合起因」分離。

S8/S9 の帯は 250 日で較正された。whale 感応計器 (γ̂・VR・β̂・sqrt・プール) は
α<2 の物理でホライズン依存なので、S10@1000 日を S8@250 日の帯と比べるのは
gph_d で回避した誤りの再演になる。ここで測るのは:

A. κ=0/c_vol=0 対照 (S9 相当 + σ̄=0.2217、板あり) 1000 日 × 10:
   meta_gamma / vr_tt(1000) / beta / sqrt_slope / pool_rel_diff / obs 歪度 / 5分JV
B. 潜在対照 (板なし、同シード) 1000 日 × 10:
   潜在の日次歪度・5 分 JV・1 秒 JV (obs_skewness / obs_jv_share_5min の
   計器自体のバイアスを分離)

実行: uv run python scripts/s10e_controls.py
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
SEEDS = tuple(range(42, 54))
N_DAYS = 1000


def _obs_stats(r, cfg) -> dict:
    from scipy import stats as sp_stats

    from simchart.validation.tails import bns_jump_test

    obs = r.observation
    r_daily = obs.to_bars(obs.session_seconds).returns()
    spd = int(round(obs.session_seconds / obs.step_seconds))
    out = {"skew_daily": float(sp_stats.skew(r_daily, bias=False))}
    stride5 = max(1, int(round(300.0 / obs.step_seconds)))
    r5 = np.diff(np.asarray(obs.log_price)[::stride5])
    out["jv_share_5min"] = bns_jump_test(r5, spd // stride5).get("jv_share")
    r1 = np.diff(np.asarray(obs.log_price))
    out["jv_share_1s"] = bns_jump_test(r1, spd).get("jv_share")
    return out


def main() -> int:
    out_dir = ROOT / "results" / "S10e"
    out_dir.mkdir(parents=True, exist_ok=True)
    from simchart.cli import _meta_seed_stats

    rows_ctrl, rows_lat, skipped = [], [], []
    for seed in SEEDS:
        base = Config.load(ROOT / "configs" / "s9.yaml").replace(
            stage="S10", seed=seed, n_days=N_DAYS, sigma_bar=0.2217,
        )
        t0 = time.perf_counter()
        try:
            r = run(base)
        except RuntimeError as e:
            skipped.append({"seed": seed, "leg": "ctrl", "err": str(e)[:80]})
            print(f"seed {seed}: ctrl SKIP", flush=True)
        else:
            row = {"seed": seed}
            row.update(_meta_seed_stats(r, base))
            row.update({f"obs_{k}": v for k, v in _obs_stats(r, base).items()})
            rows_ctrl.append(row)
            print(
                f"seed {seed}: ctrl gamma={row['meta_gamma']:.3f} "
                f"vr={row['meta_vr_trade_1000']:.2f} beta={row['meta_beta']:.3f} "
                f"sqrt={row['meta_sqrt_slope']:.3f} pool={row['meta_pool_rel_diff']:.2f} "
                f"skew={row['obs_skew_daily']:+.2f} jv5={row['obs_jv_share_5min']:.3f} "
                f"({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
        rl = run(base.without_book())
        lrow = {"seed": seed}
        lrow.update({f"lat_{k}": v for k, v in _obs_stats(rl, base).items()})
        rows_lat.append(lrow)
        print(
            f"seed {seed}: lat skew={lrow['lat_skew_daily']:+.2f} "
            f"jv5={lrow['lat_jv_share_5min']:.3f} jv1s={lrow['lat_jv_share_1s']:.3f}",
            flush=True,
        )

    def med(rows, k):
        vals = [x[k] for x in rows if x.get(k) is not None]
        return float(np.median(vals)) if vals else None

    summary = {
        "ctrl": {k: med(rows_ctrl, k) for k in (
            "meta_gamma", "meta_c1", "meta_vr_trade_1000", "meta_beta",
            "meta_sqrt_slope", "meta_pool_rel_diff",
            "obs_skew_daily", "obs_jv_share_5min", "obs_jv_share_1s")},
        "latent": {k: med(rows_lat, k) for k in (
            "lat_skew_daily", "lat_jv_share_5min", "lat_jv_share_1s")},
        "n_ctrl": len(rows_ctrl),
    }
    print("\n=== 中央値 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (out_dir / "controls_1000d.json").write_text(
        json.dumps(jsonable({
            "n_days": N_DAYS, "seeds": SEEDS, "sigma_bar": 0.2217,
            "summary": summary, "ctrl": rows_ctrl, "latent": rows_lat,
            "skipped": skipped,
        }), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'controls_1000d.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
