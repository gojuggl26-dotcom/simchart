# -*- coding: utf-8 -*-
"""S10a: κ の対数グリッドスイープ 。

各 κ (× 3 シード × 250 日) で記録する量:
  d の半減期 (1 分 AR(1)) / VR (約定時間 1000 と日次 max) / β̂ / サイズ傾き /
  γ / η / corr(p_obs, p*) 日次レベル・リターン / スプレッド中央値。
κ の選択規則 (§4.2): VR ≈ 1 を満たす範囲で **γ・η・スプレッド・半減期の
4 制約を破らない最弱の κ**。結果は results/S10a/sweep.json へ。

実行: uv run python scripts/s10a_kappa_sweep.py
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
from simchart.validation.micro import estimate_eta
from simchart.validation.suite import _meta_metrics, _qr_metrics

S = 23400.0
SEEDS = (42, 43, 44)
KAPPAS = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0)


def measure(kappa: float, seed: int) -> dict:
    cfg = Config.load(ROOT / "configs" / "s9.yaml").replace(
        stage="S10", seed=seed, kappa=kappa
    )
    r = run(cfg)
    obs = r.observation
    burn = int(cfg.book_burn_in_days * S / obs.step_seconds)
    d = (r.price.log_p_star - obs.log_price)[burn:]
    stride = max(1, int(round(60.0 / obs.step_seconds)))
    dm = d[::stride]
    phi = float(np.corrcoef(dm[:-1], dm[1:])[0, 1])
    hl_min = float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("inf")
    spd = int(round(S / obs.step_seconds))
    lp_d = obs.log_price[burn::spd]
    ps_d = r.price.log_p_star[burn::spd]
    mm = _meta_metrics(r, cfg)
    mq = _qr_metrics(r, cfg)
    dfc = mm["impact_deficit"]
    ev = r.events
    bb = ev.meta["best_bid_tick"]
    ba = ev.meta["best_ask_tick"]
    ok = (bb >= 0) & (ba >= 0) & (ev.t >= cfg.book_burn_in_days * S)
    return {
        "kappa": kappa,
        "seed": seed,
        "d_sd_bp": float(d.std() * 1e4),
        "d_halflife_min": hl_min,
        "vr_tt_1000": dfc["vr_s8_trade_1000"],
        "vr_daily_max": dfc["vr_s8_daily_max"],
        "beta": dfc["beta_measured"],
        "beta_target": dfc["beta_target"],
        "sqrt_slope": dfc["sqrt_law_exponent"],
        "gamma": mm["sign_acf_gamma"]["gamma"],
        "c1": mm["sign_acf_gamma"]["c1"],
        "eta": mq["eta_trade"]["eta"],
        "corr_daily_level": float(np.corrcoef(lp_d, ps_d)[0, 1]),
        "corr_daily_ret": float(np.corrcoef(np.diff(lp_d), np.diff(ps_d))[0, 1]),
        "spread_median": float(np.median(ba[ok] - bb[ok])),
        "obi_h1": mq["obi"].get("corr_h1"),
    }


def main() -> int:
    out_dir = ROOT / "results" / "S10a"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kappa in KAPPAS:
        for seed in SEEDS:
            t0 = time.perf_counter()
            row = measure(kappa, seed)
            rows.append(row)
            print(
                f"k={kappa:>4} s={seed}: HL={row['d_halflife_min']:>6.1f}min "
                f"VRtt={row['vr_tt_1000']:.2f} VRd={row['vr_daily_max']:.2f} "
                f"b={row['beta']:+.3f}(t {row['beta_target']:+.2f}) "
                f"sq={row['sqrt_slope']:.2f} g={row['gamma']:.2f} "
                f"eta={row['eta']:.3f} sp={row['spread_median']:.0f} "
                f"cd={row['corr_daily_level']:+.2f} ({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
    med = {}
    for kappa in KAPPAS:
        sub = [r_ for r_ in rows if r_["kappa"] == kappa]
        med[str(kappa)] = {
            k: float(np.median([r_[k] for r_ in sub if r_[k] is not None]))
            for k in sub[0]
            if k not in ("kappa", "seed")
        }
    print("\n=== medians ===")
    for k, v in med.items():
        print(
            f"k={k:>4}: HL={v['d_halflife_min']:>6.1f} VRtt={v['vr_tt_1000']:.2f} "
            f"VRd={v['vr_daily_max']:.2f} b={v['beta']:+.3f} sq={v['sqrt_slope']:.2f} "
            f"g={v['gamma']:.2f} eta={v['eta']:.3f} sp={v['spread_median']:.0f} "
            f"cd={v['corr_daily_level']:+.2f}"
        )
    (out_dir / "sweep.json").write_text(
        json.dumps(jsonable({"rows": rows, "medians": med,
                             "seeds": SEEDS, "n_days": 250}),
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nsaved {out_dir / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
