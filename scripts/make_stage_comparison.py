"""段階ごとの違いを 1 枚にまとめた比較図を作る (README 用)。

    uv run python scripts/make_stage_comparison.py

同一シードなら全段階で拡散乱数・MSM・緩慢 OU の実現が共通になる
(名前ハッシュ RNG の設計)。したがって左右に並べた 4 段階は「同じ乱数から、
成分を足していったらどう変わるか」を直接示している — 別々の乱数で描いた
それらしい絵の並置ではない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_charts import ohlc_from_log_price  # noqa: E402

from simchart import Config, run  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

N_DAYS = 1200
STEPS = 23400
WINDOW = (520, 640)  # 表示する日の範囲 (120 日)
SEED = 42

STAGES = [
    ("S0", dict(stage="S0"), "constant volatility (GBM only)"),
    ("S1", dict(stage="S1", enable_msm=True, enable_slow_ou=True),
     "+ MSM & slow OU (volatility clustering)"),
    ("S2", dict(stage="S2", enable_msm=True, enable_slow_ou=True, enable_rough=True),
     "+ rough component (H~0.1, jagged volatility)"),
    ("S3", dict(stage="S3", enable_msm=True, enable_slow_ou=True, enable_rough=True,
                enable_jump=True, enable_leverage=True),
     "+ jumps & leverage (fat tails, negative skew)"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = WINDOW
    n_show = hi - lo

    fig, axes = plt.subplots(
        len(STAGES), 2, figsize=(13, 3.0 * len(STAGES)),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    for row, (name, flags, subtitle) in enumerate(STAGES):
        cfg = Config(seed=SEED, n_days=N_DAYS, steps_per_day=STEPS, **flags)
        result = run(cfg)
        lp = result.observation.log_price
        o, h, l, c = ohlc_from_log_price(lp, N_DAYS, STEPS)
        o, h, l, c = (np.exp(v[lo:hi]) for v in (o, h, l, c))

        ax = axes[row, 0]
        x = np.arange(n_show)
        up = c >= o
        ax.vlines(x, l, h, color="0.35", lw=0.7)
        ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.66, color="#2a9d5c")
        ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.66, color="#c4453c")
        ax.set_ylabel("price")
        ax.set_title(f"{name}: {subtitle}", fontsize=10, loc="left")
        if row == len(STAGES) - 1:
            ax.set_xlabel(f"day ({lo}-{hi}, same window and same seed for every stage)")

        # 右: 同じ窓の瞬間ボラ (年率 %)。分単位のサブサンプルがあればそれを使う。
        ax2 = axes[row, 1]
        sub = result.meta.get("l2", {}).get("vol_subsample")
        if sub is not None:
            t_days = np.asarray(sub["t_days"])
            vol = np.exp(np.asarray(sub["log_vol"])) * 100.0
            mask = (t_days >= lo) & (t_days < hi)
            ax2.plot(t_days[mask], vol[mask], lw=0.5, color="#1f6fb4")
        else:
            const = float(np.exp(result.price.log_vol[0])) * 100.0
            ax2.plot([lo, hi], [const, const], lw=1.2, color="#1f6fb4")
        ax2.set_ylabel("sigma (annual, %)")
        ax2.set_ylim(0, 75)
        ax2.set_title("instantaneous volatility", fontsize=9, loc="left")
        if row == len(STAGES) - 1:
            ax2.set_xlabel("day")

        print(f"{name}: done", flush=True)
        del result, lp, o, h, l, c

    fig.suptitle(
        "What each stage adds — identical random numbers, components added one at a time",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / "stage_comparison.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
