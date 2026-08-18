"""生成済みチャート一式を検証する。

    uv run python scripts/verify_charts.py --stage S0

生成器が自分で吐いた数字を自分で確かめても意味が薄いので、**別経路で再計算して
突き合わせる**ことを主眼にしている。

1. OHLC の内部整合  … low <= open, close <= high、価格が正、log_return が
   log(close/open) と一致
2. セッションの連続性 … S0 にはオーバーナイトが無いので、d 日の終値は d+1 日の
   始値と厳密に一致していなければならない
3. **日足の独立な再計算** … 標本チャートについて、1 秒経路から
   ``np.maximum.reduceat`` などの別手段で OHLC を組み直して一致を確認する
   (生成器と同じ関数を使うと、その関数のバグは絶対に見つからないため)
4. **分足 → 日足の集約一致** … 分足標本を日足に畳んだものが、保存された日足と
   一致すること。別のバー幅で作った結果どうしの突き合わせになる
5. **seed からの再生成** … 記録された seed で作り直したチャートの終値
   ダイジェストが一致すること
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_charts import ohlc_from_log_price  # noqa: E402

from simchart import Config, run  # noqa: E402
from simchart.report import results_dir  # noqa: E402


class CheckFailed(AssertionError):
    pass


def check(name: str, condition: bool, detail: str = "") -> bool:
    mark = "OK  " if condition else "NG  "
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    return condition


def ohlc_reference(log_price: np.ndarray, n_bars: int, steps_per_bar: int):
    """``ohlc_from_log_price`` とは別経路で OHLC を組む (照合用)。

    reshape ではなく ``reduceat`` を使い、終値も明示的に添字で取る。
    """
    starts = np.arange(n_bars) * steps_per_bar
    body_high = np.maximum.reduceat(log_price[:-1], starts)
    body_low = np.minimum.reduceat(log_price[:-1], starts)
    open_log = log_price[starts]
    close_log = log_price[starts + steps_per_bar]
    return (
        open_log,
        np.maximum(body_high, close_log),
        np.minimum(body_low, close_log),
        close_log,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, default="S0")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--n-regenerate", type=int, default=3,
                        help="seed から作り直して照合するチャート本数")
    args = parser.parse_args()

    charts_dir = results_dir(args.stage, args.results_dir) / "charts"
    daily = pd.read_parquet(charts_dir / "daily_ohlc.parquet")
    index_df = pd.read_parquet(charts_dir / "charts_index.parquet")
    intraday_path = charts_dir / "intraday_ohlc_sample.parquet"

    n_charts = int(index_df.shape[0])
    n_days = int(daily["day"].max()) + 1
    passed: list[bool] = []

    print(f"{charts_dir}")
    print(f"チャート {n_charts} 本 x {n_days} 日 = {len(daily):,} 行")
    print()
    print("1. OHLC の内部整合")
    o, h, l, c = (daily[k].to_numpy() for k in ("open", "high", "low", "close"))
    passed.append(check("行数", len(daily) == n_charts * n_days, f"{len(daily):,}"))
    passed.append(check("価格が正", bool((o > 0).all() and (l > 0).all())))
    passed.append(check("high >= max(open, close)", bool((h >= np.maximum(o, c)).all())))
    passed.append(check("low <= min(open, close)", bool((l <= np.minimum(o, c)).all())))
    passed.append(check("high >= low", bool((h >= l).all())))
    ret_err = np.max(np.abs(daily["log_return"].to_numpy() - np.log(c / o)))
    passed.append(check("log_return = log(close/open)", ret_err < 1e-12, f"最大誤差 {ret_err:.2e}"))
    passed.append(check("実現ボラが正", bool((daily["realized_vol_annualized"] > 0).all())))

    print()
    print("2. セッションの連続性 (S0 はオーバーナイト無し)")
    o2d = o.reshape(n_charts, n_days)
    c2d = c.reshape(n_charts, n_days)
    gap = np.max(np.abs(c2d[:, :-1] - o2d[:, 1:]))
    passed.append(check("d 日の終値 == d+1 日の始値", gap == 0.0, f"最大差 {gap:.2e}"))
    start_dev = np.max(np.abs(o2d[:, 0] - o2d[0, 0]))
    passed.append(check("全チャートが同じ初期価格", start_dev == 0.0, f"{o2d[0, 0]:.4f}"))

    print()
    print(f"3. 日足の独立な再計算 ({args.n_regenerate} 本)")
    rng = np.random.default_rng(0)
    sample_ids = sorted(rng.choice(n_charts, size=min(args.n_regenerate, n_charts),
                                   replace=False).tolist())
    base = Config(n_days=n_days, steps_per_day=_steps_per_day(charts_dir))
    steps_per_day = base.steps_per_day
    regenerated: dict[int, np.ndarray] = {}
    for cid in sample_ids:
        seed = int(index_df.loc[index_df["chart_id"] == cid, "seed"].iloc[0])
        log_price = run(base.replace(seed=seed)).observation.log_price
        ro, rh, rl, rc = ohlc_reference(log_price, n_days, steps_per_day)
        go, gh, gl, gc = ohlc_from_log_price(log_price, n_days, steps_per_day)
        same = all(np.array_equal(x, y) for x, y in ((ro, go), (rh, gh), (rl, gl), (rc, gc)))
        passed.append(check(f"chart {cid}: reduceat 版と一致", same))
        stored = daily[daily["chart_id"] == cid]
        err = max(
            np.max(np.abs(stored["open"].to_numpy() - np.exp(ro))),
            np.max(np.abs(stored["high"].to_numpy() - np.exp(rh))),
            np.max(np.abs(stored["low"].to_numpy() - np.exp(rl))),
            np.max(np.abs(stored["close"].to_numpy() - np.exp(rc))),
        )
        passed.append(check(f"chart {cid}: 保存値と一致", err == 0.0, f"最大差 {err:.2e}"))
        regenerated[cid] = rc

    print()
    print("4. seed からの再生成 (終値ダイジェスト)")
    for cid, rc in regenerated.items():
        stored_digest = index_df.loc[index_df["chart_id"] == cid, "close_digest"].iloc[0]
        digest = hashlib.sha256(np.ascontiguousarray(rc, dtype=np.float64).tobytes()).hexdigest()[:16]
        passed.append(check(f"chart {cid}", digest == stored_digest, digest))

    print()
    print("5. 分足 -> 日足の集約一致")
    if not intraday_path.exists():
        print("  (分足標本が無いので省略)")
    else:
        intraday = pd.read_parquet(intraday_path)
        agg = intraday.groupby(["chart_id", "day"], sort=True).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
        ).reset_index()
        merged = agg.merge(daily, on=["chart_id", "day"], suffixes=("_min", "_day"))
        passed.append(check("結合行数", len(merged) == len(agg), f"{len(merged):,}"))
        for col in ("open", "high", "low", "close"):
            err = float(np.max(np.abs(merged[f"{col}_min"] - merged[f"{col}_day"])))
            passed.append(check(f"{col} が一致", err == 0.0, f"最大差 {err:.2e}"))

    print()
    n_ok = sum(passed)
    print("-" * 60)
    print(f"検査 {n_ok}/{len(passed)} 合格")
    return 0 if n_ok == len(passed) else 1


def _steps_per_day(charts_dir: Path) -> int:
    import json

    with open(charts_dir / "ensemble_metrics.json", encoding="utf-8") as fh:
        return int(json.load(fh)["generation"]["steps_per_day"])


if __name__ == "__main__":
    raise SystemExit(main())
