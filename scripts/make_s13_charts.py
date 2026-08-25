"""S13 (多資産) の構成で模擬チャートを N 本生成する。

    uv run python scripts/make_s13_charts.py --n-charts 100

``scripts/generate_charts.py`` は単一資産 (``run``) 専用で、S13 の設定
(``n_assets > 1``) を渡すと ``run()`` が構成エラーで止まる。こちらは
``run_multi`` を使い、**1 実行 = 相関した 3 銘柄**を生成する。チャート番号は
(実行, 資産) の辞書順で、100 本 = 34 実行 (最後の実行は資産 0 のみ)。

日足の作り方 (ここが本スクリプトの中身)
--------------------------------------
1. **日中**は板のミッド (観測) を 1 秒刻みで見て OHLC を取る。高値・安値は
   終値だけからは復元できない量なので、日中経路を実際に見ていることの証拠になる。
2. **日境界**には S4 のオーバーナイト・ギャップを合成する。``log_p_star`` も
   観測系列も日中のみの連続経路で、ギャップは別配列に分離されている
   (``types.PriceProcess.overnight_gaps`` の規約: 「クローズ・トゥ・クローズ系列は
   検証側でギャップと合成して作る」)。合成しないと**日次分散の 20%
   (overnight_variance_share) がチャートから消える。**
3. **出来高**は板の集約約定から日次合計。S0 では「注文流が無いので出来高を
   付けたら捏造」だったが、S6 以降は内生的に決まるので実データとして書ける。

集団としての検証 (individual が「それらしく」見えても足りない)
-----------------------------------------------------------
- **実行をまたぐチャートは独立**でなければならない (名前ハッシュ RNG)
- **同一実行内の 3 銘柄は因子構造どおりに相関**していなければならない。
  ★ただしチャートはクローズ・トゥ・クローズなので、資産別ストリーム由来で
  資産間共分散を持たない ON ギャップの分だけ相関が希釈される:

      corr_cc = corr_intraday x (1 - ON シェア)     (= x 0.80)

  この予測が当たるかを ``ensemble_metrics.json`` の ``prediction`` に記録する。
- σ̄ の正規化: クローズ・トゥ・クローズ日次リターンの SD x √252 ≈ σ̄ x √T_daily

出力 (``results/S13/charts/``)
-----------------------------
``daily_ohlcv.parquet``   全チャートの日足 OHLCV (1 行 = 1 チャート 1 日)
``charts_index.parquet``  チャートごとの seed・資産・パラメータ・要約統計・ダイジェスト
``ensemble_metrics.json`` 集団検証 (こちらが本体)
``images/``               個別チャート 100 枚 + 一覧 (ギャラリー) + showcase
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from simchart.config import TRADING_DAYS_PER_YEAR, Config
from simchart.pipeline import run_multi
from simchart.report import git_info, results_dir
from simchart.validation.base import jsonable, num
from simchart.validation.cross import theoretical_daily_corr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_PER_RUN = 3


# ---------------------------------------------------------------------------
def daily_ohlc(log_price: np.ndarray, n_days: int, steps_per_day: int):
    """1 秒刻みの対数価格から日中 OHLC を作る (バーは両端を含む)。"""
    expected = n_days * steps_per_day + 1
    if log_price.shape[0] != expected:
        raise ValueError(f"点数が合いません: {log_price.shape[0]} != {expected}")
    panel = log_price[:-1].reshape(n_days, steps_per_day)
    open_log = panel[:, 0]
    close_log = log_price[steps_per_day::steps_per_day]
    high_log = np.maximum(panel.max(axis=1), close_log)
    low_log = np.minimum(panel.min(axis=1), close_log)
    return open_log, high_log, low_log, close_log


def compose_overnight(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, gaps: np.ndarray
):
    """日中 OHLC にオーバーナイト・ギャップを累積合成する。

    ``gaps[d]`` は日 d の引けと日 d+1 の寄付の間のリターン。日 d の全価格に
    ``shift[d] = sum(gaps[:d])`` を足すと、寄付が前日終値からギャップだけ
    離れたクローズ・トゥ・クローズ系列になる (実チャートの見た目そのもの)。
    """
    n_days = o.shape[0]
    if gaps.size == 0:
        shift = np.zeros(n_days)
    else:
        if gaps.shape[0] != n_days - 1:
            raise ValueError(f"ギャップ数 {gaps.shape[0]} != n_days-1 ({n_days - 1})")
        shift = np.concatenate([[0.0], np.cumsum(gaps)])
    return o + shift, h + shift, l + shift, c + shift


def realized_vol_5min(log_price: np.ndarray, n_days: int, steps_per_day: int) -> np.ndarray:
    """5 分リターンからの日次実現ボラ (年率)。

    ★1 秒リターンではなく 5 分。1 秒はバイド・アスク・バウンスでマイクロ構造
    ノイズが実現分散を大きく膨らませる (S10b で jv_share が誤検出になった機構)。
    """
    stride = max(int(round(steps_per_day / 78)), 1)  # 6.5h / 5min = 78 本
    sub = log_price[::stride]
    per_day = (sub.shape[0] - 1) // n_days
    r = np.diff(sub[: n_days * per_day + 1])
    rv = (r**2).reshape(n_days, per_day).sum(axis=1)
    return np.sqrt(rv * TRADING_DAYS_PER_YEAR)


def max_drawdown(close: np.ndarray) -> float:
    peak = np.maximum.accumulate(close)
    return float(np.max(1.0 - close / peak))


# ---------------------------------------------------------------------------
def latent_daily_returns(cfg: Config, seed: int) -> dict[int, np.ndarray]:
    """潜在 (p*) の日次クローズ・トゥ・クローズ・リターンを資産ごとに返す。

    **板を外した run_multi から取る。** L2 は板 on/off でビット単位一致する
    (S13 の l2_frozen_multi ゲートが保証) ので潜在経路は同一で、板カーネルを
    回さない分 6 倍速い。流動性オーバーライドは L3/L1 のパラメータのみなので
    ここでも外す (L2 に影響しないことは同ゲートが検証済み)。
    """
    m = run_multi(cfg.replace(seed=seed).without_book().replace(asset_overrides=()))
    out: dict[int, np.ndarray] = {}
    for p in m.payloads:
        close = np.cumsum(p.daily_ret_latent) + np.concatenate(
            [[0.0], np.cumsum(p.overnight_gaps)]
        )
        r = np.empty(p.n_days)
        r[0] = p.daily_ret_latent[0]
        r[1:] = np.diff(close)
        out[p.asset_index] = r
    del m
    return out


def generate_runs(cfg: Config, n_runs: int, parts_dir: Path, base_seed: int) -> list[dict]:
    """必要本数に達するまで run_multi を回し、実行ごとに parts へ保存する。

    窓逸脱 (板内生ミッド歩行 — README S13 節) で落ちたシードは記録の上で
    スキップし、次のシードへ進む。各実行では板ありの生成に加えて**板なしの
    潜在経路**も取り、チャートごとの伝達比 T = Var(観測)/Var(潜在) を
    parts に保存する (品質選別に使う — build_frames を参照)。
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    skipped: list[dict] = []
    seed = base_seed
    started = time.perf_counter()

    # 窓逸脱シードの記録。品質選別で本数が不足すると generate_runs は
    # base_seed から walk し直すので、記録が無いと**落ちると分かっている
    # シードを毎ラウンド 60 秒かけて引き直す** (実測 4 分の無駄)。
    # ★設定 (シード以外) が変われば逸脱するシードも変わるので、
    # config_hash をキーにして古い記録を使い回さない。
    cache_path = parts_dir / "skipped_seeds.json"
    cfg_key = cfg.replace(seed=0).config_hash()
    skip_cache: dict[int, str] = {}
    if cache_path.exists():
        try:
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
            if blob.get("config_hash") == cfg_key:
                skip_cache = {int(k): v for k, v in blob.get("seeds", {}).items()}
        except (ValueError, OSError):
            skip_cache = {}

    def remember_skip(seed_: int, error: str) -> None:
        skip_cache[seed_] = error
        cache_path.write_text(
            json.dumps(
                {"config_hash": cfg_key,
                 "note": "窓逸脱で完走しないシード (設定が変われば無効)",
                 "seeds": {str(k): v for k, v in sorted(skip_cache.items())}},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )

    while len(runs) < n_runs:
        if seed in skip_cache:
            skipped.append({"seed": seed, "error": skip_cache[seed], "from_cache": True})
            seed += 1
            continue
        part = parts_dir / f"run_seed{seed}.npz"
        if part.exists():
            z = np.load(part)
            has_lat = "a0_rlat" in z.files
            if has_lat:
                z.close()
                runs.append({"seed": seed, "path": part})
                seed += 1
                continue
            # 旧版の parts (潜在なし) — 板なしランだけ足して書き直す
            blob = {k: z[k] for k in z.files}
            z.close()
            burn_ = int(round(cfg.book_burn_in_days))
            for a, r in latent_daily_returns(cfg, seed).items():
                blob[f"a{a}_rlat"] = r[burn_:]
            np.savez_compressed(part, **blob)
            print(f"  seed {seed}: 潜在経路を追加 (既存 parts を再利用)", flush=True)
            runs.append({"seed": seed, "path": part})
            seed += 1
            continue
        try:
            multi = run_multi(cfg.replace(seed=seed))
        except RuntimeError as exc:
            skipped.append({"seed": seed, "error": str(exc)})
            remember_skip(seed, str(exc))
            print(f"  seed {seed}: スキップ ({str(exc)[:48]})", flush=True)
            seed += 1
            continue
        latent = latent_daily_returns(cfg, seed)

        spd = int(round(cfg.steps_per_day))
        # ★板のウォームアップ期間を捨てる。板は init_levels=30 x init_size=20 の
        # 人工的に厚い状態から始まり、定常に達するまでが book_burn_in_days
        # (既定 5 日)。この区間は本プロジェクトの統計収集からも除外されており
        # (§8.1)、チャートに残すと全 100 本の冒頭が一様に「不自然に穏やか」になる。
        burn = int(round(cfg.book_burn_in_days))
        blob: dict[str, np.ndarray] = {}
        for p in multi.payloads:
            lp = p.obs_log_price_f32.astype(np.float64)
            o, h, l, c = daily_ohlc(lp, p.n_days, spd)
            # ギャップは全期間で合成してから切る (合成後に切っても系列の連続性は
            # 保たれる — shift は相対的なので冒頭を落とすだけ)
            o, h, l, c = compose_overnight(o, h, l, c, p.overnight_gaps)
            i = p.asset_index
            blob[f"a{i}_o"] = o[burn:]
            blob[f"a{i}_h"] = h[burn:]
            blob[f"a{i}_l"] = l[burn:]
            blob[f"a{i}_c"] = c[burn:]
            blob[f"a{i}_rv"] = realized_vol_5min(lp, p.n_days, spd)[burn:]
            blob[f"a{i}_vol"] = p.daily_volume[burn:]
            blob[f"a{i}_ntr"] = p.daily_n_trades[burn:].astype(np.float64)
            # 日中のみ (ギャップ非合成) の日次リターン — ON 希釈の検証用
            blob[f"a{i}_rid"] = p.daily_ret_obs[burn:]
            blob[f"a{i}_rlat"] = latent[i][burn:]  # 潜在 cc — 伝達比 T の分母
            blob[f"a{i}_beta"] = np.array([p.beta])
            blob[f"a{i}_kappa"] = np.array([p.kappa])
        np.savez_compressed(part, **blob)
        runs.append({"seed": seed, "path": part})
        el = time.perf_counter() - started
        done = len(runs)
        print(f"  実行 {done}/{n_runs} (seed {seed}) 完了  経過 {el:.0f}s / "
              f"残り約 {el / max(done, 1) * (n_runs - done):.0f}s", flush=True)
        del multi
        seed += 1
    return runs, skipped


def build_frames(cfg: Config, runs: list[dict], n_charts: int, t_max: float):
    """parts から日足テーブルと index を組み立てる (品質選別つき)。

    ★選別: 伝達比 T = Var(観測日次 cc)/Var(潜在日次 cc) が ``t_max`` 以上の
    チャートを除外する。板ミッドが p* から decouple した本 (実測で潜在が
    −0.67% の日に観測が −60.9% という例) を弾くため。

    閾値の根拠は **S12 本番 (単一資産・1000 日・24 有効シード) で実測された
    増幅の最大値 2.534** (fb_rv_excess_ari)。S12 で起きた水準までは「設計された
    危機増幅」として通し、それを超える本だけを暴走として落とす — 二重基準を
    避けるため恣意的な丸い数字ではなくタグ済み本番の実測値を使う。
    """
    n_days = cfg.n_days - int(round(cfg.book_burn_in_days))  # ウォームアップ除外後
    rows_daily: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []
    intraday_ret: list[np.ndarray] = []  # 日中のみ日次リターン (ON 希釈の検証)
    rejected: list[dict[str, Any]] = []
    chart_id = 0
    for run in runs:
        z = np.load(run["path"])
        for a in range(ASSETS_PER_RUN):
            if chart_id >= n_charts:
                break
            o, h, l, c = (np.exp(z[f"a{a}_{k}"]) for k in ("o", "h", "l", "c"))
            rv = z[f"a{a}_rv"]
            vol = z[f"a{a}_vol"]
            ntr = z[f"a{a}_ntr"]
            # 日次リターン (クローズ・トゥ・クローズ)。初日は寄付→引け。
            log_c = np.log(c)
            ret_cc = np.empty(n_days)
            ret_cc[0] = log_c[0] - math.log(o[0])
            ret_cc[1:] = np.diff(log_c)

            # --- 品質選別: 伝達比 ---
            r_lat = np.asarray(z[f"a{a}_rlat"])
            v_lat = float(r_lat.var(ddof=1))
            transmission = float(ret_cc.var(ddof=1) / v_lat) if v_lat > 0 else float("inf")
            if transmission >= t_max:
                rejected.append({
                    "seed": int(run["seed"]), "asset": a,
                    "transmission": transmission,
                    "vol_obs": float(ret_cc.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)),
                    "vol_latent": float(math.sqrt(v_lat * TRADING_DAYS_PER_YEAR)),
                    "max_abs_daily_return": float(np.abs(ret_cc).max()),
                })
                continue

            rows_daily.append(
                pd.DataFrame(
                    {
                        "chart_id": np.full(n_days, chart_id, dtype=np.int32),
                        "day": np.arange(n_days, dtype=np.int32),
                        "open": o, "high": h, "low": l, "close": c,
                        "volume": vol, "n_trades": ntr,
                        "log_return": ret_cc,
                        "realized_vol_annualized": rv,
                    }
                )
            )
            intraday_ret.append(np.asarray(z[f"a{a}_rid"]))
            index_rows.append(
                {
                    "chart_id": chart_id,
                    "seed": int(run["seed"]),
                    "asset": a,
                    "beta": float(z[f"a{a}_beta"][0]),
                    "kappa": float(z[f"a{a}_kappa"][0]),
                    "first_open": float(o[0]),
                    "last_close": float(c[-1]),
                    "total_log_return": float(log_c[-1] - math.log(o[0])),
                    "daily_return_std": float(ret_cc.std(ddof=1)),
                    "annualized_vol_cc": float(
                        ret_cc.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
                    ),
                    "realized_vol_median": float(np.median(rv)),
                    "daily_return_kurtosis": float(
                        stats.kurtosis(ret_cc, fisher=False, bias=False)
                    ),
                    "daily_return_skew": float(stats.skew(ret_cc, bias=False)),
                    "max_drawdown": max_drawdown(c),
                    "volume_per_day": float(vol.mean()),
                    "trades_per_day": float(ntr.mean()),
                    "transmission": transmission,
                    "annualized_vol_latent": float(
                        math.sqrt(v_lat * TRADING_DAYS_PER_YEAR)
                    ),
                    # ★伝達比は分散**全体**で見るので、1 日だけ飛んで翌日戻る
                    # 単発スパイク (板の一時的な暴走の名残) は捕まらない。
                    # 用途に応じて絞れるよう最大日次リターンも出す
                    # (潜在の最大は実測 9.1% — これを大きく超える本は要注意)。
                    "max_abs_daily_return": float(np.abs(ret_cc).max()),
                    "close_digest": hashlib.sha256(
                        np.ascontiguousarray(c, dtype=np.float64).tobytes()
                    ).hexdigest()[:16],
                }
            )
            chart_id += 1
        z.close()
    return (
        pd.concat(rows_daily, ignore_index=True),
        pd.DataFrame(index_rows),
        np.stack(intraday_ret),
        rejected,
    )


# ---------------------------------------------------------------------------
def ensemble_metrics(
    daily: pd.DataFrame,
    index_df: pd.DataFrame,
    intraday_ret: np.ndarray,
    cfg: Config,
    runtime_sec: float,
) -> dict[str, Any]:
    """集団としての検証。個々が「それらしい」だけでは足りない。"""
    n_charts = int(index_df.shape[0])
    n_days = cfg.n_days - int(round(cfg.book_burn_in_days))
    ret = daily["log_return"].to_numpy().reshape(n_charts, n_days)
    seeds = index_df["seed"].to_numpy()
    assets = index_df["asset"].to_numpy()
    sigma = cfg.sigma_bar

    # --- 1. 相関: 実行をまたぐ (独立のはず) / 実行内 (因子構造どおりのはず) ---
    corr_cc = np.corrcoef(ret)
    corr_id = np.corrcoef(intraday_ret[:n_charts])
    iu = np.triu_indices(n_charts, k=1)
    same_run = seeds[iu[0]] == seeds[iu[1]]
    cross = corr_cc[iu][~same_run]
    corr_se = 1.0 / math.sqrt(n_days - 1)
    z_cross = cross / corr_se
    n_pairs = int(cross.size)
    tail = 2.0 * stats.norm.sf(float(np.max(np.abs(z_cross))))

    # ★帰無対照 (符号ランダム化)。素朴な SE 1/√(n-1) は**等分散正規**を仮定した
    # 式で、この系のリターン (共有 χ による不均一分散 + Hill α ≈ 2.5 の裾) には
    # 当てはまらない。実際、素朴式では max|z| = 7.4 が p < 1e-9 の「独立性の破れ」
    # に見えるが、各チャートの |r| 経路 (= 共有ボラ構造・共有された極値日) を
    # そのままに符号だけ日ごと独立に振り直した帰無分布では max|z| の中央値が
    # 7.9 で、観測はむしろ**下側**にある。判定はこちらの経験帰無で行う。
    rng_null = np.random.default_rng(0xC0FFEE)
    absr = np.abs(ret)
    n_null = 200
    null_zsd = np.empty(n_null)
    null_maxz = np.empty(n_null)
    for b in range(n_null):
        s = rng_null.choice([-1.0, 1.0], size=ret.shape)
        c_b = np.corrcoef(absr * s)[iu][~same_run] / corr_se
        null_zsd[b] = c_b.std(ddof=1)
        null_maxz[b] = np.max(np.abs(c_b))
    obs_zsd = float(z_cross.std(ddof=1))
    obs_maxz = float(np.max(np.abs(z_cross)))
    theory = theoretical_daily_corr(cfg)
    on_share = cfg.overnight_variance_share if cfg.enable_overnight else 0.0

    within: dict[str, Any] = {}
    for a in range(ASSETS_PER_RUN):
        for b in range(a + 1, ASSETS_PER_RUN):
            m = same_run & (
                ((assets[iu[0]] == a) & (assets[iu[1]] == b))
                | ((assets[iu[0]] == b) & (assets[iu[1]] == a))
            )
            if not m.any():
                continue
            cc = float(np.median(corr_cc[iu][m]))
            ci = float(np.median(corr_id[iu][m]))
            within[f"{a}-{b}"] = {
                "corr_close_to_close": num(cc),
                "corr_intraday": num(ci),
                "predicted_cc_from_intraday": num(ci * (1.0 - on_share)),
                "cc_minus_prediction": num(cc - ci * (1.0 - on_share)),
                "latent_theory_intraday": num((theory["pairs"] or {}).get(f"{a}-{b}")),
                "n_pairs": int(m.sum()),
            }

    # --- 2. σ̄ の正規化と martingale ---
    vol_cc = index_df["annualized_vol_cc"].to_numpy()
    gross = np.exp(ret.sum(axis=1))
    per_asset = {}
    for a in range(ASSETS_PER_RUN):
        sel = assets == a
        if not sel.any():
            continue
        per_asset[str(a)] = {
            "n_charts": int(sel.sum()),
            "annualized_vol_cc_median": num(float(np.median(vol_cc[sel]))),
            "realized_vol_5min_median": num(
                float(index_df["realized_vol_median"].to_numpy()[sel].mean())
            ),
            "volume_per_day_median": num(
                float(np.median(index_df["volume_per_day"].to_numpy()[sel]))
            ),
            "trades_per_day_median": num(
                float(np.median(index_df["trades_per_day"].to_numpy()[sel]))
            ),
            "kurtosis_median": num(
                float(np.median(index_df["daily_return_kurtosis"].to_numpy()[sel]))
            ),
            "max_drawdown_median": num(
                float(np.median(index_df["max_drawdown"].to_numpy()[sel]))
            ),
        }

    pooled = ret.ravel()
    centered = ret - ret.mean(axis=1, keepdims=True)
    acf1 = float((centered[:, :-1] * centered[:, 1:]).sum() / (centered**2).sum())
    absr = np.abs(ret) - np.abs(ret).mean(axis=1, keepdims=True)
    absr_acf1 = float((absr[:, :-1] * absr[:, 1:]).sum() / (absr**2).sum())
    log_range = np.log(daily["high"].to_numpy() / daily["low"].to_numpy())
    abs_body = np.abs(np.log(daily["close"].to_numpy() / daily["open"].to_numpy()))

    return {
        "stage": cfg.stage,
        "n_charts": n_charts,
        "n_days": n_days,
        "n_runs": int(np.unique(seeds).size),
        "assets_per_run": ASSETS_PER_RUN,
        "independence_across_runs": {
            "n_pairs": n_pairs,
            "mean_corr": num(float(cross.mean())),
            "max_abs_corr": num(float(np.max(np.abs(cross)))),
            "z_std": num(obs_zsd),
            "max_abs_z": num(obs_maxz),
            # ★判定はこちら (経験帰無)。素朴式の値は下の naive_* に残す。
            "null_z_std_median": num(float(np.median(null_zsd))),
            "null_z_std_p95": num(float(np.percentile(null_zsd, 97.5))),
            "z_std_pvalue_vs_null": num(float(np.mean(null_zsd >= obs_zsd))),
            "null_max_abs_z_median": num(float(np.median(null_maxz))),
            "null_max_abs_z_p95": num(float(np.percentile(null_maxz, 97.5))),
            "max_abs_z_pvalue_vs_null": num(float(np.mean(null_maxz >= obs_maxz))),
            "n_null_replicates": n_null,
            "naive_corr_se": num(corr_se),
            "naive_expected_max_abs_z": num(
                float(stats.norm.ppf(1.0 - 1.0 / (2.0 * n_pairs)))
            ),
            "naive_max_abs_z_pvalue": num(
                float(-np.expm1(n_pairs * np.log1p(-tail)))
            ),
            "frac_abs_z_over_1_96": num(float(np.mean(np.abs(z_cross) > 1.96))),
            "note": (
                "実行が違えば全ストリームが独立 (名前ハッシュ RNG)。判定は符号"
                "ランダム化の経験帰無で行う — 素朴な SE (等分散正規の式) は"
                "この系の不均一分散と α≈2.5 の裾では偽陽性を出す"
            ),
        },
        "correlation_within_run": within,
        "prediction": {
            "rule": "corr_cc = corr_intraday x (1 - overnight_variance_share)",
            "overnight_variance_share": num(on_share),
            "why": (
                "ON ギャップは資産別ストリーム (l0.overnight) 由来で資産間に"
                "共分散を持たない。合成すると分子は不変・分母が 1/(1-share) 倍に"
                "なるので相関はちょうど (1-share) 倍に希釈される"
            ),
            "max_abs_deviation": num(
                max(abs(v["cc_minus_prediction"]) for v in within.values())
                if within else None
            ),
        },
        "normalization": {
            "sigma_bar": num(sigma),
            "annualized_vol_cc_median_all": num(float(np.median(vol_cc))),
            "note": (
                "クローズ・トゥ・クローズ SD x √252。板の伝達率 T_daily ≈ 0.99 と"
                "マイクロ構造ノイズの両方が乗るので σ̄ 丁度にはならない"
            ),
            "martingale_gross_mean": num(float(gross.mean())),
            "martingale_se": num(float(gross.std(ddof=1) / math.sqrt(n_charts))),
        },
        "per_asset": per_asset,
        "pooled_daily_returns": {
            "n": int(pooled.size),
            "std": num(float(pooled.std(ddof=1))),
            "kurtosis": num(float(stats.kurtosis(pooled, fisher=False, bias=False))),
            "skewness": num(float(stats.skew(pooled, bias=False))),
            "acf_lag1": num(acf1),
            "acf_lag1_z": num(acf1 * math.sqrt(pooled.size)),
            "abs_acf_lag1": num(absr_acf1),
        },
        "ohlc": {
            "mean_log_range": num(float(log_range.mean())),
            "mean_abs_body": num(float(abs_body.mean())),
            "range_to_body_ratio": num(float(log_range.mean() / abs_body.mean())),
            "note": "高値・安値は 1 秒経路から取得 (終値だけからは復元できない量)",
        },
        "runtime_sec": num(runtime_sec),
    }


# ---------------------------------------------------------------------------
def draw_charts(daily: pd.DataFrame, index_df: pd.DataFrame, out_dir: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single = out_dir / "charts"
    single.mkdir(parents=True, exist_ok=True)
    n_charts = int(index_df.shape[0])
    grouped = {cid: g for cid, g in daily.groupby("chart_id")}

    def candles(ax, g, width=0.68):
        x = np.arange(len(g))
        o = g["open"].to_numpy(); h = g["high"].to_numpy()
        l = g["low"].to_numpy(); c = g["close"].to_numpy()
        up = c >= o
        ax.vlines(x, l, h, color="0.35", lw=0.45)
        ax.bar(x[up], (c - o)[up], bottom=o[up], width=width, color="#2a9d5c", lw=0)
        ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=width, color="#c4453c", lw=0)

    # --- 個別チャート (ローソク足 + 出来高) ---
    for cid in range(n_charts):
        g = grouped[cid]
        row = index_df.iloc[cid]
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(13, 5.6), sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
        )
        candles(ax, g)
        ax.set_ylabel("price")
        ax.set_title(
            f"chart {cid:03d}   seed {int(row['seed'])} / asset {int(row['asset'])}"
            f"   (β={row['beta']:.1f}, κ={row['kappa']:.1f})"
            f"   ann.vol {row['annualized_vol_cc'] * 100:.1f}%"
            f"   maxDD {row['max_drawdown'] * 100:.1f}%",
            fontsize=10, loc="left",
        )
        ax.grid(alpha=0.15, lw=0.5)
        v = g["volume"].to_numpy()
        c = g["close"].to_numpy(); o = g["open"].to_numpy()
        axv.bar(np.arange(len(g)), v, width=0.68, lw=0,
                color=np.where(c >= o, "#2a9d5c", "#c4453c"))
        axv.set_ylabel("volume")
        axv.set_xlabel(f"trading day (1-{len(g)})")
        axv.grid(alpha=0.15, lw=0.5)
        fig.savefig(single / f"chart_{cid:03d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # --- ギャラリー (25 本 x 4 ページ、終値ライン) ---
    per_page = 25
    n_pages = math.ceil(n_charts / per_page)
    for page in range(n_pages):
        lo = page * per_page
        hi = min(lo + per_page, n_charts)
        fig, axes = plt.subplots(5, 5, figsize=(16, 11))
        for k, ax in enumerate(axes.ravel()):
            cid = lo + k
            if cid >= hi:
                ax.set_visible(False)
                continue
            g = grouped[cid]
            row = index_df.iloc[cid]
            ax.plot(g["close"].to_numpy(), lw=0.8, color="#1f4e79")
            ax.axhline(float(row["first_open"]), color="r", ls=":", lw=0.6)
            ax.set_title(
                f"{cid:03d}  s{int(row['seed'])}/a{int(row['asset'])}"
                f"  {row['annualized_vol_cc'] * 100:.0f}%",
                fontsize=8,
            )
            ax.tick_params(labelsize=6)
        n_show = len(grouped[lo])
        fig.suptitle(
            f"S13 simulated charts {lo}-{hi - 1} / {n_charts}"
            f"  ({n_show} trading days, close price; dotted = first open)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"gallery_p{page + 1}.png", dpi=100)
        plt.close(fig)

    # --- showcase: 各資産クラスから 2 本ずつ (直近 250 日のローソク) ---
    picks: list[int] = []
    for a in range(ASSETS_PER_RUN):
        sel = index_df.index[index_df["asset"] == a].tolist()
        picks.extend(sel[:2])
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.6 * len(picks)))
    for ax, cid in zip(np.atleast_1d(axes), picks):
        g = grouped[cid].iloc[-250:]
        row = index_df.iloc[cid]
        candles(ax, g)
        ax.set_title(
            f"chart {cid:03d}  asset {int(row['asset'])}"
            f" (β={row['beta']:.1f}, κ={row['kappa']:.1f},"
            f" {row['trades_per_day']:.0f} trades/day)  — last 250 days",
            fontsize=9, loc="left",
        )
        ax.grid(alpha=0.15, lw=0.5)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "showcase.png", dpi=110)
    plt.close(fig)
    return n_charts + n_pages + 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="S13 構成で模擬チャートを生成する")
    ap.add_argument("--n-charts", type=int, default=100)
    ap.add_argument("--config", type=str, default="configs/s13.yaml")
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--n-days", type=int, default=None)
    ap.add_argument("--results-dir", type=str, default=None)
    ap.add_argument(
        "--t-max", type=float, default=2.534,
        help="伝達比 Var(観測)/Var(潜在) の上限。これ以上のチャートは板ミッドが"
             " p* から decouple したものとして除外する。既定は S12 本番 24 シードで"
             " 実測された増幅の最大値 (fb_rv_excess_ari max = 2.534)",
    )
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(PROJECT_ROOT / args.config)
    over: dict[str, Any] = {}
    if args.base_seed is not None:
        over["seed"] = args.base_seed
    if args.n_days is not None:
        over["n_days"] = args.n_days
    if over:
        cfg = cfg.replace(**over)
    if cfg.n_assets < 2:
        raise ValueError(
            f"n_assets={cfg.n_assets} は単一資産です。S13 の設定 (configs/s13.yaml) を"
            f" 使ってください (単一資産は scripts/generate_charts.py)。"
        )

    out_dir = results_dir(cfg.stage, args.results_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_runs = math.ceil(args.n_charts / ASSETS_PER_RUN)
    started = time.perf_counter()
    print(
        f"{args.n_charts} 本 = {n_runs} 実行 x {ASSETS_PER_RUN} 銘柄"
        f" ({cfg.n_days} 日 x {cfg.steps_per_day} ステップ, stage={cfg.stage},"
        f" β={cfg.factor_betas}, base seed={cfg.seed}, 伝達比の上限 {args.t_max})",
        flush=True,
    )
    # 品質選別で落ちる分だけ実行を追加する (必要本数に達するまで)
    runs, skipped = generate_runs(cfg, n_runs, out_dir / "parts", cfg.seed)
    while True:
        daily, index_df, intraday_ret, rejected = build_frames(
            cfg, runs, args.n_charts, args.t_max
        )
        if len(index_df) >= args.n_charts:
            break
        need = args.n_charts - len(index_df)
        add = max(math.ceil(need / ASSETS_PER_RUN), 1)
        print(f"  選別で {len(rejected)} 本除外 → {need} 本不足。実行を "
              f"{add} 追加します", flush=True)
        runs, more_skipped = generate_runs(
            cfg, len(runs) + add, out_dir / "parts", cfg.seed
        )
        skipped = skipped + [s for s in more_skipped if s not in skipped]

    daily.to_parquet(out_dir / "daily_ohlcv.parquet", index=False, compression="zstd")
    index_df.to_parquet(out_dir / "charts_index.parquet", index=False, compression="zstd")
    index_df.to_csv(out_dir / "charts_index.csv", index=False)

    metrics = ensemble_metrics(
        daily, index_df, intraday_ret, cfg, time.perf_counter() - started
    )
    info = git_info()
    payload = {
        "stage": cfg.stage,
        "git_commit": info["commit"],
        "git_dirty": info["dirty"],
        "config_hash": cfg.config_hash(),
        "config": cfg.to_dict(),
        "generation": {
            "n_charts": args.n_charts,
            "n_runs": len(runs),
            "assets_per_run": ASSETS_PER_RUN,
            "seeds_used": [int(r["seed"]) for r in runs],
            "seeds_skipped": skipped,
            "overnight_composed": bool(cfg.enable_overnight),
            "runtime_sec": time.perf_counter() - started,
        },
        "screening": {
            "criterion": "transmission = Var(obs daily cc) / Var(latent daily cc)",
            "t_max": args.t_max,
            "t_max_basis": (
                "S12 本番 (単一資産 1000 日 x 24 有効シード) の fb_rv_excess_ari "
                "max = 2.534。S12 で実測された増幅までは設計された危機増幅として"
                "通し、それを超える本を板ミッドの暴走として落とす"
            ),
            "n_rejected": len(rejected),
            "rejected": rejected,
            "transmission_percentiles": {
                str(q): float(np.percentile(index_df["transmission"], q))
                for q in (5, 25, 50, 75, 90, 95, 100)
            },
            "transmission_median_by_asset": {
                str(a): float(index_df.loc[index_df["asset"] == a, "transmission"].median())
                for a in range(ASSETS_PER_RUN)
            },
            "note": (
                "★暴走は「窓に収まったか」では捕まらない (窓逸脱でスキップされた"
                "実行とは別に、窓内に留まりながら decouple する本がある)。"
                "流動性 (実効アンカー束 κ·μ) が低い資産ほど起きやすい — "
                "資産 1 (κ·μ が参照の 3 倍) は全数が T ≤ 1.6 で健全"
            ),
        },
        "ensemble": jsonable(metrics),
    }
    with open(out_dir / "ensemble_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)

    if not args.no_plots:
        n_img = draw_charts(daily, index_df, out_dir / "images")
        print(f"画像 {n_img} 枚を {out_dir / 'images'} へ", flush=True)

    ind = metrics["independence_across_runs"]
    print()
    print(f"チャート {metrics['n_charts']} 本 / 実行 {metrics['n_runs']} "
          f"(スキップ {len(skipped)}) / {metrics['n_days']} 日")
    print(f"  実行間の独立性 (符号ランダム化の帰無で判定):")
    print(f"    平均相関 {ind['mean_corr']:+.5f} / z の SD {ind['z_std']:.3f} "
          f"(帰無中央値 {ind['null_z_std_median']:.3f}, p={ind['z_std_pvalue_vs_null']:.3f})")
    print(f"    max|z| {ind['max_abs_z']:.2f} "
          f"(帰無中央値 {ind['null_max_abs_z_median']:.2f}, "
          f"p={ind['max_abs_z_pvalue_vs_null']:.3f})"
          f"  [素朴式なら p={ind['naive_max_abs_z_pvalue']:.1e} = 偽陽性]")
    for k, v in metrics["correlation_within_run"].items():
        print(f"  実行内 {k}: cc {v['corr_close_to_close']:+.3f} "
              f"(日中 {v['corr_intraday']:+.3f} x 0.8 = "
              f"{v['predicted_cc_from_intraday']:+.3f} 予測、"
              f"差 {v['cc_minus_prediction']:+.3f})")
    scr = payload["screening"]
    print(f"  品質選別: 伝達比 T < {args.t_max} で {scr['n_rejected']} 本除外 "
          f"(T 中央値 {scr['transmission_percentiles']['50']:.2f}, "
          f"p95 {scr['transmission_percentiles']['95']:.2f}) / "
          f"資産別中央値 " + ", ".join(
              f"a{k} {v:.2f}" for k, v in scr["transmission_median_by_asset"].items()))
    print(f"  年率ボラ (cc) 中央値 {metrics['normalization']['annualized_vol_cc_median_all']:.4f} "
          f"(σ̄ {cfg.sigma_bar}) / 潜在 "
          f"{index_df['annualized_vol_latent'].median():.4f}")
    print(f"  日次リターン 尖度 {metrics['pooled_daily_returns']['kurtosis']:.2f} / "
          f"|r| ACF(1) {metrics['pooled_daily_returns']['abs_acf_lag1']:+.3f} / "
          f"ACF(1) {metrics['pooled_daily_returns']['acf_lag1']:+.4f}")
    print(f"所要 {time.perf_counter() - started:.0f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
