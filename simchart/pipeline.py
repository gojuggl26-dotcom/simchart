"""層の組み立てと実行。

駆動方式について
----------------
S0 は「L0 が張った時間グリッド上で L2 を一括生成し、L3 がそれをそのまま観測する」
という**グリッド駆動**である。しかし S6 で L3 がイベント駆動になると、主役は
L1 が生成するイベント時刻に移り、L2 は ``price.at(event_times)`` で問い合わされる
側になる。この転換をコメントではなく構造で表しておくために、駆動ロジックを
:class:`GridDriver` として切り出し、:func:`select_driver` で選ぶ形にしてある。
S6 では ``EventDriver`` を足して ``select_driver`` の分岐を 1 行増やすだけで済む。
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Config
from .layers import (
    build_activity,
    build_book_layer,
    build_calendar,
    build_price_layer,
)
from .rng import STREAM_NAMES, RNGRegistry
from .types import BookSnapshot, EventLog, Observation, PriceProcess, StageResult

__all__ = [
    "run",
    "run_twice",
    "determinism_check",
    "rng_stability_check",
    "rng_diffusion_check",
    "scale_invariance_check",
    "baseline_invariance_check",
    "BASELINE_STAGE",
    "GridDriver",
]

#: 各段階の不変性照合の基準となる直前段階。
#: S2 の合否は「S1 から何が変わらなかったか」で決まる (S2 指示書 §0)。
BASELINE_STAGE: dict[str, str] = {"S2": "S1", "S3": "S2", "S4": "S3", "S5": "S4"}


@dataclass
class _Layers:
    calendar: Any
    activity: Any
    price: Any
    book: Any


class GridDriver:
    """L0 の時間グリッドで L2 を一括生成し、L3 に観測させる。

    S0〜S5 (板を導入するまで) の駆動方式。
    """

    name = "grid"

    def __call__(
        self, layers: _Layers
    ) -> tuple[PriceProcess, Observation, EventLog, BookSnapshot]:
        grid = layers.calendar.simulation_grid()
        price = layers.price.simulate(grid)
        observation, events, book = layers.book.observe(price, layers.calendar, layers.activity)
        return price, observation, events, book


def select_driver(config: Config) -> GridDriver:
    """設定に応じた駆動方式を選ぶ。

    S6 で板層を入れたら、ここに ``EventDriver`` (L1 のイベント時刻で L3 を回し、
    L2 へは ``price.at()`` で問い合わせる) を追加する。
    """
    if config.enable_book:
        raise NotImplementedError(
            "イベント駆動 (EventDriver) は S6 で simchart/pipeline.py に追加します。"
        )
    return GridDriver()


def _build_layers(config: Config, rng: RNGRegistry) -> _Layers:
    calendar = build_calendar(config, rng)
    activity = build_activity(config, rng, calendar)
    price = build_price_layer(config, rng, calendar, activity)
    book = build_book_layer(config, rng, calendar, activity)
    return _Layers(calendar=calendar, activity=activity, price=price, book=book)


def run(config: Config, *, rng: RNGRegistry | None = None) -> StageResult:
    """設定を 1 回実行して :class:`~simchart.types.StageResult` を返す。"""
    started = time.perf_counter()
    registry = rng if rng is not None else RNGRegistry(config.seed)

    layers = _build_layers(config, registry)
    driver = select_driver(config)
    price, observation, events, book = driver(layers)

    runtime = time.perf_counter() - started
    meta: dict[str, Any] = {
        "driver": driver.name,
        "layers": {
            "l0": layers.calendar.name,
            "l1": layers.activity.name,
            "l2": layers.price.name,
            "l3": layers.book.name,
        },
        # L2 の生成時診断 (MSM 切替・OU 統計・成分サブサンプル・拡散 z ダイジェスト)。
        # 生配列を含むので metrics.json へは要約だけを載せること (suite が選別する)。
        "l2": dict(getattr(layers.price, "last_diagnostics", {})),
        "grid": {
            "n_points": price.n_points,
            "t_start_sec": price.t_start,
            "t_end_sec": price.t_end,
            "step_seconds": layers.calendar.step_seconds(),
            "session_seconds": layers.calendar.session_seconds(),
            "n_days": layers.calendar.n_days(),
        },
        "rng_streams_used": list(registry.used_streams()),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    return StageResult(
        stage=config.stage,
        config=config,
        price=price,
        events=events,
        book=book,
        observation=observation,
        runtime_sec=runtime,
        rng_fingerprint=registry.fingerprint(),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# ゲート用の検査
# ---------------------------------------------------------------------------
def run_twice(config: Config) -> tuple[StageResult, StageResult]:
    """同一設定で 2 回実行する (決定性ゲート用)。"""
    return run(config), run(config)


def determinism_check(config: Config, first: StageResult | None = None) -> dict[str, Any]:
    """同一シードでの 2 回実行がビット単位で一致するかを検査する。

    ダイジェストの一致だけでなく、主要配列の :func:`numpy.array_equal` も取る。
    ハッシュ一致は「同じバイト列」の十分条件としては強いが、どの配列が壊れたかを
    切り分けられないため、両方記録する。
    """
    a = first if first is not None else run(config)
    b = run(config)
    arrays = {
        "price.t": (a.price.t, b.price.t),
        "price.log_p_star": (a.price.log_p_star, b.price.log_p_star),
        "price.log_vol": (a.price.log_vol, b.price.log_vol),
        "price.jump_times": (a.price.jump_times, b.price.jump_times),
        "observation.log_price": (a.observation.log_price, b.observation.log_price),
    }
    per_array = {name: bool(np.array_equal(x, y)) for name, (x, y) in arrays.items()}
    digest_a, digest_b = a.digest(), b.digest()
    return {
        "bitwise_identical": bool(all(per_array.values()) and digest_a == digest_b),
        "digest_first": digest_a,
        "digest_second": digest_b,
        "digests_match": digest_a == digest_b,
        "per_array": per_array,
    }


def rng_stability_check(config: Config, n_draws: int | None = None) -> dict[str, Any]:
    """新しいストリームを足しても既存ストリームが不変であることを検査する。

    後段で新ストリームを追加したときに既存の系列が動くと、段階間の差分が
    「新機能の効果」なのか「乱数がずれただけ」なのか区別できなくなる。これは
    段階構築という方法そのものを無効にするので、critical ゲートとして扱う。

    併せて、宣言済みストリームどうしが偶然同じ系列になっていないか (名前ハッシュの
    衝突や実装ミスによる別名化) も確認する。
    """
    draws = n_draws if n_draws is not None else config.validation.rng_probe_draws

    baseline_registry = RNGRegistry(config.seed)
    baseline = {name: baseline_registry.get(name).standard_normal(draws) for name in STREAM_NAMES}

    # 新段階で足されるストリームを模して、先に別のストリームを大量に消費してから、
    # さらに逆順で既存ストリームを取得する。順序依存があればここで露見する。
    probe_names = ("s3.dummy_probe_a", "s3.dummy_probe_b")
    perturbed_registry = RNGRegistry(config.seed, extra_streams=probe_names)
    for probe in probe_names:
        perturbed_registry.get(probe).standard_normal(draws * 3)
    perturbed = {
        name: perturbed_registry.get(name).standard_normal(draws)
        for name in reversed(STREAM_NAMES)
    }

    per_stream = {
        name: bool(np.array_equal(baseline[name], perturbed[name])) for name in STREAM_NAMES
    }

    distinct = True
    names = list(STREAM_NAMES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.array_equal(baseline[names[i]], baseline[names[j]]):
                distinct = False
    return {
        "unchanged": bool(all(per_stream.values())),
        "streams_distinct": bool(distinct),
        "n_streams": len(STREAM_NAMES),
        "n_draws": draws,
        "probe_streams": list(probe_names),
        "per_stream": per_stream,
    }


def rng_diffusion_check(config: Config, result: StageResult) -> dict[str, Any]:
    """パイプラインが消費した拡散乱数が、S0 相当の消費列と一致するかを検査する。

    S1 で MSM / OU のストリームを足しても、``l2.diffusion`` の系列は名前ハッシュ
    方式によって不変のはず。ここでは独立に RNGRegistry を作り、同一シードで
    同じ個数を引いた列のダイジェストを「期待値」として、パイプラインが実際に
    消費した列のダイジェスト (生成時に記録) と突き合わせる。実装が誤って
    ``l2.diffusion`` から先に別の乱数を引いたり、消費個数を変えたりすると
    ここで不一致になる。
    """
    import hashlib

    expected_z = RNGRegistry(config.seed).get("l2.diffusion").standard_normal(
        config.total_steps
    )
    expected = hashlib.sha256(np.ascontiguousarray(expected_z).tobytes()).hexdigest()
    observed = result.meta.get("l2", {}).get("diffusion_digest")
    return {
        "match": bool(observed == expected),
        "expected_digest": expected,
        "observed_digest": observed,
        "n_draws": config.total_steps,
    }


def scale_invariance_check(config: Config, reference_result: StageResult) -> dict[str, Any]:
    """時間スケール不変性の検査 (指示書 §7)。

    同一シード・同一日数のまま ``steps_per_day`` だけを対照解像度に変えて再実行し、
    **日次集計した統計量** (尖度・GPH d・|r| ACF(1)・Var(log sigma)) が許容誤差内で
    一致することを確認する。「1 ステップあたり切替確率」型の実装はここで落ちる。

    同一シードなら MSM の切替過程は物理時間定義により解像度に依らず**ビット単位で
    一致する** (switch_digest で直接確認)。残る差は拡散乱数と OU 乱数の実現差だけ
    なので、日次統計はサンプリング誤差の範囲で一致するはずである。トレランスは
    その実現差の実測分布から設定してある (tests/test_scale_invariance.py)。
    """
    from .validation.scaling import daily_invariance_stats

    v = config.validation
    low_config = config.replace(steps_per_day=v.scale_invariance_steps_per_day)
    low_result = run(low_config)

    hi = daily_invariance_stats(reference_result)
    lo = daily_invariance_stats(low_result)

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, a: float | None, b: float | None, tol: float, relative: bool) -> None:
        if a is None or b is None:
            checks[name] = {"passed": False, "hi": a, "lo": b, "reason": "統計が計算できませんでした"}
            return
        diff = abs(a - b)
        denom = abs(0.5 * (a + b)) if relative else 1.0
        value = diff / denom if denom > 0 else diff
        checks[name] = {
            "passed": bool(value <= tol),
            "hi": a,
            "lo": b,
            "diff": diff,
            "measure": "relative" if relative else "absolute",
            "value": value,
            "tol": tol,
        }

    add("kurtosis_daily", hi["kurtosis_daily"], lo["kurtosis_daily"], v.si_tol_kurtosis_rel, True)
    add("gph_d_daily", hi["gph_d"], lo["gph_d"], v.si_tol_gph_d_abs, False)
    add("acf_abs_r_lag1_daily", hi["acf_abs_lag1"], lo["acf_abs_lag1"], v.si_tol_acf1_abs, False)
    add("var_log_vol", hi["var_log_vol"], lo["var_log_vol"], v.si_tol_var_logvol_abs, False)

    digest_hi = reference_result.meta.get("l2", {}).get("msm", {}).get("switch_digest")
    digest_lo = low_result.meta.get("l2", {}).get("msm", {}).get("switch_digest")
    if config.enable_msm:
        checks["msm_switch_process_identical"] = {
            "passed": bool(digest_hi is not None and digest_hi == digest_lo),
            "hi": digest_hi,
            "lo": digest_lo,
        }
    if config.enable_rough:
        # ラフ成分は専用の物理グリッド (rough_grid_seconds) 上で生成されるため、
        # steps_per_day を変えても経路そのものがビット単位で一致するはず。
        y_hi = reference_result.meta.get("l2", {}).get("rough", {}).get("y_digest")
        y_lo = low_result.meta.get("l2", {}).get("rough", {}).get("y_digest")
        checks["rough_path_identical"] = {
            "passed": bool(y_hi is not None and y_hi == y_lo),
            "hi": y_hi,
            "lo": y_lo,
        }

    return {
        "passed": bool(all(c["passed"] for c in checks.values())),
        "steps_per_day_hi": config.steps_per_day,
        "steps_per_day_lo": v.scale_invariance_steps_per_day,
        "checks": checks,
    }


def baseline_invariance_check(
    config: Config,
    metrics: dict[str, Any],
    baseline_stage: str,
    results_root: str | None = None,
) -> dict[str, Any]:
    """保存済みの前段階 metrics.json と突き合わせ、不変であるべき量を照合する。

    S2 の合否は「何が増えたか」ではなく「何が変わらなかったか」で決まる
    (S2 指示書 §0)。同一シードなら S1 のストリーム (拡散・MSM・OU) は名前ハッシュ
    RNG によりビット単位で不変のはずで、日次統計の差はラフ成分の追加効果だけになる。

    - **gph_d (±0.03) が最重要** — 動いたらスケール分離の失敗であり、ラフ成分が
      MSM/OU の帯域 (1〜100 日) に漏れている (診断手順は指示書 §10)
    - RNG の証人: MSM の成分別切替回数・占有率、OU の x0・経路統計が JSON の
      float 往復 (repr 17 桁) で**厳密一致**すること — S1 のストリームに 1 draw
      でも触れていれば一致しない
    """
    from .report import load_metrics

    try:
        base = load_metrics(baseline_stage, root=results_root)
    except FileNotFoundError as exc:
        return {"passed": False, "baseline_stage": baseline_stage, "error": str(exc), "checks": {}}

    bm = base.get("metrics", {})
    v = config.validation
    checks: dict[str, dict[str, Any]] = {}

    def get(tree: dict, path: str) -> Any:
        node: Any = tree
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def add_abs(name: str, path: str, tol: float) -> None:
        a, b = get(bm, path), get(metrics, path)
        okc = a is not None and b is not None and abs(b - a) <= tol
        checks[name] = {
            "passed": bool(okc), "baseline": a, "current": b,
            "diff": (b - a) if (a is not None and b is not None) else None, "tol": tol,
        }

    # ★最重要: 長スケールの記憶が動いていないこと。
    # 両段階に潜在 log sigma の GPH (③ の構造の直接測定) があればそちらで判定する。
    # 観測 |r| の GPH は S3 のジャンプ・レバレッジが加える白色成分で下方バイアス
    # されるため、S3 以降は潜在側が本判定で観測側は記録になる。
    #
    # ★S5 (chi_2 有効) では判定の帯域を 0.65 → 0.50 に移す (2026-08-21 裁定)。
    # 指示書は「ピークを 20〜40 日に置け」と「gph_d ±0.03」を同時に要求するが、
    # 帯域 0.65 の測定帯は周期 >= 20 日で**設計した 30 日線を必ず含む** — 実測で
    # どの配置でも Δd = -0.08〜-0.11 となり両立不能。帯域 0.50 (周期 >= 70 日 =
    # ゲートが守る長期記憶の帯) では Δd 中央値 +0.0006 で、かつ誤配置 (36〜40 日)
    # は副次調波が帯に入り -0.03〜-0.05 で正しく落ちる (検出力あり)。
    # 測定は同一 run 内のアブレーション (chi は決定論なので厳密に引ける —
    # without_chi 系列は同一シードの S4 潜在 log σ と機械精度で一致する)。
    abl = get(metrics, "chaos.latent_gph_ablation")
    if isinstance(abl, dict) and abl.get("delta_bw050") is not None:
        delta = float(abl["delta_bw050"])
        tol = min(float(v.inv_tol_gph_d_abs), 0.03)
        checks["gph_d"] = {
            "passed": bool(abs(delta) <= tol),
            "basis": "latent_chaos_ablation_bw050",
            "delta_bw050": delta,
            "tol": tol,
            "d_with_chi_bw050": abl.get("d_with_chi_bw050"),
            "d_without_chi_bw050": abl.get("d_without_chi_bw050"),
            # 帯域 0.65 は設計線を含むため記録のみ (汚染ではなく設計の帰結)。
            "delta_bw065_recorded": abl.get("delta_bw065"),
            "baseline_latent_bw065": get(bm, "daily.latent_gph_d.d"),
            "current_latent_bw065": get(metrics, "daily.latent_gph_d.d"),
        }
    elif (
        get(bm, "daily.latent_gph_d.d") is not None
        and get(metrics, "daily.latent_gph_d.d") is not None
    ):
        add_abs("gph_d", "daily.latent_gph_d.d", v.inv_tol_gph_d_abs)
        a_obs, b_obs = get(bm, "daily.gph_abs_r.d"), get(metrics, "daily.gph_abs_r.d")
        checks["gph_d"]["observed_baseline"] = a_obs
        checks["gph_d"]["observed_current"] = b_obs
        checks["gph_d"]["basis"] = "latent_log_sigma"
    else:
        add_abs("gph_d", "daily.gph_abs_r.d", v.inv_tol_gph_d_abs)

    # |r| ACF のべき則指数 (相対 ±10%) と binned R^2 の非劣化。
    g1 = get(bm, "daily.acf_abs_r_powerlaw.gamma")
    g2 = get(metrics, "daily.acf_abs_r_powerlaw.gamma")
    r2_1 = get(bm, "daily.acf_abs_r_powerlaw.r2")
    r2_2 = get(metrics, "daily.acf_abs_r_powerlaw.r2")
    gamma_ok = (
        g1 is not None and g2 is not None and g1 != 0
        and abs(g2 - g1) / abs(g1) <= v.inv_tol_powerlaw_gamma_rel
    )
    checks["absr_powerlaw_gamma"] = {
        "passed": bool(gamma_ok), "baseline": g1, "current": g2,
        "rel_diff": (abs(g2 - g1) / abs(g1)) if (g1 not in (None, 0) and g2 is not None) else None,
        "tol_rel": v.inv_tol_powerlaw_gamma_rel,
        "r2_baseline": r2_1, "r2_current": r2_2,
        "r2_not_degraded": bool(
            r2_1 is not None and r2_2 is not None and r2_2 >= r2_1 - 0.05
        ),
    }

    # 日次 |r| ACF のプロファイル (ラグ 10〜100 の平均 |差|)。
    vals1 = get(bm, "daily.acf_abs_r.values")
    vals2 = get(metrics, "daily.acf_abs_r.values")
    if vals1 and vals2:
        hi = min(len(vals1), len(vals2), 101)
        a1 = np.array([x if x is not None else np.nan for x in vals1[10:hi]])
        a2 = np.array([x if x is not None else np.nan for x in vals2[10:hi]])
        mean_abs = float(np.nanmean(np.abs(a2 - a1)))
        checks["absr_acf_profile"] = {
            "passed": bool(mean_abs <= v.inv_tol_acf_profile_mean_abs),
            "mean_abs_diff": mean_abs, "tol": v.inv_tol_acf_profile_mean_abs,
            "lags": [10, hi - 1],
        }
    else:
        checks["absr_acf_profile"] = {"passed": False, "reason": "ACF 値が取得できません"}

    # 日次尖度: ラフ成分の分散混合で微増するのは正しい (+0.5 まで)。
    #
    # ★S4 (ON 有効時) では判定しない — **この標本量では検定として成立しない**から。
    # 根拠 (10 シード x 2000 日の対応づけ実測、2026-08-20):
    #   S3 の日次尖度は 6.61〜34.79 (中央値 13.07) と 5 倍に振れる。ジャンプが
    #   8 年で 28〜52 本しか無く、尖度が最大級の 1〜2 本に支配されるため。
    #   ペア差 S4-S3 は -4.92 ± 8.54 (t=-1.82, p=0.10) で、符号すら定まらない。
    # 理論上は ON が総分散の share を取ると日中系列の**超過**尖度が 1/(1-share)
    # = 1.25 倍になる (分子 λE[J^4] は 1 乗、分母 (σ²+λE[J²])² は 2 乗で縮むため)。
    # 中央値ベースで +2.52 に相当するが、ペア差の SD 8.54 に対して検出には
    # 90 シード以上要る。±0.5 のトレランスは季節性と無関係にコイン投げになる。
    # ★「有意でない」を「効果がない」と書かないこと: 効果は理論上あり、
    #   測れていないだけである。
    # 設計上の予算量 (JV シェア) は jv_share_preserved が ±0.005 で照合しており
    # 通っている。分散配分そのものは日次 RV の**中央値比 0.8014** (設計 0.80) で
    # 確認済み — 尖度ではなくこちらが分散設計の証人である。
    k1 = get(bm, "daily.moments.kurtosis")
    k2 = get(metrics, "daily.moments.kurtosis")
    kurt_gated = not config.enable_overnight
    checks["kurtosis_daily"] = {
        "passed": bool(
            not kurt_gated
            or (
                k1 is not None and k2 is not None
                and (k2 - k1) <= v.inv_tol_kurtosis_daily_increase
            )
        ),
        "gated": kurt_gated,
        "baseline": k1, "current": k2,
        "increase": (k2 - k1) if (k1 is not None and k2 is not None) else None,
        "tol_increase": v.inv_tol_kurtosis_daily_increase,
        "note": None if kurt_gated else (
            "記録のみ。ジャンプ 30〜50 本では日次尖度のペア差の SD が 8.5 あり "
            "(10 シード実測)、理論効果 +2.5 を検出できない。分散配分の証人は "
            "日次 RV の中央値比 (設計 0.80) のほう。"
        ),
        "kurtosis_close_to_close": get(metrics, "seasonality.overnight.kurtosis_close_to_close"),
    }

    # zeta 曲率: 悪化していない (より凹でなくなっていない) こと。
    c1 = get(bm, "daily.zeta_curvature.c2")
    c2v = get(metrics, "daily.zeta_curvature.c2")
    checks["zeta_c2"] = {
        "passed": bool(
            c1 is not None and c2v is not None and c2v <= c1 + v.inv_tol_zeta_c2_abs
        ),
        "baseline": c1, "current": c2v, "tol": v.inv_tol_zeta_c2_abs,
    }

    # H_latent の不変 (S3 指示書 §9: レバレッジ相関は粗さを変えない)。
    # 両段階に測定があるときだけ (S1 基準には無い)。
    h1 = get(bm, "rough.h_latent.h")
    h2 = get(metrics, "rough.h_latent.h")
    if h1 is not None and h2 is not None:
        checks["h_latent"] = {
            "passed": bool(abs(h2 - h1) <= 0.02),
            "baseline": h1, "current": h2, "diff": h2 - h1, "tol": 0.02,
        }

    # --- S4 固有 ---------------------------------------------------------
    # ジャンプの QV シェア: S4 の強度補正 (jump_intensity_scale) が効いていれば
    # 季節性・ON を入れても S3 から動かない。補正が抜けると ON の取り分の分だけ
    # 拡散側だけが縮んで実測 12.7% → 14.9% に跳ねる (実際にそうなった) ので、
    # この照合は補正欠落を確実に捕らえる。
    jv1 = get(bm, "jumps.generator.jv_share_theory")
    jv2 = get(metrics, "jumps.generator.jv_share_theory")
    if jv1 is not None and jv2 is not None:
        checks["jv_share_preserved"] = {
            "passed": bool(abs(jv2 - jv1) <= 0.005),
            "baseline": jv1, "current": jv2, "diff": jv2 - jv1, "tol": 0.005,
            "intensity_scale": get(metrics, "jumps.generator.intensity_scale_s4"),
        }

    # 観測 |r| の GPH: 季節性は日内周期成分をスペクトルに足して d を**上方**へ
    # 偏らせる (実測 +0.017、範囲 +0.011〜+0.025)。脱季節化でそれが取れることを
    # 記録する。★閾値が緩いのは、この構成ではジャンプ抽選の違いが d を最大
    # ±0.05 動かし、季節性のバイアス (+0.017) を覆い隠すため — 単一経路では
    # 判定できない。除去の**厳密さ**はジャンプ無し構成のテストが検証し
    # (差が 4 桁でゼロ)、水準の判定は多シード中央値のゲートが行う。
    d_base = get(bm, "memory.gph_abs_r.d")
    d_raw = get(metrics, "memory.gph_abs_r.d")
    d_dsn = get(metrics, "seasonality.gph_abs_r.d_true_phi_removed")
    if d_base is not None and d_dsn is not None:
        checks["gph_d_deseasonalized"] = {
            "passed": bool(abs(d_dsn - d_base) <= 0.08),
            "baseline": d_base, "current": d_dsn, "diff": d_dsn - d_base, "tol": 0.08,
            "raw_current": d_raw,
            "raw_diff": (d_raw - d_base) if d_raw is not None else None,
            "basis": "observed_abs_r_primary_bar",
            "note": (
                "ジャンプ抽選差が支配的なため緩い帯。厳密性はテストと多シードで判定"
            ),
        }

    # RNG の証人: 前段階のストリームがビット単位で不変であることの実測 (JSON の
    # float は repr 17 桁で往復するため、厳密一致 = ビット単位一致)。
    table1 = get(bm, "vol.msm.table") or []
    table2 = get(metrics, "vol.msm.table") or []
    msm_ok = (
        len(table1) == len(table2) > 0
        and all(
            r1.get("n_switches") == r2.get("n_switches")
            and r1.get("occupancy_hi") == r2.get("occupancy_hi")
            for r1, r2 in zip(table1, table2)
        )
    )
    # OU の証人: レバレッジ有効時は OU の駆動が価格革新と相関する構成に置き換わる
    # (それがレバレッジそのもの) ため、経路統計は必然的に変わる。x0 は常に
    # l2.vol_slow から引く設計なので、x0 の厳密一致がストリーム健全性の証人になる。
    ou_fields = ("x0",) if config.enable_leverage else ("x0", "sample_var", "sample_mean")
    ou_ok = all(
        get(bm, f"vol.slow_ou.{f}") is not None
        and get(bm, f"vol.slow_ou.{f}") == get(metrics, f"vol.slow_ou.{f}")
        for f in ou_fields
    )
    # ラフ経路の証人 (S2 以降の基準に存在)。レバレッジは fGn の使い方を変えるだけで
    # ラフ経路 Y 自体は変えない。
    y1 = get(bm, "rough.generator.y_digest")
    y2 = get(metrics, "rough.generator.y_digest")
    rough_ok = True if y1 is None else (y1 == y2)
    checks["rng_s1_streams"] = {
        "passed": bool(msm_ok and ou_ok and rough_ok),
        "msm_witness_equal": bool(msm_ok),
        "ou_witness_equal": bool(ou_ok),
        "ou_witness_fields": list(ou_fields),
        "rough_witness_equal": bool(rough_ok),
    }

    return {
        "passed": bool(all(c.get("passed") for c in checks.values())),
        "baseline_stage": baseline_stage,
        "baseline_git_commit": base.get("git_commit"),
        "baseline_seed": (base.get("config") or {}).get("seed"),
        "checks": checks,
    }
