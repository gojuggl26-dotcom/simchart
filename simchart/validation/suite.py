"""検証スイート全体の実行。

``run_all`` は :class:`~simchart.types.StageResult` だけを入力に取り、
tails / memory / scaling / micro / cross の 5 群を辞書で返す。個々の推定器は
:func:`~simchart.validation.base.safe_call` でくるんであり、1 つが壊れても
他の指標を失わない。壊れた箇所は ``status="error"`` として残り、
``validation_callable`` ゲートがそれを見る。

系列の作り方についての決定
--------------------------
- **リターンの自己相関はセッション構造を保ったまま測る** (日をまたぐ差分を
  作らない)。S0 にオーバーナイトは無いが、S4 で入れた瞬間に効いてくる。
- **|リターン| の長期記憶は連結した 1 次元系列で測る**。ボラティリティ過程は
  日をまたいで続くものなので、日をまたぐラグを見られないと「今日のボラが明日の
  ボラを予測するか」が測れない。連結によって偽の値が生じることはない
  (各 |r| はセッション内で完結した値であり、並べているだけ)。
- スケール依存量 (尖度・zeta_q・signature plot) は最細粒度のリターンを
  セッション内で足し上げて作る。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ..config import TRADING_DAYS_PER_YEAR, Config
from ..types import StageResult
from . import chaos as chaos_val
from . import cross, ensemble, memory, micro, scaling, seasonality, tails
from .base import na, safe_call

__all__ = ["run_all", "collect_errors", "flatten", "standardized_returns"]


def _latent_gph(result: StageResult, bandwidth_exponent: float) -> dict:
    """潜在 log sigma の日次平均系列に対する GPH (③ の構造の直接測定)。"""
    sub = result.meta.get("l2", {}).get("vol_subsample")
    if sub is None:
        return {"status": "not_applicable", "reason": "確率ボラが無効です", "value": None}
    log_vol = np.asarray(sub["log_vol"], dtype=np.float64)
    t_days = np.asarray(sub["t_days"], dtype=np.float64)
    n_days = int(round(t_days[-1] - t_days[0])) or 1
    per_day = log_vol.shape[0] // n_days
    daily_mean = log_vol[: n_days * per_day].reshape(n_days, per_day).mean(axis=1)
    return memory.gph_estimator(daily_mean, bandwidth_exponent=bandwidth_exponent)


def standardized_returns(result: StageResult, bar_seconds: float) -> np.ndarray:
    """真の瞬間ボラで標準化したバーリターン (条件付きで iid N(0,1))。

    確率ボラの下では、リターンが無相関でも Ljung-Box や ACF の z 検定は**サイズが
    歪む** (rho の標本分散が iid 時の E[sigma^4]/E[sigma^2]^2 倍に膨らみ、S1 の
    予算では Q(20) の期待値だけで p < 0.01 に達する)。「ボラ変動が方向情報を
    作っていない」ことを正しく検定するには、真の sigma で標準化してから測る。
    シミュレータは真値を知っているのでこれができる。

    バー r の条件付き分布は N(sum (mu - sigma^2/2) dt, sum sigma^2 dt) なので、

        z_bar = (r - drift_bar) / sqrt(sum sigma^2 dt)

    は厳密に iid N(0,1)。**S0 では sigma が定数なのでこの標準化は定数スケールに
    退化し、z 統計量は非標準化版と完全に同値になる** — つまり S0 の測定の自然な
    一般化であり、段階間比較を壊さない。

    前提: 観測グリッドと価格グリッドが同一 (S0〜S5)。S6 で観測が板イベントに
    移ったら、この関数は price.log_vol を ``price.vol_at`` で引く形に直すこと。
    """
    obs = result.observation
    if obs.step_seconds is None:
        raise ValueError("等間隔観測でないと標準化できません (S6 以降は要改修)")
    if obs.n_points != result.price.n_points:
        raise ValueError("観測グリッドと価格グリッドが一致していません (S6 以降は要改修)")

    steps_per_bar = int(round(bar_seconds / obs.step_seconds))
    n_bars_per_day = int(obs.session_seconds // bar_seconds)
    n_days = int(round((obs.t[-1] - obs.t[0]) / obs.session_seconds))

    dt_years = obs.step_seconds / (TRADING_DAYS_PER_YEAR * obs.session_seconds)
    sigma2_left = np.exp(2.0 * result.price.log_vol[:-1])
    steps_per_day = int(round(obs.session_seconds / obs.step_seconds))
    sigma2_days = sigma2_left.reshape(n_days, steps_per_day)
    usable = n_bars_per_day * steps_per_bar
    rv_bars = (
        sigma2_days[:, :usable].reshape(n_days, n_bars_per_day, steps_per_bar).sum(axis=2)
        * dt_years
    )

    bars = obs.to_bars(bar_seconds)
    r = bars.returns_2d()
    mu = result.config.mu_drift
    drift = mu * (steps_per_bar * dt_years) - 0.5 * rv_bars
    return ((r - drift) / np.sqrt(rv_bars)).ravel()


def run_all(result: StageResult, config: Config | None = None) -> dict[str, Any]:
    """全検証関数を実行して指標の入れ子辞書を返す。"""
    cfg = config if config is not None else result.config
    v = cfg.validation
    obs = result.observation

    step_seconds = obs.step_seconds if obs.step_seconds is not None else float(np.median(np.diff(obs.t)))
    seconds_per_year = TRADING_DAYS_PER_YEAR * obs.session_seconds

    base_bars = obs.to_bars(step_seconds)
    primary_bars = obs.to_bars(v.primary_bar_sec)
    r_primary_2d = primary_bars.returns_2d()
    r_primary = r_primary_2d.ravel()
    abs_r_primary = np.abs(r_primary)

    # S4: 真の phi をバー粒度で取っておく (季節性が無い段階では None)。
    # 分散比・粗さ H・GPH はいずれも日内の決定論的な不均一分散に汚染されるので、
    # 判定はこれで割った系列で行う (詳細は各測定の直前のコメント)。
    phi_bars = _true_phi_bars_for(result, cfg, r_primary_2d.shape[1])

    metrics: dict[str, Any] = {}

    metrics["series"] = {
        "source": obs.source,
        "step_seconds": float(step_seconds),
        "session_seconds": float(obs.session_seconds),
        "n_sessions": int(primary_bars.n_days),
        "primary_bar_sec": int(v.primary_bar_sec),
        "n_primary_returns": int(r_primary.size),
        "n_base_returns": int(base_bars.n_returns),
        "n_observations": int(obs.n_points),
        "seconds_per_year": float(seconds_per_year),
    }

    # ------------------------------------------------------------------
    metrics["tails"] = {
        "moments": safe_call(tails.basic_moments, r_primary),
        "hill": safe_call(tails.hill_estimator, r_primary, v.hill_primary_k_frac, "both"),
        "hill_profile": safe_call(tails.hill_profile, r_primary, v.hill_k_fracs, "both"),
        "hill_profile_right": safe_call(tails.hill_profile, r_primary, v.hill_k_fracs, "right"),
        "hill_profile_left": safe_call(tails.hill_profile, r_primary, v.hill_k_fracs, "left"),
        "qq_normal": safe_call(tails.qq_normal, r_primary, v.qq_n_points),
    }

    # ------------------------------------------------------------------
    # 標準化リターン (真の sigma で除す)。S0 では非標準化と z 統計が同値になり、
    # S1 以降は「ボラ変動が方向情報を作っていない」ことの正しいサイズの検定になる。
    try:
        r_std = standardized_returns(result, v.primary_bar_sec)
        r_std_2d = r_std.reshape(r_primary_2d.shape)
        std_error = None
    except Exception as exc:  # noqa: BLE001
        r_std = None
        r_std_2d = None
        std_error = f"{type(exc).__name__}: {exc}"

    acf_r = safe_call(memory.acf, r_primary_2d, v.acf_max_lag)
    acf_abs_r = safe_call(memory.acf, abs_r_primary, v.acf_abs_max_lag)
    metrics["memory"] = {
        "acf_r": acf_r,
        "acf_abs_r": acf_abs_r,
        "acf_r_std": (
            safe_call(memory.acf, r_std_2d, v.acf_max_lag)
            if r_std_2d is not None
            else {"status": "error", "value": None, "error": std_error}
        ),
        "ljung_box_r_std": (
            safe_call(memory.ljung_box, r_std_2d, v.ljung_box_lags, v.ljung_box_primary_lag)
            if r_std_2d is not None
            else {"status": "error", "value": None, "error": std_error}
        ),
        "acf_abs_r_power_law": safe_call(memory.acf_power_law, acf_abs_r, v.micro_fit_lag_range),
        "ljung_box_r": safe_call(
            memory.ljung_box, r_primary_2d, v.ljung_box_lags, v.ljung_box_primary_lag
        ),
        "ljung_box_abs_r": safe_call(
            memory.ljung_box, np.abs(r_primary_2d), v.ljung_box_lags, v.ljung_box_primary_lag
        ),
        "gph_abs_r": safe_call(memory.gph_estimator, abs_r_primary, v.gph_bandwidth_exponent),
        "gph_abs_r_profile": {
            f"exp_{exponent}": safe_call(memory.gph_estimator, abs_r_primary, float(exponent))
            for exponent in v.gph_bandwidth_profile
        },
        "local_whittle_abs_r": safe_call(
            memory.local_whittle, abs_r_primary, v.lw_bandwidth_exponent
        ),
    }

    # ------------------------------------------------------------------
    r_base_2d = base_bars.returns_2d()
    adf_log_price = safe_call(
        scaling.adf_test, primary_bars.log_price_flat(), v.adf_maxlag, "c"
    )
    adf_returns = safe_call(scaling.adf_test, r_primary, v.adf_maxlag, "c")
    # ★S4: 分散比は**脱季節化した**価格で測る。
    # Lo-MacKinlay の重複窓は、窓に含まれる回数がセッションの端で少なくなる
    # (バー j は内部なら q 個の窓に入るが、端では 1 個しか入らない)。日内の分散が
    # 一定ならこれは無害だが、φ² が最大なのは寄付と引け = まさにその端なので、
    # q 期分散だけが系統的に過小評価される。実測は q=64 で VR 0.847 まで落ち、
    # **φ だけから (乱数も価格も使わず) 計算した予測 0.852 と一致した** ので、
    # マルチンゲール性の破れではなく推定量の重み付けの問題と確定している。
    # 脱季節化して測り直すと max|VR-1| は 0.151 → 0.016 (S3 の 0.021 と同水準)。
    vr_series = (
        _deseasonalized_log_price(r_primary_2d, phi_bars)
        if phi_bars is not None
        else primary_bars.log_price
    )
    metrics["scaling"] = {
        "variance_ratio": safe_call(scaling.variance_ratio, vr_series, v.vr_qs),
        "variance_ratio_deseasonalized": phi_bars is not None,
        "variance_ratio_raw": (
            safe_call(scaling.variance_ratio, primary_bars.log_price, v.vr_qs)
            if phi_bars is not None
            else {"status": "not_applicable", "reason": "季節性なし (variance_ratio と同一)",
                  "value": None}
        ),
        "kurtosis_by_scale": safe_call(
            scaling.kurtosis_by_scale, r_base_2d, v.scales_sec, step_seconds, v.min_obs_for_gate
        ),
        "zeta_q": safe_call(
            scaling.zeta_q, r_base_2d, v.zeta_qs, v.scales_sec, step_seconds, v.min_obs_for_gate
        ),
        "signature_plot": safe_call(
            scaling.signature_plot,
            base_bars.log_price,
            v.scales_sec,
            step_seconds,
            seconds_per_year,
            v.min_obs_for_gate,
        ),
        "adf_log_price": adf_log_price,
        "adf_returns": adf_returns,
        "adf": safe_call(scaling.adf_combined, adf_log_price, adf_returns, 0.01),
    }

    # ------------------------------------------------------------------
    # daily: 日次集計系列に対する測定 (S1 の長期記憶・集計正規性ゲートが参照する)。
    # MSM の成分レンジ (1〜500 日) は日内より長いため、記憶とマルチフラクタル性は
    # 日次スケールで測る。S0 でも計算され、d ~ 0・尖度 ~ 3 が基準値になる。
    daily_bars = obs.to_bars(obs.session_seconds)
    r_daily = daily_bars.returns()
    abs_r_daily = np.abs(r_daily)
    metrics["daily"] = {
        "n_days": int(r_daily.size),
        "moments": safe_call(tails.basic_moments, r_daily),
        "acf_r": safe_call(memory.acf, r_daily, min(v.daily_acf_max_lag, r_daily.size - 1)),
        "acf_abs_r": safe_call(
            memory.acf, abs_r_daily, min(v.daily_acf_max_lag, r_daily.size - 1)
        ),
        "acf_abs_r_powerlaw": safe_call(
            memory.acf_powerlaw_fit, abs_r_daily, v.daily_powerlaw_lag_range, v.daily_acf_max_lag
        ),
        "gph_abs_r": safe_call(
            memory.gph_estimator, abs_r_daily, v.daily_gph_bandwidth_exponent
        ),
        "local_whittle_abs_r": safe_call(
            memory.local_whittle, abs_r_daily, v.daily_gph_bandwidth_exponent
        ),
        "hill": safe_call(tails.hill_estimator, r_daily, v.hill_primary_k_frac, "both"),
        "kurtosis_decay": safe_call(
            scaling.kurtosis_decay_fit, r_daily, v.daily_scales_days, v.daily_min_obs_for_gate
        ),
        "zeta_q": safe_call(
            scaling.zeta_q,
            r_daily[None, :],
            v.daily_zeta_qs,
            v.daily_scales_days,
            1.0,  # step = 1 日
            v.daily_min_obs_for_gate,
        ),
        "zeta_curvature": safe_call(scaling.zeta_curvature, r_daily),
        # 潜在 log sigma (日次平均) の GPH d — ③ の**構造**の直接測定 (S3 で追加)。
        # 観測 |r| の GPH はジャンプ・レバレッジが加える白色成分でスペクトル勾配が
        # 平坦化し、真の記憶が不変でも d の測定値が下方にバイアスされる
        # (perturbed fractional process)。log sigma 自体の記憶は MSM/OU/rough の
        # 法則で決まり、S3 の追加成分はそれを変えないので、こちらで不変性を判定する。
        "latent_gph_d": safe_call(_latent_gph, result, v.daily_gph_bandwidth_exponent),
    }

    # ------------------------------------------------------------------
    # vol: ボラ過程そのものの診断 (S1 で追加)。
    # - path: この経路の成分分散シェア (遅い成分のせいで大きくゆらぐ — 記録用)
    # - ensemble: 定常断面での正規化検証 (E[sigma^2]・分散予算のゲートはこちら)
    # - msm: 切替動学の検証 (実測切替率 vs 指定 gamma_i)
    l2_meta = result.meta.get("l2", {})
    sub = l2_meta.get("vol_subsample")
    path_components: dict[str, np.ndarray] = {}
    if sub is not None and cfg.enable_msm:
        path_components["msm"] = np.asarray(sub["half_log_msm"])
    if sub is not None and cfg.enable_slow_ou:
        path_components["slow_ou"] = np.asarray(sub["x_slow"])
    if sub is not None and cfg.enable_chaos_vol:
        path_components["chaos"] = np.asarray(sub["chi_term"])
    metrics["vol"] = {
        "path_budget": safe_call(scaling.vol_variance_budget, path_components or None),
        "ensemble": safe_call(ensemble.vol_cross_section, cfg),
        "msm": safe_call(scaling.msm_diagnostics, l2_meta.get("msm")),
        "slow_ou": (
            {"status": "ok", "value": None, **{
                k_: v_ for k_, v_ in l2_meta.get("slow_ou", {}).items()
            }}
            if cfg.enable_slow_ou and l2_meta.get("slow_ou")
            else {"status": "not_applicable", "reason": "enable_slow_ou=False", "value": None}
        ),
    }

    # ------------------------------------------------------------------
    # rough: 粗さの測定 (S2 で追加)。★H の測定窓 (5 分〜4 時間) と GPH の測定窓
    # (1〜100 日) は重ねない — 前者はラフ成分、後者は MSM/OU が支配する帯域で、
    # 重ねると互いに汚染してどちらの推定も信用できなくなる (S2 指示書 §7)。
    # H は潜在 log sigma (真値) で判定し、RV 側は記録のみ (推定誤差で下方に偏る
    # のが既知であり、実証と同じ見え方をするかの参考値)。
    sub = result.meta.get("l2", {}).get("vol_subsample")
    rough_meta = result.meta.get("l2", {}).get("rough")
    if sub is not None:
        sub_dt_sec = float(sub["step_seconds"]) * float(sub["stride"])
        h_scales_steps = [
            max(int(round(s / sub_dt_sec)), 1) for s in v.rough_h_scales_seconds
        ]
        log_vol_raw = np.asarray(sub["log_vol"])
        y_rough_sub = np.asarray(sub.get("y_rough", np.zeros(0)))
        # ★S4: 粗さ H は**脱季節化した** log sigma で測る。
        # phi(u) は日内で滑らかに変化する決定論的成分なので、そのまま H を測ると
        # 5 分〜4 時間の増分が滑らかになり H が跳ね上がる (本番実測 0.136 -> 0.310)。
        # これは長期記憶への汚染 (GPH d で +0.017) よりはるかに大きく、S4 を作る
        # 動機そのもの — 決定論的な時間構造を確率的な構造と取り違える典型例である。
        # 生成側が log phi を残してくれているので、真値で厳密に差し引ける。
        log_phi_sub = np.asarray(sub.get("log_phi_sigma", np.zeros(0)))
        deseasonalized = bool(
            log_phi_sub.shape == log_vol_raw.shape and np.any(log_phi_sub)
        )
        log_vol_sub = log_vol_raw - log_phi_sub if deseasonalized else log_vol_raw
    else:
        h_scales_steps = []
        log_vol_raw = np.zeros(0)
        log_vol_sub = np.zeros(0)
        y_rough_sub = np.zeros(0)
        deseasonalized = False

    def _h_rv() -> dict:
        if obs.step_seconds is None or v.rv_window_seconds % obs.step_seconds != 0:
            return {"status": "not_applicable", "reason": "RV 窓が刻みの整数倍ではありません", "value": None}
        steps_per_window = int(v.rv_window_seconds / obs.step_seconds)
        rv = scaling.realized_variance(np.diff(obs.log_price), steps_per_window)
        if not np.all(rv > 0):
            return {"status": "not_applicable", "reason": "RV に 0 が含まれます", "value": None}
        log_sigma_rv = 0.5 * np.log(rv)
        rv_scales = [
            max(int(round(s / v.rv_window_seconds)), 1)
            for s in v.rough_h_scales_seconds
            if s >= v.rv_window_seconds
        ]
        return scaling.roughness_exponent(log_sigma_rv, rv_scales, v.rough_h_qs)

    metrics["rough"] = {
        "generator": (
            {"status": "ok", "value": None, **{
                k_: v_ for k_, v_ in rough_meta.items()
            }}
            if rough_meta
            else {"status": "not_applicable", "reason": "enable_rough=False", "value": None}
        ),
        # 判定用 (S4 では脱季節化済み)。S0〜S3 では生と同一。
        "h_latent": safe_call(
            scaling.roughness_exponent, log_vol_sub, h_scales_steps, v.rough_h_qs
        ),
        "h_latent_deseasonalized": deseasonalized,
        # 記録用: 季節性を残したまま測るとどれだけ歪むか (S4 の動機の定量化)。
        "h_latent_raw": (
            safe_call(scaling.roughness_exponent, log_vol_raw, h_scales_steps, v.rough_h_qs)
            if deseasonalized
            else {"status": "not_applicable", "reason": "季節性なし (h_latent と同一)", "value": None}
        ),
        "h_pure_y": (
            safe_call(scaling.roughness_exponent, y_rough_sub, h_scales_steps, v.rough_h_qs)
            if rough_meta
            else {"status": "not_applicable", "reason": "enable_rough=False", "value": None}
        ),
        "h_rv": safe_call(_h_rv),
        "increment_acf": safe_call(memory.vol_increment_acf, log_vol_sub, v.vol_incr_acf_max_lag),
        "stationarity_y": (
            safe_call(scaling.path_stationarity, y_rough_sub)
            if rough_meta
            else {"status": "not_applicable", "reason": "enable_rough=False", "value": None}
        ),
        "share_of_budget_path": (
            {
                "status": "ok",
                "value": rough_meta["sample_var"] / cfg.vol_var_budget_total,
                "sample_var": rough_meta["sample_var"],
                "var_discrete": rough_meta["var_discrete"],
                "budget": cfg.vol_var_budget_total,
                "note": (
                    "ラフ成分は半減期が短く経路分散が良く推定できる (SD ~2.5%) ため、"
                    "予算ゲートは経路実測で判定できる (MSM/OU は断面でしか判定できない)"
                ),
            }
            if rough_meta
            else {"status": "not_applicable", "reason": "enable_rough=False", "value": None}
        ),
    }

    # ------------------------------------------------------------------
    # jumps / leverage: S3 の測定。
    # ★Hill α は測定条件 (日次リターン・上位 5%) を固定して報告する (§3.2) —
    # 条件を書かない α の議論は無意味。α ≈ 3〜5 は有限標本の性質として狙う。
    l2m = result.meta.get("l2", {})
    steps_per_day_obs = (
        int(round(obs.session_seconds / obs.step_seconds)) if obs.step_seconds else None
    )
    rv_daily = (
        scaling.realized_variance(np.diff(obs.log_price), steps_per_day_obs)
        if steps_per_day_obs
        else None
    )
    metrics["jumps"] = {
        "generator": (
            {"status": "ok", "value": None, **l2m["jump"]}
            if l2m.get("jump")
            else {"status": "not_applicable", "reason": "enable_jump=False", "value": None}
        ),
        "bns": (
            safe_call(tails.bns_jump_test, np.diff(obs.log_price), steps_per_day_obs)
            if steps_per_day_obs
            else {"status": "not_applicable", "reason": "等間隔観測ではありません", "value": None}
        ),
        "hill_daily_top5": safe_call(tails.hill_estimator, r_daily, 0.05, "both"),
        "hill_by_scale": safe_call(tails.hill_by_scale, r_daily),
        "skewness_by_scale": safe_call(scaling.skewness_by_scale, r_daily),
    }
    if l2m.get("leverage"):
        lev_raw = dict(l2m["leverage"])
        mid_raw = dict(l2m.get("leverage_mid") or {})
        lev_gen = {
            "status": "ok",
            "value": None,
            **lev_raw,
            "mid": mid_raw or None,
            # ゲート (§6.4) が参照する {realized, target} の組。
            "corr_rough_check": {
                "realized": lev_raw.get("corr_rough_realized"),
                "target": lev_raw.get("rho_rough"),
            },
            "corr_slow_check": {
                "realized": lev_raw.get("corr_slow_realized"),
                "target": lev_raw.get("rho_slow"),
            },
            "corr_mid_check": {
                "realized": mid_raw.get("corr_mid_realized"),
                "target": mid_raw.get("rho_mid"),
            },
        }
    else:
        lev_gen = {"status": "not_applicable", "reason": "enable_leverage=False", "value": None}
    metrics["leverage"] = {
        "generator": lev_gen,
        "function": safe_call(memory.leverage_function, r_daily, rv_daily),
    }

    # ------------------------------------------------------------------
    # chaos: 決定論的カオス成分 chi_2 (S5)。
    metrics["chaos"] = _chaos_metrics(result, cfg, r_daily)

    # ------------------------------------------------------------------
    # seasonality: 日内季節性とオーバーナイト (S4)。
    # ★S4 の成果物は「季節性を入れたこと」ではなく「除去すれば S1〜S3 の構造が
    # そのまま出てくることを示せる道具」なので、測るのは主に**除去の効き目**。
    metrics["seasonality"] = _seasonality_metrics(result, cfg, r_primary_2d, r_daily)

    # ------------------------------------------------------------------
    trades = result.events.trades()
    signs = trades.side.astype(np.float64) if not trades.is_empty else None
    sizes = trades.size if not trades.is_empty else None
    trade_log_price = result.meta.get("trade_log_price")

    sign_acf_result = safe_call(micro.sign_acf, signs, v.micro_max_lag, v.micro_fit_lag_range)
    propagator_result = safe_call(
        micro.propagator_fit, signs, sizes, trade_log_price, v.micro_max_lag, v.micro_fit_lag_range
    )
    metrics["micro"] = {
        "sign_acf": sign_acf_result,
        "response_function": safe_call(
            micro.response_function, signs, trade_log_price, v.micro_max_lag
        ),
        "propagator": propagator_result,
        "impact_consistency": safe_call(
            micro.impact_consistency,
            sign_acf_result.get("gamma"),
            propagator_result.get("beta"),
            _nested(sign_acf_result, "gamma_fit", "se"),
            _nested(propagator_result, "beta_fit", "se"),
        ),
        "sqrt_law": safe_call(micro.sqrt_law_check, result.meta.get("metaorders")),
        "branching_ratio": safe_call(
            micro.branching_ratio_reestimate,
            result.events,
            None,
            None,
            result.meta.get("branching_ratio_target"),
        ),
    }

    # ------------------------------------------------------------------
    assets = result.meta.get("assets")
    asset1 = obs if assets is None else assets[0]
    asset2 = None if not assets or len(assets) < 2 else assets[1]
    metrics["cross"] = {
        "hayashi_yoshida": safe_call(cross.hayashi_yoshida, asset1, asset2, 0.0),
        "lead_lag": safe_call(
            cross.hayashi_yoshida_lead_lag, asset1, asset2, (-60.0, -10.0, 0.0, 10.0, 60.0)
        ),
    }

    return metrics


def _true_phi_bars_for(
    result: StageResult, cfg: Config, n_bars: int
) -> np.ndarray | None:
    """真の φ_σ をバー粒度で返す (季節性が無ければ ``None``)。

    カレンダーは ``config`` から組み直す (``SeasonalCalendar`` は乱数を消費しない
    ので経路には影響しない)。``StageResult.meta`` に生オブジェクトを入れると
    JSON 化できなくなるため、meta 経由では渡していない。
    """
    if not cfg.enable_seasonality:
        return None
    from ..layers.l0_calendar import build_calendar
    from ..rng import RNGRegistry

    obs = result.observation
    steps_per_day = (
        int(round(obs.session_seconds / obs.step_seconds)) if obs.step_seconds else None
    )
    truth = seasonality.true_phi_bars(
        build_calendar(cfg, RNGRegistry(cfg.seed)), n_bars, steps_per_day=steps_per_day
    )
    return np.asarray(truth["value"], dtype=np.float64) if truth["status"] == "ok" else None


def _deseasonalized_log_price(r_2d: np.ndarray, phi_bars: np.ndarray) -> np.ndarray:
    """φ で割ったリターンから対数価格の行列を組み直す (各セッション 0 始まり)。

    ``variance_ratio`` は価格を入力に取るので、リターンを割ってから積み上げる。
    水準は任意 (差分しか使われない) なので各日 0 から始めてよい。
    """
    d = seasonality.deseasonalize(r_2d, phi_bars)
    return np.concatenate([np.zeros((d.shape[0], 1)), np.cumsum(d, axis=1)], axis=1)


def _chaos_metrics(result: StageResult, cfg: Config, r_daily: np.ndarray) -> dict[str, Any]:
    """S5 の測定群。chi_2 が無効なら全枝が ``not_applicable``。

    3 つの対象に**期待の違う**検定を掛ける (S5 指示書 §9):
    chi_2 単体はカオスの証拠 (critical)、合成 log σ と価格は「検出困難/不能」の
    記録 — 実データから低次元カオスが検出されないという実証と整合するのが正しい。
    """
    if not cfg.enable_chaos_vol:
        reason = "enable_chaos_vol=False"
        return {k: na(reason) for k in (
            "generator", "chi_tests", "spectral", "latent_gph_ablation",
            "dilution", "marginal_log_vol", "composite_tests", "price_tests",
            "no_direction",
        )}

    from ..layers.l2_price import prepare_chaos_component

    l2m = result.meta.get("l2", {})
    gen = l2m.get("chaos")
    sub = l2m.get("vol_subsample")
    out: dict[str, Any] = {
        "generator": (
            {"status": "ok", "value": None, **gen} if gen else na("生成側診断がありません")
        ),
    }

    # --- chi_2 単体のカオス性 ---
    # ★注入窓ではなく**固定長 20,000 単位の参照系列**で測る。Lyapunov・相関次元は
    # 力学系そのもの (パラメータ) の性質で、注入に使う窓の長さに依存しない。
    # 窓で測ると n_days が短い設定で「点数不足」になり、系の性質という不変の事実が
    # 設定依存で NA になってしまう。参照系列はキャッシュされ再計算はほぼ無料。
    # 市場グリッドの補間版で測ってはならない — 分単位グリッドは特徴周期の 500 倍の
    # オーバーサンプリングで、0-1 test が規則側に偏る (連続系の既知の問題)。
    from ..chaos import chaos_generate

    ref = chaos_generate(
        system=cfg.chaos_system,
        params={
            "tau": cfg.chaos_tau_delay, "beta": cfg.chaos_beta,
            "gamma": cfg.chaos_gamma, "n_exponent": cfg.chaos_n_exponent,
        },
        length_units=20000.0,
        dt=cfg.chaos_dt,
        ic=cfg.chaos_ic,
        burn_in_units=cfg.chaos_burn_in_units,
        cache_dir=cfg.chaos_cache_dir or None,
    )
    dt_units = cfg.chaos_dt
    out["chi_tests"] = {
        "reference_length_units": 20000.0,
        "lyapunov": safe_call(chaos_val.lyapunov_rosenstein, ref.x, dt_units),
        "correlation_dimension": safe_call(
            chaos_val.correlation_dimension, ref.x, dt_units
        ),
        # 0-1 test: 特徴周期 (~49.7 単位) の 1/8 に間引く。
        "zero_one": safe_call(
            chaos_val.test_0_1_chaos, ref.x, max(int(round(6.0 / dt_units)), 1)
        ),
    }

    # --- スペクトル (写像の検証)。参照系列を市場時間に写像して測る — 注入窓だと
    # 短い設定で分解能が足りず、分単位サブサンプルだと Welch のセグメント長が
    # ボトルネックで偽ピークが出る (実測 42 日: セグメント長そのもの)。---
    chaos_t_days, chi_norm, _a, _c, _diag = prepare_chaos_component(
        cfg, float(cfg.n_days)
    )
    ref_std = (ref.x - ref.x.mean()) / ref.x.std()
    out["spectral"] = safe_call(
        scaling.spectral_peak, ref_std, _diag["grid_spacing_days"]
    )

    if sub is None:
        out["latent_gph_ablation"] = na("vol_subsample がありません")
        out["dilution"] = na("vol_subsample がありません")
        out["marginal_log_vol"] = na("vol_subsample がありません")
        out["composite_tests"] = na("vol_subsample がありません")
        out["price_tests"] = na("観測が必要です")
        out["no_direction"] = na("vol_subsample がありません")
        return out

    log_vol = np.asarray(sub["log_vol"])
    log_phi = np.asarray(sub["log_phi_sigma"])
    chi_term = np.asarray(sub["chi_term"])
    c_chi = float(sub["c_chi"])
    lv_with = log_vol - log_phi  # 脱季節化した log σ (chi 込み)
    lv_without = lv_with - chi_term + c_chi  # ≡ S4 の log σ (機械精度で厳密)

    # --- 潜在日次 GPH のアブレーション (S5 の ③ 判定 — 2026-08-21 裁定) ---
    # 帯域 0.50 の測定帯は周期 >= 70 日で、設計した 30 日線 (と 62 日の副次調波) の
    # **外側** — ここが動かないことが「長期記憶の構造は不変」の判定。実測の検出力:
    # ピークを 36〜40 日に誤配置すると副次調波が帯に入り -0.03〜-0.05 で落ちる。
    # 帯域 0.65 (周期 >= 20 日) は設計線を**含む**ので、そこの変化 (-0.11) は
    # 汚染ではなく設計の帰結 — 記録として残す。
    t_days = np.asarray(sub["t_days"])
    n_days = int(round(t_days[-1] - t_days[0])) or 1
    per_day = lv_with.shape[0] // n_days
    abl: dict[str, Any] = {"status": "ok", "value": None}
    for bwe, tag in ((0.50, "bw050"), (0.65, "bw065")):
        pair = []
        for lv in (lv_with, lv_without):
            daily = lv[: n_days * per_day].reshape(n_days, per_day).mean(axis=1)
            pair.append(memory.gph_estimator(daily, bandwidth_exponent=bwe))
        abl[f"d_with_chi_{tag}"] = pair[0].get("value")
        abl[f"d_without_chi_{tag}"] = pair[1].get("value")
        abl[f"delta_{tag}"] = (
            pair[0]["value"] - pair[1]["value"]
            if pair[0].get("value") is not None and pair[1].get("value") is not None
            else None
        )
    abl["value"] = abl.get("delta_bw050")
    abl["note"] = (
        "without_chi 系列は同一シードの S4 潜在 log σ と機械精度で一致する"
        " (chi は決定論の加算なので厳密に引ける)"
    )
    out["latent_gph_ablation"] = abl

    # --- レバレッジ希釈の SD 比 (2026-08-21 裁定の計器) ---
    # sqrt(Var_S4/Var_S5) — 希釈式が**厳密に**成り立つ量で、推定ノイズがない。
    # 相関ベースの比は |L| ~ 0.02 (S3 裁定の水準) では SE が信号の 30-40% になり
    # 判定不能 — multiseed が 3 計器の実測スプレッドを記録する。
    v_with = float(lv_with.var())
    v_without = float(lv_without.var())
    out["dilution"] = {
        "status": "ok",
        "value": math.sqrt(v_without / v_with) if v_with > 0 else None,
        "sd_ratio": math.sqrt(v_without / v_with) if v_with > 0 else None,
        "theory": math.sqrt(v_without / (v_without + cfg.vol_var_target_chaos)),
        "theory_nominal": 0.894,
        "var_path_with_chi": v_with,
        "var_path_without_chi": v_without,
    }

    # --- 合成 log σ の周辺分布 (§3.2 のゲート対象は合成後) ---
    out["marginal_log_vol"] = safe_call(scaling.marginal_normality, lv_with)

    # --- 合成 log σ / 価格でのカオス検出 (記録のみ — 検出困難/不能が期待) ---
    stride_days = float(t_days[1] - t_days[0])
    out["composite_tests"] = {
        "lyapunov": safe_call(
            chaos_val.lyapunov_rosenstein, lv_with, stride_days,
        ),
        "correlation_dimension": safe_call(
            chaos_val.correlation_dimension, lv_with, stride_days
        ),
        "zero_one": safe_call(
            chaos_val.test_0_1_chaos, lv_with, max(int(round(4.0 / stride_days)), 1)
        ),
        "note": "検出困難が期待値 (確率成分 4:1 に埋もれる)。カオスの価値は識別可能性ではない",
    }
    out["price_tests"] = {
        "bds_daily_returns": safe_call(chaos_val.bds_test, r_daily),
        "zero_one_daily_abs": safe_call(chaos_val.test_0_1_chaos, np.abs(r_daily), 1),
        "note": (
            "BDS の棄却は確率ボラだけで説明でき、カオスの証拠ではない。"
            "検出不能が実証 (実データから低次元カオスは検出されない) と整合"
        ),
    }

    # --- 帰無対照: chi は方向を持たない (§15 の第一禁止事項の検証) ---
    # chi_2 は σ にのみ入るので、リターンの**方向**とは無相関のはず。
    chi_daily = chi_term[: n_days * per_day].reshape(n_days, per_day).mean(axis=1)
    nd = min(r_daily.shape[0], chi_daily.shape[0])
    if nd > 30 and np.std(chi_daily[:nd]) > 0:
        c = float(np.corrcoef(r_daily[:nd], chi_daily[:nd])[0, 1])
        out["no_direction"] = {
            "status": "ok",
            "value": c,
            "corr_r_chi": c,
            "se": 1.0 / math.sqrt(nd),
            "abs_z": abs(c) * math.sqrt(nd),
            "n": nd,
        }
    else:
        out["no_direction"] = na("日数が足りません")
    return out


def _seasonality_metrics(
    result: StageResult, cfg: Config, r_primary_2d: np.ndarray, r_daily: np.ndarray
) -> dict[str, Any]:
    """S4 の測定群。季節性・ON が無効なら全枝が ``not_applicable`` になる。

    カレンダーは ``config`` から組み直す。``StageResult.meta`` に生オブジェクトを
    入れると JSON 化できなくなるので入れていない。``SeasonalCalendar`` は乱数を
    消費しないため、組み直しても経路には一切影響しない。
    """
    from ..layers.l0_calendar import build_calendar
    from ..rng import RNGRegistry

    calendar = build_calendar(cfg, RNGRegistry(cfg.seed))
    l2m = result.meta.get("l2", {})
    obs = result.observation
    steps_per_day = (
        int(round(obs.session_seconds / obs.step_seconds)) if obs.step_seconds else None
    )

    out: dict[str, Any] = {
        "enabled": {"seasonality": cfg.enable_seasonality, "overnight": cfg.enable_overnight},
        "calendar": (
            {"status": "ok", "value": None, **calendar.diagnostics()}
            if cfg.enable_seasonality or cfg.enable_overnight
            else na("季節性・ON とも無効です")
        ),
        "phi_normalization": safe_call(seasonality.phi_normalization_check, calendar),
    }

    # --- 脱季節化 (真値経路・推定経路) ---
    if cfg.enable_seasonality:
        out["deseasonalization"] = safe_call(
            seasonality.deseasonalization_report,
            r_primary_2d,
            calendar,
            3,
            "median_abs",
            steps_per_day,
        )
        # ★README が要求する数値: 季節性が長期記憶の推定を汚す量と、除去で戻る量。
        out["gph_abs_r"] = safe_call(
            _gph_deseasonalized, r_primary_2d, calendar, cfg, steps_per_day
        )
    else:
        reason = "enable_seasonality=False"
        out["deseasonalization"] = na(reason)
        out["gph_abs_r"] = na(reason)

    # --- オーバーナイト ---
    gaps = result.price.overnight_gaps
    if cfg.enable_overnight and gaps.size:
        sigma_close = None
        if steps_per_day:
            n_days = int(round((obs.t[-1] - obs.t[0]) / obs.session_seconds))
            close_idx = np.arange(1, n_days) * steps_per_day - 1
            if close_idx.size == gaps.size:
                sigma_close = np.exp(result.price.log_vol[close_idx])
        out["overnight"] = safe_call(seasonality.overnight_stats, gaps, r_daily, sigma_close)
        out["overnight_generator"] = (
            {"status": "ok", "value": None, **l2m["overnight"]}
            if l2m.get("overnight")
            else na("生成側診断がありません")
        )
    else:
        out["overnight"] = na("enable_overnight=False")
        out["overnight_generator"] = na("enable_overnight=False")

    # --- ジャンプ強度の S4 補正 (QV 予算が S3 から動いていないかの根拠) ---
    jump = l2m.get("jump")
    out["jump_intensity_scale"] = (
        {
            "status": "ok",
            "value": jump.get("intensity_scale_s4"),
            "cap_binding_fraction": jump.get("cap_binding_fraction"),
            "jv_share_theory": jump.get("jv_share_theory"),
            "lambda_effective_per_year": jump.get("lambda_effective_per_year"),
        }
        if jump
        else na("enable_jump=False")
    )
    return out


def _gph_deseasonalized(
    r_2d: np.ndarray, calendar: Any, cfg: Config, steps_per_day: int | None
) -> dict[str, Any]:
    """|r| の GPH d を raw / 真値 φ 除去 / 推定 φ̂ 除去の 3 通りで測る。

    季節性は ``|r|`` のスペクトルの高調波に力を足すので、GPH の回帰に低周波側から
    漏れて ``d`` を**上方に**偏らせる。除去でどれだけ戻るかがここの主題。
    差は GPH の漸近標準誤差 ``pi/sqrt(24m)`` と比べて読むこと — 単独の経路では
    1 標準誤差前後の差は判定できない (複数シードで見る)。
    """
    bwe = cfg.validation.gph_bandwidth_exponent
    truth = seasonality.true_phi_bars(calendar, r_2d.shape[1], steps_per_day=steps_per_day)
    est = seasonality.estimate_phi(r_2d)

    series: dict[str, np.ndarray] = {"raw": r_2d}
    if truth["status"] == "ok":
        series["true_phi_removed"] = seasonality.deseasonalize(
            r_2d, np.asarray(truth["value"])
        )
    if est["status"] == "ok":
        series["est_phi_removed"] = seasonality.deseasonalize(r_2d, np.asarray(est["value"]))

    fits = {
        name: memory.gph_estimator(np.abs(arr).ravel(), bandwidth_exponent=bwe)
        for name, arr in series.items()
    }
    d_raw = fits["raw"].get("value")
    d_true = fits.get("true_phi_removed", {}).get("value")
    se = fits["raw"].get("se_asymptotic")
    return {
        "status": "ok",
        "value": d_raw,
        "d_raw": d_raw,
        "d_true_phi_removed": d_true,
        "d_est_phi_removed": fits.get("est_phi_removed", {}).get("value"),
        # README 記載必須: 季節性による d の汚染量。
        "d_raw_minus_true_phi": (
            d_raw - d_true if d_raw is not None and d_true is not None else None
        ),
        "se_asymptotic": se,
        "bias_in_se_units": (
            (d_raw - d_true) / se
            if d_raw is not None and d_true is not None and se
            else None
        ),
        "fits": fits,
    }


def _nested(result: Mapping[str, Any], key: str, subkey: str) -> Any:
    node = result.get(key)
    if isinstance(node, Mapping):
        return node.get(subkey)
    return None


def collect_errors(metrics: Mapping[str, Any], prefix: str = "") -> list[dict[str, str]]:
    """指標ツリーから ``status="error"`` のノードを集める。"""
    found: list[dict[str, str]] = []
    for key, value in metrics.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            if value.get("status") == "error":
                found.append({"path": path, "error": str(value.get("error"))})
            else:
                found.extend(collect_errors(value, prefix=f"{path}."))
    return found


def flatten(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """指標ツリーをドット記法のスカラー辞書へ平坦化する (``compare`` 用)。

    リストや長い配列は畳んで長さだけにする。段階間で見たいのは代表値であって
    ACF の全ラグではないため。
    """
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, prefix=f"{path}."))
        elif isinstance(value, (list, tuple)):
            out[f"{path}.__len__"] = len(value)
        else:
            out[path] = value
    return out
