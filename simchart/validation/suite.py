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

from scipy import stats as _sp_stats

from ..config import TRADING_DAYS_PER_YEAR, Config
from ..types import StageResult
from . import chaos as chaos_val
from . import cross, ensemble, memory, micro, scaling, seasonality, tails
from .base import na, num, ok, safe_call

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
    # ★S6 (κ=0 の板): 観測は ZI ミッドで φ の季節性を**持たない**。φ で割ると
    # 存在しないパターンの逆数が乗り、逆向きの歪みを作ってしまう。脱季節化は
    # 「観測が L2 由来」のときだけ (板有効かつ κ=0 なら生のまま測る)。
    obs_carries_phi = phi_bars is not None and not (cfg.enable_book and cfg.kappa == 0.0)
    vr_series = (
        _deseasonalized_log_price(r_primary_2d, phi_bars)
        if obs_carries_phi
        else primary_bars.log_price
    )
    metrics["scaling"] = {
        "variance_ratio": safe_call(scaling.variance_ratio, vr_series, v.vr_qs),
        "variance_ratio_deseasonalized": bool(obs_carries_phi),
        "variance_ratio_raw": (
            safe_call(scaling.variance_ratio, primary_bars.log_price, v.vr_qs)
            if obs_carries_phi
            else {"status": "not_applicable",
                  "reason": "観測が季節性を持たない (variance_ratio と同一)",
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
    # book: ZI 板 (S6)。エンジン正当性・板の性質・S7/S8/S10 のベースライン。
    metrics["book"] = _book_metrics(result, cfg)

    # ------------------------------------------------------------------
    # hawkes: 符号対称 Hawkes 注文流 (S7)。分岐比 3 経路・残差検定・過分散。
    metrics["hawkes"] = _hawkes_metrics(result, cfg)

    # ------------------------------------------------------------------
    # meta: メタオーダー分割 (S8)。γ・プール・propagator・インパクト赤字。
    metrics["meta"] = _meta_metrics(result, cfg)

    # ------------------------------------------------------------------
    # qr: queue-reactive (S9)。η・OBI・戻り曲線・状態依存の実測。
    metrics["qr"] = _qr_metrics(result, cfg)

    # ------------------------------------------------------------------
    # coupling: κ 結合 + c_vol (S10)。乖離 d・伝達率・残差 γ・追随・⑦。
    metrics["coupling"] = _coupling_metrics(result, cfg)

    # ------------------------------------------------------------------
    # feedback: RV フィードバックと内生的危機 (S11)。ペア量 (g・発散) は
    # multiseed 側 (off 対をシードごとに回す)。ここは単独ランで測れる分。
    metrics["feedback"] = _feedback_metrics(result, cfg)
    if cfg.enable_feedback:
        # §8.3: n̂ (定数カーネル MLE、φ·Z 補償) ≈ E[n_t] の照合 (時変 n の平均)
        n_hat = ((metrics.get("hawkes") or {}).get("three_way") or {}).get(
            "n_hat_true_phi"
        )
        nt_mean = ((result.meta.get("l3") or {}).get("feedback") or {}).get("nt_mean")
        metrics["feedback"]["n_hat_vs_nt_mean"] = {
            "status": "ok" if (n_hat is not None and nt_mean is not None) else "not_applicable",
            "value": (
                float(n_hat) - float(nt_mean)
                if (n_hat is not None and nt_mean is not None) else None
            ),
            "n_hat": n_hat,
            "nt_mean": nt_mean,
        }

    # ------------------------------------------------------------------
    # chaos: 決定論的カオス成分 chi_2 (S5)。
    metrics["chaos"] = _chaos_metrics(result, cfg, r_daily)

    # ------------------------------------------------------------------
    # chaos_l1: χ₁ (活動度) / χ₃ (脆弱窓) — S12。
    metrics["chaos_l1"] = _chi_l1_metrics(result, cfg)

    # ------------------------------------------------------------------
    # seasonality: 日内季節性とオーバーナイト (S4)。
    # ★S4 の成果物は「季節性を入れたこと」ではなく「除去すれば S1〜S3 の構造が
    # そのまま出てくることを示せる道具」なので、測るのは主に**除去の効き目**。
    metrics["seasonality"] = _seasonality_metrics(result, cfg, r_primary_2d, r_daily)

    # ------------------------------------------------------------------
    # S6+: 符号 ACF と propagator は**攻撃注文単位**の系列で測る。TRADE 行のままだと
    # 複数レベルを掃いた成行が同符号の行を連続させ、機械的な正の自己相関 (+0.38 を
    # 実測) が出る — 記録粒度の人工物であって注文流の性質ではない。
    ev_meta = result.events.meta
    if isinstance(ev_meta, dict) and "agg_trade_side" in ev_meta and np.asarray(
        ev_meta["agg_trade_side"]
    ).size:
        signs = np.asarray(ev_meta["agg_trade_side"], dtype=np.float64)
        sizes = np.asarray(ev_meta["agg_trade_size"], dtype=np.float64)
        trade_log_price = np.asarray(ev_meta["agg_trade_log_vwap"], dtype=np.float64)
    else:
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


def _book_metrics(result: StageResult, cfg: Config) -> dict[str, Any]:
    """S6 の測定群。板が無効なら全枝 ``not_applicable``。

    ★この段階の観測価格 (ZI ミッド) に L2 の性質は現れない (κ=0)。tails/memory 等の
    既存の枝が測る値は「純マイクロ構造ベースライン」であり、S10 で結合したときに
    L2 の水準まで戻るかの比較対象になる (指示書 §11)。
    """
    from . import engine as engine_val

    if not cfg.enable_book:
        reason = "enable_book=False"
        return {k: na(reason) for k in (
            "engine_invariants", "throughput", "spread", "depth", "queue",
            "order_size", "placement", "liveness", "interevent", "obi",
            "corr_mid_pstar", "trade_price", "mid_vs_trade_signature",
        )}

    l3 = result.meta.get("l3", {})
    ev = result.events
    meta = ev.meta if isinstance(ev.meta, dict) else {}
    burn_sec = cfg.book_burn_in_days * result.observation.session_seconds
    horizon = cfg.n_days * result.observation.session_seconds

    out: dict[str, Any] = {
        "engine_invariants": safe_call(engine_val.engine_invariants, l3, ev.t),
        "throughput": safe_call(engine_val.throughput, l3),
    }

    bb = np.asarray(meta.get("best_bid_tick", np.empty(0)))
    ba = np.asarray(meta.get("best_ask_tick", np.empty(0)))
    out["spread"] = safe_call(micro.spread_distribution, bb, ba, ev.t, burn_sec)
    out["depth"] = safe_call(
        micro.depth_profile, result.book, burn_sec, cfg.tick_size
    )
    out["queue"] = safe_call(micro.queue_length_distribution, result.book, burn_sec)
    out["obi"] = safe_call(micro.obi, result.book, 5, burn_sec)
    out["liveness"] = safe_call(
        engine_liveness_from_meta, l3, horizon
    )

    # 発注 (LO/MO) のサイズ仕様適合。burn-in 後のみ。
    from ..types import EventType

    is_order = (ev.event_type == int(EventType.LIMIT_ADD)) | (
        ev.event_type == int(EventType.MARKET)
    )
    order_mask = is_order & (ev.t >= burn_sec)
    out["order_size"] = safe_call(
        micro.order_size_check, ev.size[order_mask],
        cfg.book_lot_values, cfg.book_lot_probs,
        cfg.book_size_round_weight, cfg.book_size_pareto_alpha,
    )

    # 配置分布: 発注直前の best を復元して渡す (自分の improvement の影響を除く)。
    lo_mask = (ev.event_type == int(EventType.LIMIT_ADD)) & (ev.t >= burn_sec)
    if bb.size:
        bb_prev = np.concatenate([[np.nan], bb[:-1].astype(np.float64)])
        ba_prev = np.concatenate([[np.nan], ba[:-1].astype(np.float64)])
        base_price = float(meta.get("base_price", 0.0))
        tick = float(meta.get("tick_size", cfg.tick_size))
        lo_ticks = np.round((ev.price[lo_mask] - base_price) / tick)
        out["placement"] = safe_call(
            micro.placement_check, lo_ticks, ev.side[lo_mask],
            bb_prev[lo_mask], ba_prev[lo_mask],
            cfg.book_mu_place, cfg.book_place_offset, cfg.book_max_place_ticks,
        )
    else:
        out["placement"] = na("best 系列がありません")

    # 到着間隔 (S7 ベースライン): 定数レートの到着 (MO+LO) で測る。取消は
    # レートが N(t) 比例なので Poisson でなく、混ぜると意味が濁る。
    arr_mask = is_order & (ev.t >= burn_sec)
    out["interevent"] = safe_call(micro.interevent_times, ev.t[arr_mask], 0.0)
    out["interevent_all_types"] = safe_call(
        micro.interevent_times, ev.t[ev.t >= burn_sec], 0.0
    )

    # 符号 ACF (S8 ベースライン): 攻撃注文単位の符号で全ラグを見る。
    # ★閾値は 2/√N ではなく Bonferroni 補正の 3.7/√N (指示書の字義 2/√N は
    # 200 ラグの最大値に対して iid でもほぼ確実に破れる — S0 の ±2σ、S3 の
    # z_no_autocorr と同型の問題で、同じ解決を適用する)。
    agg_side = meta.get("agg_trade_side")
    if agg_side is not None and np.asarray(agg_side).size > 5000:
        s_arr = np.asarray(agg_side, dtype=np.float64)
        agg_t = np.asarray(meta.get("agg_trade_t"))
        s_arr = s_arr[agg_t >= burn_sec]
        d = s_arr - s_arr.mean()
        denom = float(d @ d)
        nlags = 200
        max_abs = 0.0
        argmax = 0
        for k_ in range(1, nlags + 1):
            r_ = abs(float(d[:-k_] @ d[k_:]) / denom)
            if r_ > max_abs:
                max_abs = r_
                argmax = k_
        out["sign_acf_zero"] = {
            "status": "ok",
            "value": max_abs * math.sqrt(s_arr.size),
            "max_abs_acf": max_abs,
            "max_abs_z": max_abs * math.sqrt(s_arr.size),
            "at_lag": argmax,
            "threshold_bonferroni": 3.7,
            "n": int(s_arr.size),
            "n_lags": nlags,
        }
    else:
        out["sign_acf_zero"] = na("約定が足りません")

    # κ=0 の確認 (S10 ベースライン): **リターンの相関**で測る。
    # ★水準 (ミッドと p* そのもの) の相関を使ってはならない — 独立なランダム
    # ウォーク同士の標本相関は 0 に集中しない (arcsine 分布) ため、コイン投げの
    # ゲートになる (S5 の価格シード横断相関と同じ教訓)。
    lp_star = meta.get("log_pstar")
    if lp_star is not None and bb.size:
        okm = (bb >= 0) & (ba >= 0) & (ev.t >= burn_sec)
        mid_t = ev.t[okm]
        mid_v = 0.5 * (bb[okm] + ba[okm]).astype(np.float64)
        ps_v = np.asarray(lp_star)[okm]
        # 1 分ごとに間引いてリターン化
        stride = max(int(round(60.0 / max(float(np.median(np.diff(mid_t[:1000]))), 1e-9))), 1)
        dm = np.diff(mid_v[::stride])
        dp = np.diff(ps_v[::stride])
        good = (dm != 0) | (dp != 0)
        if good.sum() > 100:
            c = float(np.corrcoef(dm[good], dp[good])[0, 1])
            out["corr_mid_pstar"] = {
                "status": "ok", "value": c, "corr_returns": c,
                "se": 1.0 / math.sqrt(good.sum()), "n": int(good.sum()),
                "abs_z": abs(c) * math.sqrt(good.sum()),
            }
        else:
            out["corr_mid_pstar"] = na("リターン標本が足りません")
    else:
        out["corr_mid_pstar"] = na("p* の配線記録がありません")

    # ミッドの分散比 — **日次スケール**で測る (S10 ベースライン)。
    # ★分単位スケール (60s バー、q<=64 分) の VR は ZI 板では**強い平均回帰**を
    # 示す (実測 VR(64min)=0.19)。これはバグではなく ZI の既知の物理: 板が
    # バネとして働き、注文の平均寿命 1/δ (=0.2 日 ≈ 94 分) より短い時間層では
    # subdiffusive になる (Smith et al. 2003)。「長スケールで拡散的」の判定は
    # クロスオーバーより上の日次バー (q=2..64 日) で行い、分単位は記録する。
    obs = result.observation
    daily_bars = obs.to_bars(obs.session_seconds)
    lp_daily = daily_bars.log_price_flat()
    n_burn_days = int(cfg.book_burn_in_days)
    out["mid_vr_daily"] = safe_call(
        scaling.variance_ratio, lp_daily[n_burn_days:], (2, 4, 8, 16, 32, 64)
    )

    # 約定価格系列 (bid-ask bounce): ACF(1) < 0 と signature plot の非平坦 (soft)。
    agg_px = meta.get("agg_trade_log_vwap")
    if agg_px is not None and np.asarray(agg_px).size > 1000:
        r_tr = np.diff(np.asarray(agg_px))
        d = r_tr - r_tr.mean()
        acf1 = float(d[:-1] @ d[1:] / (d @ d))
        out["trade_price"] = {
            "status": "ok", "value": acf1, "acf1": acf1,
            "se": 1.0 / math.sqrt(r_tr.size), "n": int(r_tr.size),
        }
    else:
        out["trade_price"] = na("約定が足りません")

    # signature plot: 約定価格 (bounce で短スケール上振れ) vs ミッド (ほぼ平坦)。
    # 既存の scaling.signature_plot は観測 (ミッド) に対して走る — ここでは
    # 両者の対比のため約定側の簡易版 (1/5/30 分の実現分散比) を記録する。
    if agg_px is not None and np.asarray(agg_px).size > 5000:
        agg_t = np.asarray(meta.get("agg_trade_t"))
        px_arr = np.asarray(agg_px)
        rows = {}
        for sec in (60.0, 300.0, 1800.0):
            grid = np.arange(burn_sec, horizon, sec)
            idx = np.searchsorted(agg_t, grid, side="right") - 1
            valid = idx >= 0
            series = px_arr[idx[valid]]
            rr = np.diff(series)
            rows[f"var_per_sec_{int(sec)}"] = float(rr.var() / sec) if rr.size > 100 else None
        v1, v30 = rows.get("var_per_sec_60"), rows.get("var_per_sec_1800")
        rows["ratio_60_over_1800"] = (v1 / v30) if (v1 and v30) else None
        out["mid_vs_trade_signature"] = {"status": "ok", "value": rows.get("ratio_60_over_1800"), **rows}
    else:
        out["mid_vs_trade_signature"] = na("約定が足りません")

    return out


def _hawkes_metrics(result: StageResult, cfg: Config) -> dict[str, Any]:
    """S7 の測定群。Hawkes が無効なら全枝 ``not_applicable``。

    中心は分岐比の 3 経路再推定 (raw / true-φ / est-φ̂) — S4 の脱季節化機構が
    無いと n̂ が系統的に過大になる (Filimonov–Sornette) ことの実証と、
    その対策が効いていることの確認を同時に行う。
    """
    keys = (
        "three_way", "rescaling", "overdispersion", "intraday_shape",
        "volume_acf", "guards", "realized_rates", "fano_reestimate",
    )
    if not cfg.enable_hawkes:
        return {k: na("enable_hawkes=False") for k in keys}

    from . import hawkes as hk

    times, marks = hk.marks_from_eventlog(result.events)
    session = float(result.observation.session_seconds)
    t_end = cfg.n_days * session
    burn_sec = cfg.book_burn_in_days * session
    a_mat = np.asarray(cfg.hawkes_a, dtype=np.float64)
    betas = 1.0 / np.asarray(cfg.hawkes_tau_seconds, dtype=np.float64)  # [1/秒]
    w = np.asarray(cfg.hawkes_weights, dtype=np.float64)
    n_design = float(np.max(np.abs(np.linalg.eigvals(a_mat))))

    # 真の φ_λ テーブル (エンジンが消費するのと同じ 4096 格子)
    true_phi: np.ndarray | None = None
    if cfg.enable_seasonality:
        from ..layers.l0_calendar import build_calendar
        from ..rng import RNGRegistry

        cal = build_calendar(cfg, RNGRegistry(cfg.seed))
        m_phi = 4096
        u_grid = (np.arange(m_phi, dtype=np.float64) + 0.5) / m_phi
        true_phi = np.asarray(cal.phi_lambda_of_u(u_grid), dtype=np.float64)

    out: dict[str, Any] = {}

    # S10c: c_vol>0 では真のベースラインは φ·Z — Z を補償しない MLE は Z の
    # クラスタリングを励起へ誤帰属し n̂ が上振れする (+0.063 実測)。
    # Z はエンジンが公開したものを使う (再導出は単一情報源の原則に反する)。
    ev_meta_h = result.events.meta if isinstance(result.events.meta, dict) else {}
    z_grid_h = ev_meta_h.get("cvol_z_grid")
    z_step_h = ev_meta_h.get("cvol_z_step_sec")
    zkw_h: dict[str, Any] = {}
    if z_grid_h is not None and z_step_h is not None:
        zkw_h = {"z_grid": np.asarray(z_grid_h, dtype=np.float64),
                 "z_step_sec": float(z_step_h)}

    # --- 分岐比の 3 経路 (中心ゲート) ---
    block_days = 50.0 if cfg.n_days >= 200 else max(10.0, cfg.n_days / 4.0)
    out["three_way"] = safe_call(
        hk.branching_three_ways, times, marks, t_end, betas, w, session,
        true_phi, n_design, block_days=block_days, **zkw_h,
    )

    # --- 残差検定 (真の φ (·Z) を与えた当てはめモデルで時間再スケーリング) ---
    def _rescaling() -> dict[str, Any]:
        fit = hk.hawkes_mle(
            times, marks, t_end, betas, w,
            phi_table=true_phi, session_seconds=session if true_phi is not None else None,
            **zkw_h,
        )
        res = hk.time_rescaling_test(
            times, marks, t_end, betas, w, fit["mu_hat_per_sec"], fit["a_hat"],
            phi_table=true_phi, session_seconds=session if true_phi is not None else None,
            **zkw_h,
        )
        res["fit_converged"] = bool(fit["converged"])
        return res

    out["rescaling"] = safe_call(_rescaling)

    # --- 過分散 (バーンイン後)。Fano は複数窓、間隔は CV² と KS ---
    def _overdispersion() -> dict[str, Any]:
        t_b = times[times >= burn_sec]
        rows: dict[str, Any] = {}
        for win in (60.0, 300.0, 1800.0):
            edges = np.arange(burn_sec, t_end + win, win)
            c, _ = np.histogram(t_b, bins=edges)
            rows[f"fano_{int(win)}s"] = num(c.var() / c.mean()) if c.mean() > 0 else None
        d = np.diff(t_b)
        d = d[d > 0]
        cv2 = float(d.var() / d.mean() ** 2)
        ks_stat, ks_p = _sp_stats.kstest(d / d.mean(), "expon")
        rows.update({
            "interevent_cv2": num(cv2),
            "ks_stat_vs_exponential": num(ks_stat),
            # p は指数分布の**棄却**を期待する側 (小さいほど良い)
            "ks_pvalue_vs_exponential": num(ks_p),
            "n_events": int(t_b.size),
        })
        return ok(rows["fano_60s"], **rows)

    out["overdispersion"] = safe_call(_overdispersion)

    # --- 日内 U 字がベースラインに乗っているか (φ_λ との相関) ---
    def _intraday() -> dict[str, Any]:
        if true_phi is None:
            return na("enable_seasonality=False (φ_λ ≡ 1)")
        t_b = times[times >= burn_sec]
        u = np.mod(t_b / session, 1.0)
        n_bins = 26
        counts, edges = np.histogram(u, bins=np.linspace(0.0, 1.0, n_bins + 1))
        centers = ((edges[:-1] + edges[1:]) / 2.0 * true_phi.size).astype(int)
        phi_c = true_phi[np.minimum(centers, true_phi.size - 1)]
        c = float(np.corrcoef(counts, phi_c)[0, 1])
        return ok(num(c), corr=num(c), n_bins=n_bins,
                  counts_ratio_max_min=num(counts.max() / max(counts.min(), 1)))

    out["intraday_shape"] = safe_call(_intraday)

    # --- 出来高の分単位 ACF (活動度クラスタリング → 出来高クラスタリング) ---
    def _volume_acf() -> dict[str, Any]:
        meta = result.events.meta if isinstance(result.events.meta, dict) else {}
        agg_t = np.asarray(meta.get("agg_trade_t", np.empty(0)), dtype=np.float64)
        agg_sz = np.asarray(meta.get("agg_trade_size", np.empty(0)), dtype=np.float64)
        keep = agg_t >= burn_sec
        if int(keep.sum()) < 5000:
            return na("約定が足りません")
        edges = np.arange(burn_sec, t_end + 60.0, 60.0)
        vol, _ = np.histogram(agg_t[keep], bins=edges, weights=agg_sz[keep])
        d = vol - vol.mean()
        denom = float(d @ d)
        lags = {}
        for k in (1, 2, 3, 5, 10, 30):
            lags[f"lag{k}"] = num(float(d[:-k] @ d[k:]) / denom)
        se = 1.0 / math.sqrt(vol.size)
        return ok(lags["lag1"], **lags, se=num(se), n_minutes=int(vol.size),
                  z_lag1=num(lags["lag1"] / se if lags["lag1"] is not None else None))

    out["volume_acf"] = safe_call(_volume_acf)

    # --- ガード発動 (§5.3) と受理率 ---
    def _guards() -> dict[str, Any]:
        h = (result.meta.get("l3") or {}).get("hawkes") or {}
        cand = float(h.get("candidates") or 0)
        cap_rate = (h.get("cap_hits", 0) / cand) if cand > 0 else None
        return ok(
            num(cap_rate),
            cap_hits=int(h.get("cap_hits", -1)),
            cap_hit_rate=num(cap_rate),
            daycap_hits=int(h.get("daycap_hits", -1)),
            cx_noop=int(h.get("cx_noop", -1)),
            acceptance_rate=num(h.get("acceptance_rate")),
            candidates=int(cand),
        )

    out["guards"] = safe_call(_guards)

    # --- 実現レート vs 定常目標 ---
    def _rates() -> dict[str, Any]:
        from ..layers.l1_activity import HawkesActivity

        targets = HawkesActivity(cfg, None).stationary_rates()
        keep = times >= burn_sec
        days = cfg.n_days - cfg.book_burn_in_days
        # S10c: c_vol>0 では設計上レート ∝ Z なので、閉ループ確認は実現平均 Z で
        # 正規化した値で行う (生の ±5% は Z≡1 を前提としており、エポック効果で
        # 実行ごとに ±10〜15% 動くのが正しい物理 — results/S10c/DECISION.md)。
        z_norm = 1.0
        if float(cfg.c_vol) > 0.0:
            cv = (result.meta.get("l3") or {}).get("cvol") or {}
            zm = cv.get("z_mean")
            if zm:
                z_norm = float(zm)
        # S12: n_t が広く振れる (χ₃、n_max=0.95) と 1/(1−n) の凸性で平均レートが
        # 設計アンカーの ~2 倍になる — これは脆弱窓の設計帰結なので、閉ループ確認は
        # **n_t 込みの予測** rate ∝ E[(1−n_design)/(1−n_t)]⁻¹ に対して行う。
        n_factor = 1.0
        if float(cfg.chi3_b) > 0.0 or float(cfg.fb_b_n) > 0.0:
            try:
                from ..cli import _nt_series

                nt = _nt_series(result, cfg)
                if nt is not None and nt.size:
                    n_design_ = float(
                        np.max(np.abs(np.linalg.eigvals(
                            np.asarray(cfg.hawkes_a, dtype=np.float64))))
                    )
                    amp = 1.0 / (1.0 - nt)
                    # ★z との結合平均 E[z/(1−n_t)] を使う — 積の平均 ≠ 平均の積:
                    # 高ボラ期 (z 高) ⟺ u 高 ⟺ n 高の正相関が +31% の共分散を
                    # 持つ (事前測定 #3 で全タイプ均一な残差として実測)。
                    ev_meta_r = result.events.meta if isinstance(
                        result.events.meta, dict) else {}
                    zg_r = ev_meta_r.get("cvol_z_grid")
                    if zg_r is not None:
                        z_arr = np.asarray(zg_r, dtype=np.float64)
                        zs_r = float(ev_meta_r.get("cvol_z_step_sec", 60.0))
                        us_r = float(ev_meta_r.get("fb_u_step_sec", 60.0))
                        start_r = int(cfg.book_burn_in_days * 23400.0 / us_r)
                        idx_r = np.minimum(
                            ((start_r + np.arange(nt.size)) * us_r / zs_r).astype(np.int64),
                            z_arr.size - 1,
                        )
                        joint = float(np.mean(z_arr[idx_r] * amp) * (1.0 - n_design_))
                        n_factor = joint / z_norm if z_norm > 0 else joint
                    else:
                        n_factor = float(amp.mean() * (1.0 - n_design_))
            except Exception:
                n_factor = 1.0
        rows: dict[str, Any] = {}
        rels = []
        raw_rels = []
        for y, name in ((0, "mo"), (1, "lo"), (2, "cx")):
            rate = float(((marks == y) & keep).sum()) / days
            rel_raw = rate / float(targets[y]) - 1.0
            rel = rate / (float(targets[y]) * z_norm * n_factor) - 1.0
            rows[f"{name}_per_day"] = num(rate)
            rows[f"{name}_target"] = num(targets[y])
            rows[f"{name}_rel_diff"] = num(rel)
            rows[f"{name}_rel_diff_raw"] = num(rel_raw)
            rels.append(abs(rel))
            raw_rels.append(abs(rel_raw))
        rows["max_abs_rel_diff"] = num(max(rels))
        rows["max_abs_rel_diff_raw"] = num(max(raw_rels))
        rows["z_mean_norm"] = num(z_norm)
        rows["n_factor_norm"] = num(n_factor)
        return ok(rows["max_abs_rel_diff"], **rows)

    out["realized_rates"] = safe_call(_rates)

    # --- Fano 法の n̂ (カーネル形状フリーの相互参照 — 記録のみ) ---
    # ★φ の U 字も Fano を膨らませる (raw 経路と同じ罠) ので、ゲートには使わず
    # S8〜S11 での経年比較の記録として残す。
    out["fano_reestimate"] = safe_call(
        micro.branching_ratio_reestimate, times[times >= burn_sec], target=n_design
    )

    return out


def _meta_metrics(result: StageResult, cfg: Config) -> dict[str, Any]:
    """S8 の測定群。メタオーダーが無効なら全枝 ``not_applicable``。

    中心は 3 つ:
    1. ⑪ 符号 ACF の γ (対数ビン回帰 — 生ラグ点は whale 支配で暴れる) と C(1)
    2. propagator の**実測** (課さない、測る — §7.1)。イベント時間 (約定
       インデックス)、系列は各攻撃注文**直前**のミッド (VWAP は bounce が乗る)
    3. インパクト赤字の 4 値 (§8.3 — S9/S10 の到達目標として記録)
    """
    keys = (
        "sign_acf_gamma", "length_fit", "pool", "flow_balance", "iceberg",
        "response_mid", "propagator_mid", "propagator_stability",
        "sqrt_law", "impact_vs_size", "impact_deficit",
    )
    if not cfg.enable_metaorder:
        return {k: na("enable_metaorder=False") for k in keys}

    ev = result.events
    meta = ev.meta if isinstance(ev.meta, dict) else {}
    obs = result.observation
    session = float(obs.session_seconds)
    burn_sec = cfg.book_burn_in_days * session
    out: dict[str, Any] = {}

    s_all = np.asarray(meta.get("agg_trade_side", np.empty(0)), dtype=np.float64)
    t_all = np.asarray(meta.get("agg_trade_t", np.empty(0)), dtype=np.float64)
    keep = t_all >= burn_sec
    s = s_all[keep]

    # --- ⑪ γ と C(1)。γ の量的判定はこの対数ビン推定を正とする ---
    gamma_theory = float(cfg.meta_alpha) - 1.0

    def _gamma() -> dict[str, Any]:
        if s.size < 20_000:
            return na(f"攻撃注文が足りません (n={s.size})")
        fit = memory.acf_powerlaw_fit(s, (2, 1000), max_lag=1000)
        if fit["status"] != "ok":
            return fit
        d = s - s.mean()
        c1 = float(d[:-1] @ d[1:]) / float(d @ d)
        return ok(
            num(fit["gamma"]),
            gamma=num(fit["gamma"]),
            gamma_theory=num(gamma_theory),
            gamma_minus_theory=num(fit["gamma"] - gamma_theory),
            r2=num(fit.get("r2")),
            c1=num(c1),
            n=int(s.size),
            fit_lag_range=[2, 1000],
        )

    out["sign_acf_gamma"] = safe_call(_gamma)

    # --- 長さ分布・プール・フロー ---
    mo_rec = meta.get("metaorders") or {}
    out["length_fit"] = safe_call(
        micro.metaorder_length_check,
        mo_rec.get("n_total", np.empty(0)), float(cfg.meta_alpha), int(cfg.meta_n_min),
    )
    steps_per_day = int(round(session / obs.step_seconds)) if obs.step_seconds else 0
    out["pool"] = safe_call(
        micro.pool_stationarity,
        meta.get("pool_grid", np.empty(0)),
        int(cfg.book_burn_in_days * steps_per_day),
    )

    def _flow() -> dict[str, Any]:
        d = (result.meta.get("l3") or {}).get("meta") or {}
        children = float(d.get("children", 0))
        noise = float(d.get("noise_trades", 0))
        total = children + noise
        if total <= 0:
            return na("成行イベントがありません")
        child_frac = children / total
        ratio = child_frac / float(cfg.meta_psi)
        arrivals = float(d.get("arrivals", 0))
        spawns = float(d.get("spawned_on_empty", 0))
        return ok(
            num(ratio),
            realized_child_fraction=num(child_frac),
            psi=num(cfg.meta_psi),
            balance_ratio=num(ratio),
            arrivals=int(arrivals),
            spawned_on_empty=int(spawns),
            poisson_supply_share=num(
                arrivals / (arrivals + spawns) if arrivals + spawns > 0 else None
            ),
            supply_ratio_config=num(cfg.meta_supply_ratio),
            note=(
                "判定は実現子比率/ψ (Bernoulli 混合の恒等)。指示書 §3.2 の式は"
                " 文字どおりだと供給/需要 = 1/ψ² で発散する — README 参照"
            ),
        )

    out["flow_balance"] = safe_call(_flow)
    out["iceberg"] = safe_call(
        micro.iceberg_stats, (result.meta.get("l3") or {}).get("iceberg")
    )

    # --- propagator (イベント時間・直前ミッド基準) ---
    base_price = float(meta.get("base_price", 0.0))
    tick = float(meta.get("tick_size", cfg.tick_size))
    pm = np.asarray(meta.get("agg_trade_prev_mid_tick", np.empty(0)), dtype=np.float64)
    sizes_all = np.asarray(meta.get("agg_trade_size", np.empty(0)), dtype=np.float64)
    pm_k = pm[keep]
    fin = np.isfinite(pm_k)
    s_f = s[fin]
    logmid_f = np.log(base_price + tick * pm_k[fin]) if fin.any() else np.empty(0)
    sizes_f = sizes_all[keep][fin]

    out["response_mid"] = safe_call(micro.response_function, s_f, logmid_f, 200)
    prop = safe_call(
        micro.propagator_fit, s_f, sizes_f, logmid_f, 200, (5, 150)
    )
    out["propagator_mid"] = prop

    def _prop_stability() -> dict[str, Any]:
        """§12: サブサンプル (3 分割) で β を再推定して安定性を見る。"""
        n = s_f.size
        if n < 60_000:
            return na(f"3 分割には標本が足りません (n={n})")
        betas = []
        third = n // 3
        for i in range(3):
            sl = slice(i * third, (i + 1) * third)
            f = micro.propagator_fit(s_f[sl], sizes_f[sl], logmid_f[sl], 200, (5, 150))
            betas.append(f.get("beta") if f["status"] == "ok" else None)
        vals = [b for b in betas if b is not None]
        if not vals:
            return na("サブサンプル推定が全て失敗")
        return ok(
            num(max(vals) - min(vals)),
            betas=[num(b) if b is not None else None for b in betas],
            spread=num(max(vals) - min(vals)),
        )

    out["propagator_stability"] = safe_call(_prop_stability)

    # --- 平方根則 (完走・子 2 本以上のメタオーダー) ---
    def _sqrt_records() -> list[dict[str, float]]:
        n_tot = np.asarray(mo_rec.get("n_total", np.empty(0)))
        n_exec = np.asarray(mo_rec.get("n_exec", np.empty(0)))
        own = np.asarray(mo_rec.get("own_vol", np.empty(0)))
        v_first = np.asarray(mo_rec.get("vol_first", np.empty(0)))
        v_last = np.asarray(mo_rec.get("vol_last", np.empty(0)))
        mid_a = np.asarray(mo_rec.get("mid_first", np.empty(0)))
        mid_b = np.asarray(mo_rec.get("mid_last", np.empty(0)))
        t_a = np.asarray(mo_rec.get("t_first", np.empty(0)))
        t_b = np.asarray(mo_rec.get("t_last", np.empty(0)))
        sgn = np.asarray(mo_rec.get("sign", np.empty(0)))
        sel = (
            (n_exec >= 2)
            & (n_exec >= n_tot)  # 完走のみ (右打ち切りを混ぜない)
            & (t_a >= burn_sec)
            & (own > 0)
            & (mid_a > 0)
            & (mid_b > 0)
        )
        if not sel.any():
            return []
        # σ: 観測ミッドの分単位実現ボラ (prefix 和で O(1)/件)。短スパンは
        # 最低 30 分の対称窓に広げる (単児の σ=0 を避ける)。
        stride = max(int(round(60.0 / obs.step_seconds)), 1)
        lp_min = obs.log_price[::stride]
        r2 = np.diff(lp_min) ** 2
        prefix = np.concatenate([[0.0], np.cumsum(r2)])
        minutes_a = np.clip((t_a[sel] / 60.0).astype(np.int64), 0, r2.size)
        minutes_b = np.clip((t_b[sel] / 60.0).astype(np.int64), 0, r2.size)
        half_pad = np.maximum(0, 15 - (minutes_b - minutes_a) // 2)
        a_idx = np.clip(minutes_a - half_pad, 0, r2.size)
        b_idx = np.clip(minutes_b + half_pad, 0, r2.size)
        n_min_bars = np.maximum(b_idx - a_idx, 1)
        sigma = np.sqrt((prefix[b_idx] - prefix[a_idx]) / n_min_bars)
        v_mkt = v_last[sel] - v_first[sel] + own[sel]
        impact = sgn[sel] * np.log(
            (base_price + tick * mid_b[sel]) / (base_price + tick * mid_a[sel])
        )
        rows = []
        for q, v, sg, imp in zip(own[sel], v_mkt, sigma, impact):
            rows.append({"q": float(q), "v": float(v), "sigma": float(sg),
                         "impact": float(imp)})
        return rows

    records = _sqrt_records()
    out["sqrt_law"] = safe_call(lambda: micro.sqrt_law_check(records))

    # ★「サイズに線形か」のゲートは **N ビン別の符号つき平均インパクトの傾き**で
    # 判定する。frozen の sqrt_law_check (log-log + impact>0 選別) は S8 の
    # 高ノイズ域で歪む: (a) Q/V 形式は V が Q と共変して混雑度の回帰になる
    # (実測 δ=−0.47)、(b) 生 Q でも「正のみ」選別が小 N を上方バイアスして
    # 傾きが 0.37 に潰れる。ビン平均は符号つきでノイズを殺し選別を使わない —
    # 実測 0.888 (子 1 本あたり一定インパクトの加算にほぼ線形 ✓ §8.1)。
    def _impact_vs_size() -> dict[str, Any]:
        n_tot = np.asarray(mo_rec.get("n_total", np.empty(0)))
        n_exec = np.asarray(mo_rec.get("n_exec", np.empty(0)))
        mid_a = np.asarray(mo_rec.get("mid_first", np.empty(0)))
        mid_b = np.asarray(mo_rec.get("mid_last", np.empty(0)))
        t_a = np.asarray(mo_rec.get("t_first", np.empty(0)))
        sgn = np.asarray(mo_rec.get("sign", np.empty(0)))
        sel = (
            (n_exec >= 1) & (n_exec >= n_tot) & (t_a >= burn_sec) & (mid_a > 0)
        )
        if int(sel.sum()) < 5000:
            return na(f"完走メタオーダーが足りません (n={int(sel.sum())})")
        imp = sgn[sel] * np.log(
            (base_price + tick * mid_b[sel]) / (base_price + tick * mid_a[sel])
        )
        n_ex = n_exec[sel]
        edges = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 400])
        xs, ys, table = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (n_ex >= a) & (n_ex < b)
            cnt = int(m.sum())
            if cnt < 50:
                continue
            mean_i = float(imp[m].mean())
            gm = float(np.exp(np.mean(np.log(n_ex[m]))))
            table.append({
                "n_lo": int(a), "n_hi": int(b - 1), "count": cnt,
                "mean_impact": num(mean_i),
                "se": num(float(imp[m].std() / np.sqrt(cnt))),
            })
            if mean_i > 0:
                xs.append(np.log(gm))
                ys.append(np.log(mean_i))
        if len(xs) < 4:
            return na("正の平均インパクトを持つビンが足りません", bins=table)
        slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
        return ok(
            num(slope),
            slope=num(slope),
            intercept=num(intercept),
            n_bins=len(xs),
            n_metaorders=int(sel.sum()),
            bins=table,
        )

    out["impact_vs_size"] = safe_call(_impact_vs_size)

    # --- インパクト赤字の 4 値 (§8.3 — S9/S10 の到達目標) ---
    def _deficit() -> dict[str, Any]:
        daily = obs.to_bars(session).log_price_flat()
        vr = scaling.variance_ratio(
            daily[int(cfg.book_burn_in_days):], (2, 4, 8, 16, 32, 64)
        )
        # ★超拡散の主計器は**約定時間**の VR (指示書 §8.1 の機構そのもの:
        # Var[n 約定のミッド変化] ~ n^{2−γ})。日次 (壁時計) VR は whale の
        # 出方でシード間 {0.97, 1.9, 14.7} と乱れ、標本平均の除去が
        # 標本長スケールの成分を食う — 記録して S10 の目標値の座標系に使う。
        vr_tt: dict[str, Any] = {}
        if s_f.size > 20_000:
            r1 = np.diff(logmid_f)
            v1 = float(r1.var())
            for n_tr in (10, 100, 1000):
                rn = logmid_f[n_tr:] - logmid_f[:-n_tr]
                vr_tt[f"n{n_tr}"] = num(float(rn.var() / (n_tr * v1)))
        g = out["sign_acf_gamma"].get("gamma")
        beta_t = (1.0 - g) / 2.0 if g is not None else None
        beta_m = prop.get("beta")
        return ok(
            vr_tt.get("n1000"),
            vr_s8_trade_1000=vr_tt.get("n1000"),
            vr_trade_time=vr_tt,
            vr_s8_daily_max=num(vr.get("max_vr") if vr["status"] == "ok" else None),
            vr_daily_table=vr.get("table"),
            beta_measured=num(beta_m),
            beta_target=num(beta_t),
            beta_deficit=num(
                beta_m - beta_t if beta_m is not None and beta_t is not None else None
            ),
            sqrt_law_exponent=num(out["impact_vs_size"].get("slope")),
            sqrt_law_exponent_qv=num(out["sqrt_law"].get("delta")),
            targets_for_s10={
                "vr_daily": "0.90〜1.10",
                "beta": "(1−γ)/2 ± 0.05",
                "sqrt_law_exponent_qv": "0.4〜0.7",
            },
        )

    out["impact_deficit"] = safe_call(_deficit)
    return out


def _coupling_metrics(result: StageResult, cfg: Config) -> dict[str, Any]:
    """S10 の測定群。κ=0 かつ c_vol=0 なら全枝 ``not_applicable``。

    - gap / transmission / tracking: κ 結合の中核 (d の定常性・T(h)・日次相関)。
    - residual_sign_acf: 生成時バイアス E[ε|d] を引いた残差符号の γ — raw の
      C(ℓ) に重畳する情報チャネル (追跡ハーディング) は結合の物理なので、
      ⑪ 保存の判定は残差側 (S10a の解剖 — results/S10a/DECISION.md)。
    - vol_activity: ⑦ ボラ・出来高リンク (S10c、log-log 主計器) と §7.3/§7.4。
    """
    from . import coupling

    keys = ("gap", "transmission", "residual_sign_acf", "tracking", "vol_activity")
    if cfg.kappa <= 0 and cfg.c_vol <= 0:
        return {k: na("kappa=0 かつ c_vol=0 (結合なし)") for k in keys}
    return {
        "gap": safe_call(coupling.gap_metrics, result, cfg),
        "transmission": safe_call(coupling.transmission, result, cfg),
        "residual_sign_acf": safe_call(coupling.residual_sign_acf, result, cfg),
        "tracking": safe_call(coupling.pstar_tracking, result, cfg),
        "vol_activity": safe_call(coupling.vol_activity_link, result, cfg),
    }


def _chi_l1_metrics(result: StageResult, cfg: Config) -> dict[str, Any]:
    """S12 の測定群: χ₁/χ₃ のカオス性・スペクトル・独立性・予算・注入診断。

    Lyapunov 等の力学系性質は S5 と同じく**固定長参照系列** (初期値ごとに
    キャッシュ) で測る。独立性は 3 系列を共通日格子へ補間して相互相関。
    """
    keys = ("chi1", "chi3", "independence", "chi1_budget", "kernel_band")
    if not (cfg.enable_chaos_lambda or cfg.enable_chaos_branching):
        return {k: na("enable_chaos_lambda/branching=False") for k in keys}

    from ..chaos import chaos_generate, chi_window

    out: dict[str, Any] = {}
    l3d = result.meta.get("l3") or {}
    chi1_diag = ((l3d.get("cvol") or {}).get("chi1")) or {}
    chi3_diag = l3d.get("chi3") or {}

    def _series_tests(ic: float, days_per_unit: float, diag: dict) -> dict[str, Any]:
        ref = chaos_generate(
            system=cfg.chaos_system,
            params={"tau": cfg.chaos_tau_delay, "beta": cfg.chaos_beta,
                    "gamma": cfg.chaos_gamma, "n_exponent": cfg.chaos_n_exponent},
            length_units=20000.0, dt=cfg.chaos_dt, ic=ic,
            burn_in_units=cfg.chaos_burn_in_units,
            cache_dir=cfg.chaos_cache_dir or None,
            name=f"ref_ic{ic}",
        )
        ref_std = (ref.x - ref.x.mean()) / ref.x.std()
        lya = safe_call(chaos_val.lyapunov_rosenstein, ref.x, cfg.chaos_dt)
        spec = safe_call(scaling.spectral_peak, ref_std, cfg.chaos_dt * days_per_unit)
        return {
            "status": "ok", "value": lya.get("lyapunov_per_unit"),
            "lyapunov": lya, "spectral": spec,
            "peak_days": spec.get("peak_period_days"),
            "injection": diag or None,
        }

    if cfg.enable_chaos_lambda:
        out["chi1"] = safe_call(
            lambda: _series_tests(cfg.chi1_ic, cfg.chi1_days_per_unit, chi1_diag)
        )
    else:
        out["chi1"] = na("enable_chaos_lambda=False")
    if cfg.enable_chaos_branching:
        out["chi3"] = safe_call(
            lambda: _series_tests(cfg.chi3_ic, cfg.chi3_days_per_unit, chi3_diag)
        )
    else:
        out["chi3"] = na("enable_chaos_branching=False")

    def _independence() -> dict[str, Any]:
        from ..layers.l2_price import prepare_chaos_component

        n_days = max(float(cfg.n_days), 1000.0)
        grid = np.arange(1.0, n_days - 1.0, 0.5)
        series = {}
        if cfg.enable_chaos_lambda:
            t1, x1, _ = chi_window(cfg, n_days, "chi1")
            series["chi1"] = np.interp(grid, t1, x1)
        if cfg.enable_chaos_branching:
            t3, x3, _ = chi_window(cfg, n_days, "chi3")
            series["chi3"] = np.interp(grid, t3, x3)
        if cfg.enable_chaos_vol:
            t2, x2, _a, _c, _d = prepare_chaos_component(cfg, n_days)
            series["chi2"] = np.interp(grid, t2, x2)
        names = sorted(series)
        pairs = {}
        worst = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c = float(np.corrcoef(series[names[i]], series[names[j]])[0, 1])
                pairs[f"{names[i]}-{names[j]}"] = num(c)
                worst = max(worst, abs(c))
        return ok(num(worst), max_abs_corr=num(worst), pairs=pairs)

    out["independence"] = safe_call(_independence)

    def _budget() -> dict[str, Any]:
        if not cfg.enable_chaos_lambda:
            return na("enable_chaos_lambda=False")
        meta = result.events.meta if isinstance(result.events.meta, dict) else {}
        zg = meta.get("cvol_z_grid")
        if zg is None:
            return na("cvol_z_grid がありません")
        z_step = float(meta.get("cvol_z_step_sec", 60.0))
        t1, x1, _ = chi_window(cfg, float(cfg.n_days) + 1.0, "chi1")
        a1 = chi1_diag.get("a1")
        if a1 is None:
            return na("chi1 診断がありません")
        tz = np.arange(np.asarray(zg).size, dtype=np.float64) * z_step / 23400.0
        term = float(a1) * np.interp(tz, t1, x1)
        share = float(term.var() / np.log(np.asarray(zg)).var())
        return ok(
            num(share),
            var_share_realized=num(share),
            a1=num(float(a1)),
            e_factor=num(chi1_diag.get("e_factor")),
        )

    out["chi1_budget"] = safe_call(_budget)

    def _kernel_band() -> dict[str, Any]:
        peak = (out.get("chi1") or {}).get("peak_days")
        if peak is None:
            return na("chi1 のピークが測れていません")
        kernel_max_sec = float(max(cfg.hawkes_tau_seconds))
        ratio = float(peak) * 23400.0 / kernel_max_sec
        return ok(num(ratio), ratio_over_kernel=num(ratio),
                  peak_days=num(float(peak)), kernel_max_sec=num(kernel_max_sec))

    out["kernel_band"] = safe_call(_kernel_band)
    return out


def _feedback_metrics(result: StageResult, cfg: Config) -> dict[str, Any]:
    """S11 の測定群 (単独ランで測れる分)。enable_feedback=False なら全枝 na。

    - nt_dist: n_t の実現分布 (§8.3 — max < 0.97 が critical)
    - crisis / anatomy: 3 条件エピソードの頻度・継続・深さ・回復 (§6)
    - divergence_single: 単独版の発散検出 (記録 — L2 エポックの偽陽性込み。
      ゲートは multiseed のペア版 fb_divergences が担う)
    - u_stats: 驚き信号の定常性 (signal_is_surprise ゲート)
    """
    from . import feedback as fbv

    keys = ("nt_dist", "crisis", "anatomy", "divergence_single", "u_stats")
    if not cfg.enable_feedback:
        return {k: na("enable_feedback=False") for k in keys}
    out: dict[str, Any] = {}
    out["nt_dist"] = safe_call(fbv.nt_distribution, result, cfg)
    det = safe_call(fbv.crisis_detect, result, cfg)
    out["crisis"] = det
    out["anatomy"] = safe_call(
        fbv.crisis_anatomy, result, cfg,
        detection=det if det.get("status") == "ok" else None,
    )
    out["divergence_single"] = safe_call(fbv.divergence_monitor, result, cfg)

    def _u_stats() -> dict[str, Any]:
        fb = (result.meta.get("l3") or {}).get("feedback") or {}
        u_mean = fb.get("u_mean")
        return ok(
            num(u_mean),
            u_mean=num(u_mean),
            u_sd=num(fb.get("u_sd")),
            stationary=bool(u_mean is not None and abs(float(u_mean)) < 0.5),
        )

    out["u_stats"] = safe_call(_u_stats)

    def _saturation() -> dict[str, Any]:
        # §3.2 のコード検査を実行可能な形で: 全チャネルの乗数は tanh 飽和により
        # 有界 — 範囲を式から導出して記録する (非有界なら inf が出て落ちる)。
        import math

        rng_delta = (math.exp(-cfg.fb_b_delta), math.exp(cfg.fb_b_delta))
        rng_place = (math.exp(-cfg.fb_b_place), math.exp(cfg.fb_b_place))
        rng_n = (cfg.fb_n_min, cfg.fb_n_max)
        bounded = all(math.isfinite(x) for x in (*rng_delta, *rng_place, *rng_n))
        return ok(
            bounded,
            bounded=bool(bounded),
            delta_mult_range=[num(rng_delta[0]), num(rng_delta[1])],
            place_mult_range=[num(rng_place[0]), num(rng_place[1])],
            nt_range=[num(rng_n[0]), num(rng_n[1])],
        )

    out["saturation"] = safe_call(_saturation)
    return out


def _qr_metrics(result: StageResult, cfg: Config) -> dict[str, Any]:
    """S9 の測定群。queue-reactive が無効なら全枝 ``not_applicable``。

    η は**取引価格**系列で判定する (Robert–Rosenbaum の枠組み自体が取引価格の
    離散化モデルで、経験値 0.1〜0.3 もそこの値)。ミッド版は別枝で記録。
    ② (短期の負の自己相関) は**イベント時間**のミッド変化方向相関で判定 —
    壁時計 (1 分) の ACF(1) は whale トレンドの出方でシード間 ±0.13 揺れて
    符号すら安定しない (S8 の VR と同じ理由の計器選択。1 分版は記録)。
    """
    keys = (
        "eta_trade", "eta_trade_rows", "eta_mid", "mid_return_acf",
        "signature_mid", "obi", "state_diag", "reversion",
        "depth_tick_profile",
    )
    if not cfg.enable_queue_reactive:
        return {k: na("enable_queue_reactive=False") for k in keys}

    from ..types import EventType

    ev = result.events
    meta = ev.meta if isinstance(ev.meta, dict) else {}
    obs = result.observation
    session = float(obs.session_seconds)
    burn_sec = cfg.book_burn_in_days * session
    tick = float(meta.get("tick_size", cfg.tick_size))
    base_price = float(meta.get("base_price", 0.0))
    out: dict[str, Any] = {}

    et = ev.event_type
    tr_mask = et == int(EventType.TRADE)
    tr_t = ev.t[tr_mask]
    tr_px = np.round((ev.price[tr_mask] - base_price) / tick)
    keep_tr = tr_t >= burn_sec

    # --- η: 攻撃注文ごとの最終約定価格 (ゲート系列) と補助系列 ---
    ends = np.flatnonzero(np.concatenate([np.diff(tr_t) > 0, [True]]))
    last_px = tr_px[ends]
    last_t = tr_t[ends]
    eta_gate = safe_call(micro.estimate_eta, last_px[last_t >= burn_sec])
    eta_gate["uz_enabled"] = bool(cfg.enable_uncertainty_zones)
    out["eta_trade"] = eta_gate
    out["eta_trade_rows"] = safe_call(micro.estimate_eta, tr_px[keep_tr])

    bb = np.asarray(meta.get("best_bid_tick", np.empty(0)))
    ba = np.asarray(meta.get("best_ask_tick", np.empty(0)))
    okm = (bb >= 0) & (ba >= 0) & (ev.t >= burn_sec)
    mid_ev = 0.5 * (bb[okm] + ba[okm]).astype(np.float64)
    eta_mid = safe_call(micro.estimate_eta, mid_ev)
    out["eta_mid"] = eta_mid

    # --- ② ミッドリターンの 1 次自己相関 ---
    def _acf() -> dict[str, Any]:
        # 判定: イベント時間 (ゼロでないミッド変化の方向相関 = η_mid の恒等変換)
        cs = eta_mid.get("change_sign_corr")
        # 記録: 1 分バー ACF(1) — whale トレンド汚染つき
        stride = max(1, int(round(60.0 / obs.step_seconds)))
        lp_min = obs.log_price[::stride]
        rm = np.diff(lp_min[int(burn_sec / 60.0):])
        acf1_min = (
            float(np.corrcoef(rm[:-1], rm[1:])[0, 1]) if rm.size > 1000 else None
        )
        if cs is None:
            return na("η_mid が計算できていません")
        return ok(
            num(cs),
            change_sign_corr_event=num(cs),
            acf1_1min_recorded=num(acf1_min),
            note="判定はイベント時間 (1 分版は whale トレンドで符号不安定 — 記録)",
        )

    out["mid_return_acf"] = safe_call(_acf)

    # --- signature plot (ミッド): サンプリング間隔別の分散/秒 ---
    def _signature() -> dict[str, Any]:
        step = obs.step_seconds
        lp = obs.log_price[int(burn_sec / step):]
        rows = {}
        for sec in (1.0, 5.0, 15.0, 60.0, 300.0, 900.0):
            stride = int(round(sec / step))
            if stride < 1 or lp.size // stride < 1000:
                continue
            rr = np.diff(lp[::stride])
            rows[f"s{int(sec)}"] = num(float(rr.var() / sec))
        if len(rows) < 3:
            return na("有効なサンプリングが足りません")
        vals = [v for v in rows.values() if v is not None]
        ratio = vals[0] / vals[-1] if vals[-1] and vals[-1] > 0 else None
        return ok(
            num(ratio),
            per_sec_variance=rows,
            short_over_long=num(ratio),
            decreasing=bool(ratio is not None and ratio > 1.0),
        )

    out["signature_mid"] = safe_call(_signature)

    # --- ⑩ OBI: 攻撃注文格子での予測相関 (機構創発 — バイアスなし) ---
    def _obi() -> dict[str, Any]:
        trade_idx = np.flatnonzero(tr_mask)
        starts = np.flatnonzero(np.concatenate([[True], np.diff(tr_t) > 0]))
        pre_idx = np.maximum(trade_idx[starts] - 2, 0)
        db = np.asarray(meta.get("depth_bid"))[pre_idx].astype(np.float64)
        da = np.asarray(meta.get("depth_ask"))[pre_idx].astype(np.float64)
        imb = (db - da) / np.maximum(db + da, 1e-9)
        pm = np.asarray(meta.get("agg_trade_prev_mid_tick"), dtype=np.float64)
        t_agg = np.asarray(meta.get("agg_trade_t"))
        keep = t_agg >= burn_sec
        res = micro.obi_predictive(imb[keep], pm[keep], horizons=(1, 2, 5, 10))
        if res.get("status") == "ok":
            res["obi_bias_config"] = float(cfg.qr_obi_bias)
        return res

    out["obi"] = safe_call(_obi)

    # --- 状態依存の実測 (配置と取消が実際に状態を見ているか) ---
    def _state_diag() -> dict[str, Any]:
        lo_mask = (et == int(EventType.LIMIT_ADD)) & (ev.t >= burn_sec)
        bb_f = bb.astype(np.float64)
        ba_f = ba.astype(np.float64)
        sp_prev = np.concatenate([[np.nan], (ba_f - bb_f)[:-1]])
        bb_prev = np.concatenate([[np.nan], bb_f[:-1]])
        ba_prev = np.concatenate([[np.nan], ba_f[:-1]])
        px_ticks = np.round((ev.price[lo_mask] - base_price) / tick)
        side_lo = ev.side[lo_mask]
        inside = (
            np.where(side_lo == 1, px_ticks > bb_prev[lo_mask],
                     px_ticks < ba_prev[lo_mask])
            & np.isfinite(bb_prev[lo_mask])
        )
        s_at = sp_prev[lo_mask]
        rates = {}
        for lo_s, hi_s in ((2, 3), (3, 5), (5, 9), (9, 30)):
            m = np.isfinite(s_at) & (s_at >= lo_s) & (s_at < hi_s)
            if int(m.sum()) > 500:
                rates[f"s{lo_s}_{hi_s}"] = num(float(inside[m].mean()))
        vals = [v for v in rates.values() if v is not None]
        cx_mask = (et == int(EventType.CANCEL)) & (ev.t >= burn_sec)
        px_cx = np.round((ev.price[cx_mask] - base_price) / tick)
        d_cx = np.where(
            ev.side[cx_mask] == 1,
            bb_prev[cx_mask] - px_cx,
            px_cx - ba_prev[cx_mask],
        )
        d_cx = d_cx[np.isfinite(d_cx) & (d_cx >= 0)]
        return ok(
            None,
            inspread_rate_by_spread=rates,
            inspread_monotone=bool(
                len(vals) >= 2 and all(b > a for a, b in zip(vals, vals[1:]))
            ),
            cancel_dist_median=num(float(np.median(d_cx))) if d_cx.size else None,
            cancel_dist_p90=num(float(np.percentile(d_cx, 90))) if d_cx.size else None,
        )

    out["state_diag"] = safe_call(_state_diag)

    # --- 約定後のミッド戻り (ノイズトレード条件付け — 純粋な板応答) ---
    def _reversion() -> dict[str, Any]:
        pm = np.asarray(meta.get("agg_trade_prev_mid_tick"), dtype=np.float64)
        sd = np.asarray(meta.get("agg_trade_side"), dtype=np.float64)
        mt = np.asarray(meta.get("agg_trade_meta"), dtype=np.float64)
        t_agg = np.asarray(meta.get("agg_trade_t"))
        keep = t_agg >= burn_sec
        return micro.mean_reversion_profile(
            sd[keep], pm[keep], meta_ids=mt[keep]
        )

    out["reversion"] = safe_call(_reversion)

    # --- ⑳ デプスのハンプ位置 (best からの tick 距離) ---
    def _depth_ticks() -> dict[str, Any]:
        b = result.book
        keep = b.t > burn_sec
        prof: dict[int, list[float]] = {}
        for px_side, sz_side in ((b.bid_px, b.bid_sz), (b.ask_px, b.ask_sz)):
            px_k = px_side[keep]
            sz_k = sz_side[keep]
            best = px_k[:, 0:1]
            d_ticks = np.abs(np.round((px_k - best) / tick)) + 1  # best = 距離 1
            for dd in range(1, 31):
                m = d_ticks == dd
                if int(m.sum()) > 200:
                    prof.setdefault(dd, []).append(float(np.nanmean(sz_k[m])))
        if len(prof) < 5:
            return na("スナップショットの距離ビンが足りません")
        avg = {d: float(np.mean(v)) for d, v in prof.items()}
        peak = max(avg, key=avg.get)
        return ok(
            num(float(peak)),
            peak_tick_distance=int(peak),
            profile={str(k): num(v) for k, v in sorted(avg.items())},
        )

    out["depth_tick_profile"] = safe_call(_depth_ticks)
    return out


def engine_liveness_from_meta(l3_meta: dict, horizon_sec: float) -> dict[str, Any]:
    """枯渇カウンタの読み出し (micro.book_liveness の配線)。"""
    from ..layers.book_engine import C_EMPTY_SIDE_TIME, C_MO_REJECT_EVENTS, C_MO_REJECT_VOL

    c = np.asarray(l3_meta.get("counters"))
    return micro.book_liveness(
        float(c[C_EMPTY_SIDE_TIME]), horizon_sec,
        float(c[C_MO_REJECT_EVENTS]), float(c[C_MO_REJECT_VOL]),
    )


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
    # ★時間の単位に注意: 合成系列の dt は「日」なので、Theiler 窓や先読み時間も
    # 日で渡す。chi 単体の既定値 (系固有単位で 100/400) を流用すると窓が系列長を
    # 超えて空集合になる (実際にそうなって IndexError を出した)。
    stride_days = float(t_days[1] - t_days[0])
    out["composite_tests"] = {
        "lyapunov": safe_call(
            chaos_val.lyapunov_rosenstein, lv_with, stride_days,
            5, None, None, 40.0, (2.0, 20.0),
        ),
        "correlation_dimension": safe_call(
            chaos_val.correlation_dimension, lv_with, stride_days,
            (3, 4, 5, 6), None, 5.0,
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
