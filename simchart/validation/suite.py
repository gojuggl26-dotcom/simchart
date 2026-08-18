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

from typing import Any, Mapping

import numpy as np

from ..config import TRADING_DAYS_PER_YEAR, Config
from ..types import StageResult
from . import cross, memory, micro, scaling, tails
from .base import safe_call

__all__ = ["run_all", "collect_errors", "flatten"]


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
    acf_r = safe_call(memory.acf, r_primary_2d, v.acf_max_lag)
    acf_abs_r = safe_call(memory.acf, abs_r_primary, v.acf_abs_max_lag)
    metrics["memory"] = {
        "acf_r": acf_r,
        "acf_abs_r": acf_abs_r,
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
    metrics["scaling"] = {
        "variance_ratio": safe_call(scaling.variance_ratio, primary_bars.log_price, v.vr_qs),
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
