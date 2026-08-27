"""S13: 多資産の共通因子状態。

因子構造のうち全資産で共有される実現を 1 か所で生成する:

- ``z_F`` 共通リターン革新 (資産側で z_i = β_i z_F + √(1−β_i²) z_id に合成)
- 共通 MSM 遅い側 k_c 成分 (``0.5 Σ_{j<k_c} log M_j``)
- 共通 緩慢 OU 分散 f_c·var_slow。レバレッジ有効時は駆動を ρ_slow で z_F と相関
                させる (設計要件 — 共通チャネルにも同じ ρ)
- 共通ジャンプ 強度シェア s_J のシステマティック Kou 過程。**全資産に同一の
                対数サイズ**で乗る (市場全体のニュース §4.3)。強度は共通ボラ状態
                (共通 MSM + 共通 OU + χ₂ + φ_σ) で変調 — 資産固有の状態は読まない
                (共通過程が資産に依存したら共通でなくなる)

このモジュールの生成は n_assets にも資産オーバーライドにも依存しない。
これが §8.2 (資産追加でも既存資産はビット単位不変) の共通側の前提である。
乱数は ``cross.*`` ストリームのみ消費する。

χ₂ は決定論 (乱数不消費) なので共有は自動 — 全資産が同じ設定から同じ系列を
再構成する。ここでは共通ジャンプの変調に使うためだけに参照する。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config
from ..rng import RNGRegistry
from .l2_price import (
    prepare_chaos_component,
    simulate_msm_path,
    simulate_ou_path,
    solve_m0,
)

__all__ = ["CommonFactorState", "build_common_state"]


@dataclass
class CommonFactorState:
    """全資産が読む共有実現。資産側は読み取り専用で扱うこと (in-place 禁止)。"""

    z_f: np.ndarray
    half_log_msm: np.ndarray | float
    x_slow: np.ndarray | float
    msm_k_common: int
    ou_common_var: float
    jump_idx: np.ndarray | None
    jump_sizes: np.ndarray | None
    jump_comp: np.ndarray | None
    jump_lam_eff: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_common_state(
    config: Config, registry: RNGRegistry, calendar, t: np.ndarray
) -> CommonFactorState:
    """共通因子状態を生成する (run_multi から 1 回だけ呼ばれる)。"""
    if config.n_assets < 2:
        raise ValueError("build_common_state は n_assets >= 2 でのみ意味を持ちます")
    seconds_per_day = calendar.session_seconds()
    t_days = t / seconds_per_day
    n = int(t.shape[0])
    diag: dict[str, Any] = {"n_steps": n - 1}

    # --- 共通リターン革新 z_F --------------------------------------------
    z_f = registry.get("cross.factor").standard_normal(n - 1)
    diag["z_f_digest"] = hashlib.sha256(np.ascontiguousarray(z_f).tobytes()).hexdigest()

    # --- 共通 MSM (遅い側 k_c 成分) --------------------------------------
    half_log_msm: np.ndarray | float = 0.0
    if config.enable_msm and config.msm_k_common > 0:
        half_log_msm, msm_diag = simulate_msm_path(
            config, registry.get("cross.vol_msm"), t_days, (0, config.msm_k_common)
        )
        diag["msm"] = msm_diag

    # --- 共通 緩慢 OU (分散 f_c·var_slow) --------------------------------
    x_slow: np.ndarray | float = 0.0
    ou_common_var = 0.0
    if config.enable_slow_ou and config.ou_common_share > 0.0:
        ou_common_var = config.ou_common_share * config.vol_var_target_slow
        theta = math.log(2.0) / config.ou_half_life_days
        driver = None
        if config.enable_leverage:
            # 共通チャネルのレバレッジ (§4.4): ξ_F = ρ_slow z_F + √(1−ρ²) w_F。
            # 固有チャネル (資産側の xi) と同じ値の ρ を使う。
            rho_s = config.leverage_rho_slow
            driver = registry.get("cross.leverage_slow").standard_normal(n - 1)
            driver *= math.sqrt(1.0 - rho_s * rho_s)
            # z_f を変異させない (資産側が読む)。チャンクで加算する。
            chunk = 8_000_000
            for i0 in range(0, n - 1, chunk):
                i1 = min(i0 + chunk, n - 1)
                driver[i0:i1] += rho_s * z_f[i0:i1]
        x_slow, ou_diag = simulate_ou_path(
            registry.get("cross.vol_slow"), t_days, theta, ou_common_var, driver
        )
        ou_diag["half_life_days"] = config.ou_half_life_days
        ou_diag["driver"] = "leverage_common" if driver is not None else "independent"
        diag["slow_ou"] = ou_diag
        del driver

    # --- 共通 (システマティック) ジャンプ --------------------------------
    jump_idx: np.ndarray | None = None
    jump_sizes: np.ndarray | None = None
    jump_comp: np.ndarray | None = None
    jump_lam_eff = 0.0
    if config.enable_jump and config.jump_common_share > 0.0:
        jump_idx, jump_sizes, jump_comp, jump_lam_eff, jd = _common_jumps(
            config, registry, calendar, t, t_days, half_log_msm, x_slow, ou_common_var
        )
        diag["jump"] = jd

    return CommonFactorState(
        z_f=z_f,
        half_log_msm=half_log_msm,
        x_slow=x_slow,
        msm_k_common=int(config.msm_k_common),
        ou_common_var=ou_common_var,
        jump_idx=jump_idx,
        jump_sizes=jump_sizes,
        jump_comp=jump_comp,
        jump_lam_eff=jump_lam_eff,
        diagnostics=diag,
    )


def _common_jumps(
    config: Config,
    registry: RNGRegistry,
    calendar,
    t: np.ndarray,
    t_days: np.ndarray,
    half_log_msm: np.ndarray | float,
    x_slow: np.ndarray | float,
    ou_common_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """共通ジャンプ (時刻・サイズ・補償) を生成する。

    強度は資産側 (:meth:`GBMPriceLayer._simulate_jumps`) と同じ規約:

        λ_c(t) = s_J·λ0·scale·min(ratio_c(t)^ρ, cap)

    ratio_c = 共通ボラ状態 exp(0.5Σ_slow log M + x_F − f_c·var_slow + aχ₂ − c_χ)·φ_σ(u)
    で、scale は S4/S5 の補正 (ON の取り分・φ^ρ の Jensen・χ₂ の Jensen) を資産側と
    同じ形で打ち消す。共通成分自身の Jensen (E[√M]^{k_c} < 1、e^{−f_c·var/2} 型) は
    資産側が固有成分で残すのと同じ規約で残す (S12 の λ0=5 → 実効 ~4.2/年 と
    同型 — 実効強度は診断で報告する)。

    サイズ分布は資産側と同一の Kou (同じ η — 市場ニュースだから大きい等の
    追加自由度は入れない §2 作り込まない)。補償 −λ_c k dt は全資産共通なので
    ここで 1 回だけ計算する。
    """
    n_steps = int(t.shape[0]) - 1
    dt_years = float(t[1] - t[0]) / (config.ann_days * calendar.session_seconds())

    # log ratio_c を組み立ててから exp (in-place、大配列 1 本)
    lam = np.zeros(n_steps, dtype=np.float64)
    if isinstance(half_log_msm, np.ndarray):
        lam += half_log_msm[:-1]
    if isinstance(x_slow, np.ndarray):
        lam += x_slow[:-1]
    lam -= ou_common_var  # OU 族の凸性補正の共通取り分 (資産側 compose と同じ規約)

    chi_jensen = 1.0
    if config.enable_chaos_vol:
        chaos_t_days, chi_norm, a_chi, c_chi, chi_diag = prepare_chaos_component(
            config, float(t_days[-1])
        )
        chunk = 8_000_000
        for i0 in range(0, n_steps, chunk):
            i1 = min(i0 + chunk, n_steps)
            lam[i0:i1] += a_chi * np.interp(t_days[i0:i1], chaos_t_days, chi_norm)
        lam -= c_chi
        chi_jensen = float(chi_diag["jensen_intensity_factor"])
        del chaos_t_days, chi_norm

    np.exp(lam, out=lam)
    if config.enable_seasonality and hasattr(calendar, "phi_sigma_of_u"):
        u = calendar.intraday_position(t[:-1])
        lam *= np.asarray(calendar.phi_sigma_of_u(u), dtype=np.float64)
        del u
    if config.jump_vol_exponent != 1.0:
        np.power(lam, config.jump_vol_exponent, out=lam)
    cap_binding = float((lam > config.jump_intensity_cap).mean())
    np.minimum(lam, config.jump_intensity_cap, out=lam)

    # S4/S5 と同じ補正 (資産側 jump_intensity_scale と同じ構成要素)
    scale = 1.0
    if config.enable_overnight:
        scale *= 1.0 - config.overnight_variance_share
    if config.enable_seasonality and hasattr(calendar, "phi_sigma_of_u"):
        uu = np.linspace(0.0, 1.0, 20001)
        phi = np.asarray(calendar.phi_sigma_of_u(uu), dtype=np.float64)
        scale /= float(np.trapezoid(phi**config.jump_vol_exponent, uu))
    if config.enable_chaos_vol:
        scale /= chi_jensen
    lam *= config.jump_lambda_per_year * config.jump_common_share * scale  # [1/年]

    u_draw = registry.get("cross.jump_time").uniform(size=n_steps)
    prob = lam * dt_years
    mask = u_draw < prob
    del u_draw, prob
    n_jumps = int(mask.sum())

    rng_size = registry.get("cross.jump_size")
    u_sign = rng_size.uniform(size=n_jumps)
    e_mag = rng_size.standard_exponential(size=n_jumps)
    up = u_sign < config.jump_p_up
    sizes = np.where(up, e_mag / config.jump_eta_up, -e_mag / config.jump_eta_down)

    k_comp = (
        config.jump_p_up * config.jump_eta_up / (config.jump_eta_up - 1.0)
        + (1.0 - config.jump_p_up) * config.jump_eta_down / (config.jump_eta_down + 1.0)
        - 1.0
    )
    comp = lam  # in-place: lam を潰して補償列に転用 (資産側と同じ)
    comp *= -k_comp * dt_years
    lam_eff = -float(comp.mean()) / (k_comp * dt_years)

    idx = np.nonzero(mask)[0].astype(np.int64)
    del mask

    # 共通ボラ状態の理論 Jensen (実効強度 < s_J·λ0 の要因分解 — 診断用)
    m0 = solve_m0(config.msm_k, config.vol_var_target_msm) if config.enable_msm else 1.0
    e_sqrt_m = 0.5 * (math.sqrt(m0) + math.sqrt(2.0 - m0))
    jensen_msm = e_sqrt_m ** config.msm_k_common if config.enable_msm else 1.0
    jensen_ou = math.exp(ou_common_var * (config.jump_vol_exponent**2) / 2.0 - config.jump_vol_exponent * ou_common_var)

    jd = {
        "common_share": config.jump_common_share,
        "n_jumps": n_jumps,
        "n_up": int(up.sum()),
        "lambda_effective_per_year": lam_eff,
        "lambda_nominal_per_year": config.jump_lambda_per_year * config.jump_common_share,
        "cap_binding_fraction": cap_binding,
        "k_compensation": k_comp,
        "scale_s4s5": scale,
        "jensen_msm_common_theory": jensen_msm,
        "jensen_ou_common_theory": jensen_ou,
        "mean_jump": float(sizes.mean()) if n_jumps else None,
        "sizes_digest": hashlib.sha256(np.ascontiguousarray(sizes).tobytes()).hexdigest(),
    }
    return idx, sizes, comp, lam_eff, jd
