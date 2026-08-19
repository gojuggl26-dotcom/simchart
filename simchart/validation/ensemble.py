"""ボラ正規化のアンサンブル断面検証。

なぜ断面か
----------
S1 ゲートの ``e_sigma2`` (|E[sigma^2]/sigma_bar^2 - 1| < 0.02) と ``var_total`` /
``var_budget`` は**期待値・分散に対する条件**である。ところが log sigma には
時定数 500 日の成分が入っているため、1 経路の時間平均は 5000 日でも実効独立標本が
~30 しかなく、E[sigma^2] の推定値は ±17% ゆらぐ。閾値 2% は 1 経路では原理的に
判定できない。

そこで E[·] を文字どおりに実装する: **定常初期化した独立標本の断面**を大量に引き、
その標本平均で判定する。20 万本で E[sigma^2] の標準誤差は ~0.22% になり、
閾値 2% が 9 標準誤差に相当する本物の検定になる。

MSM の断面分布は各成分 {m0, 2-m0} 等確率 (Poisson 切替の定常分布は t に依らず
これ)、OU の断面分布は N(0, Var(X))。したがって断面は**切替動学を経由せずに**
サンプリングできる。この断面が検証するのは:

- :func:`~simchart.layers.l2_price.solve_m0` の逆算 (m0 の値)
- :func:`~simchart.layers.l2_price.compose_log_sigma` の合成式
  (0.5 係数・凸性補正 -Var(X) の配線)
- 分散配分がターゲットどおりか

**切替動学そのものは検証しない** — それは :func:`~simchart.validation.scaling.msm_diagnostics`
(実測切替率 vs 指定 gamma_i) の担当。合成式は生成側と同じ
:func:`~simchart.layers.l2_price.compose_log_sigma` を使うので、式が 2 か所で
乖離する事故は起きない。
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..layers.l2_price import compose_log_sigma, msm_theoretical_var_log_sigma, solve_m0
from ..rng import RNGRegistry
from .base import na, num, ok

__all__ = ["vol_cross_section"]


def vol_cross_section(config: Config, n_paths: int | None = None) -> dict:
    """定常断面から E[sigma^2]・Var(log sigma)・成分シェアを推定する。

    乱数は ``validation.ensemble`` ストリーム (config.seed から名前ハッシュ導出)
    を使うので決定論的で、生成系のストリームには一切触れない。
    """
    if not (config.enable_msm or config.enable_slow_ou):
        return na("確率ボラが無効です (enable_msm / enable_slow_ou とも False)")

    n = int(n_paths if n_paths is not None else config.validation.ensemble_n_paths)
    rng = RNGRegistry(config.seed).get("validation.ensemble")
    log_sigma_bar = math.log(config.sigma_bar)

    half_log_msm: np.ndarray | float = 0.0
    var_msm_theory = 0.0
    m0 = None
    if config.enable_msm:
        k = config.msm_k
        m0 = solve_m0(k, config.vol_var_target_msm)
        states = rng.integers(0, 2, size=(n, k))
        log_hi, log_lo = math.log(m0), math.log(2.0 - m0)
        half_log_msm = 0.5 * np.where(states == 1, log_hi, log_lo).sum(axis=1)
        var_msm_theory = msm_theoretical_var_log_sigma(k, m0)

    x_slow: np.ndarray | float = 0.0
    var_slow = 0.0
    if config.enable_slow_ou:
        var_slow = config.vol_var_target_slow
        x_slow = rng.normal(0.0, math.sqrt(var_slow), size=n)

    log_sigma = np.asarray(
        compose_log_sigma(log_sigma_bar, half_log_msm, x_slow, var_slow), dtype=np.float64
    )

    sigma2_ratio = np.exp(2.0 * (log_sigma - log_sigma_bar))
    mean_ratio = float(sigma2_ratio.mean())
    se_ratio = float(sigma2_ratio.std(ddof=1) / math.sqrt(n))

    var_log_sigma = float(log_sigma.var(ddof=1))
    var_components: dict[str, float] = {}
    if config.enable_msm:
        var_components["msm"] = float(np.asarray(half_log_msm).var(ddof=1))
    if config.enable_slow_ou:
        var_components["slow_ou"] = float(np.asarray(x_slow).var(ddof=1))
    shares = {k_: v / var_log_sigma for k_, v in var_components.items()}

    var_total_theory = var_msm_theory + var_slow
    budget = config.vol_var_budget_total
    return ok(
        num(mean_ratio),
        n_paths=n,
        e_sigma2_ratio=num(mean_ratio),
        e_sigma2_se=num(se_ratio),
        e_sigma2_z=num((mean_ratio - 1.0) / se_ratio) if se_ratio > 0 else None,
        var_log_sigma=num(var_log_sigma),
        var_log_sigma_theory=num(var_total_theory),
        component_vars={k_: num(v) for k_, v in var_components.items()},
        # ★シェアには分母の違う 2 種類がある。取り違えると §6 の予算表と 1.4 倍
        # ずれる (実際に一度やった)。
        #   shares_of_current   … 分母 = 現在の Var(log sigma) 合計 (S1 で ~0.175)
        #   shares_of_budget    … 分母 = 最終予算 vol_var_budget_total (0.25)。
        #                         指示書 §6/§10 の「MSM 50% / OU 20%」はこちら
        shares_of_current={k_: num(v) for k_, v in shares.items()},
        shares_of_budget={k_: num(v / budget) for k_, v in var_components.items()},
        budget_total=num(budget),
        budget_used_fraction=num(var_log_sigma / budget),
        shares_theory_of_budget={
            **({"msm": num(var_msm_theory / budget)} if config.enable_msm else {}),
            **({"slow_ou": num(var_slow / budget)} if config.enable_slow_ou else {}),
        },
        m0=num(m0) if m0 is not None else None,
        sd_log_sigma=num(math.sqrt(var_log_sigma)),
    )
