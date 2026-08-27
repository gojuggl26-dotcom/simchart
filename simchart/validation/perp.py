"""perp 固有の検証 (S0-perp §8)。

全関数を S0-perp で宣言し、該当層が無効なら not_applicable を返す —
S0 の validation スイートと同じ規約 (例外を投げない。「まだ使えないから」と
省略すると、後段で関数名・シグネチャ・返り値の形が場当たりに決まる)。

S0-perp 時点で実測が動くのは 2 つ:

- :func:`weekly_profile` — 週内プロファイル。S0-perp では平坦が正解
  (週次季節性は S4-perp)。ゲート weekly_profile_flat が使う。
- :func:`phi_normalization_check` — φ の正規化検査 (§3.1)。equity 側の
  正規化分離 (φ_σ: mean(φ²)=1 / φ_λ: mean(φ)=1) を検証する汎用計器。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import na, num, ok

__all__ = [
    "basis_stats",
    "funding_stats",
    "funding_sawtooth",
    "arb_band_analysis",
    "oi_dynamics",
    "liquidation_cascade_sizes",
    "liq_density_profile",
    "g_liquidation_derived",
    "block_discretization_effect",
    "weekly_profile",
    "phi_normalization_check",
]

_S10 = "基差・funding は S10-perp で実装されます (enable_funding=False)"
_S11 = "建玉・清算は S11-perp で実装されます (enable_positions=False)"
_S6 = "ブロック時間の離散化は S6-perp で実装されます"


# ---------------------------------------------------------------------------
# S10-perp (基差・funding・裁定) — S0-perp では N/A
# ---------------------------------------------------------------------------
def basis_stats(perp: Any = None, index: Any = None) -> dict:
    """基差 (perp − index)/index の統計。S10-perp で実装。"""
    if perp is None or index is None:
        return na(_S10)
    return na(_S10)


def funding_stats(history: Any = None) -> dict:
    """funding レート履歴の統計 (平均・符号持続・cap 到達率)。"""
    if history is None:
        return na(_S10)
    return na(_S10)


def funding_sawtooth(basis: Any = None, times: Any = None) -> dict:
    """funding 確定時刻を挟んだ基差の鋸歯パターン検出。"""
    if basis is None:
        return na(_S10)
    return na(_S10)


def arb_band_analysis(basis: Any = None, threshold: float | None = None) -> dict:
    """裁定閾値バンド内滞在率と超過時の復帰速度。"""
    if basis is None:
        return na(_S10)
    return na(_S10)


# ---------------------------------------------------------------------------
# S11-perp (建玉・清算) — S0-perp では N/A
# ---------------------------------------------------------------------------
def oi_dynamics(oi: Any = None, prices: Any = None) -> dict:
    """open interest の動学 (価格との共変動・平均回帰)。"""
    if oi is None:
        return na(_S11)
    return na(_S11)


def liquidation_cascade_sizes(events: Any = None) -> dict:
    """清算カスケードのサイズ分布 (裾指数・連鎖長)。"""
    if events is None:
        return na(_S11)
    return na(_S11)


def liq_density_profile(book: Any = None) -> dict:
    """清算価格密度のプロファイル (現在価格からの距離帯別)。"""
    if book is None:
        return na(_S11)
    return na(_S11)


def g_liquidation_derived(rho_liq: Any = None, impact: Any = None) -> dict:
    """清算ループゲイン g_liq の導出値 (清算密度 × インパクト)。

    S11 の g (RV フィードバック) と同じ役割の量を清算経路で定義する。
    """
    if rho_liq is None:
        return na(_S11)
    return na(_S11)


# ---------------------------------------------------------------------------
# S6-perp (ブロック時間) — S0-perp では N/A
# ---------------------------------------------------------------------------
def block_discretization_effect(prices: Any = None, block_ms: int | None = None) -> dict:
    """ブロック単位の約定確定が短期統計に与える離散化効果。"""
    if prices is None:
        return na(_S6)
    return na(_S6)


# ---------------------------------------------------------------------------
# S0-perp で実装済み
# ---------------------------------------------------------------------------
def weekly_profile(x: np.ndarray, times_sec: np.ndarray, n_bins: int = 7,
                   period_hours: float = 168.0) -> dict:
    """週内プロファイル: 週期間 (既定 168h) を n_bins に割った |x| の平均。

    S4-perp の週次季節性の計器。S0-perp では平坦が正解 (φ ≡ 1) で、
    ゲート weekly_profile_flat がビン平均の最大/最小比が 1 に近いことを
    確認する。ビン数 7 = 曜日粒度。

    Parameters
    ----------
    x:
        測る系列 (例: バーの |リターン|)。
    times_sec:
        各要素の時刻 (シミュレーション秒)。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.asarray(times_sec, dtype=np.float64).ravel()
    if x.size != t.size or x.size < n_bins * 10:
        return na(f"標本不足 (n={x.size}, 必要 {n_bins * 10})")
    period = period_hours * 3600.0
    pos = np.mod(t, period) / period
    bins = np.minimum((pos * n_bins).astype(np.int64), n_bins - 1)
    sums = np.bincount(bins, weights=np.abs(x), minlength=n_bins)
    counts = np.bincount(bins, minlength=n_bins)
    if (counts == 0).any():
        return na("空のビンがあります (期間が週数に対して短すぎる)")
    prof = sums / counts
    mean = float(prof.mean())
    if mean <= 0:
        return na("プロファイル平均が 0 です")
    rel = prof / mean
    return ok(
        num(float(rel.max() / rel.min())),
        max_over_min=num(float(rel.max() / rel.min())),
        max_abs_dev_from_flat=num(float(np.max(np.abs(rel - 1.0)))),
        profile=[float(v) for v in rel],
        n_bins=int(n_bins),
        period_hours=float(period_hours),
        n_obs=int(x.size),
    )


def phi_normalization_check(
    phi: np.ndarray | Sequence[float], kind: str
) -> dict:
    """φ の正規化検査 (S0-perp §3.1 の検証計器)。

    - ``kind="sigma"``: mean(φ²) = 1 — 加算されるのが分散なので二乗の平均。
      mean(φ)=1 にすると Jensen の不等式で日次積分分散が目標を超える。
    - ``kind="lambda"``: mean(φ) = 1 — 強度なので一乗。

    equity 側は S4 でこの分離を実装済み (normalize_phi_sigma /
    normalize_phi_lambda)。この関数はそれを外部から検査する汎用形で、
    perp の週次 φ (S4-perp) にも同じ規約を適用する。
    """
    arr = np.asarray(phi, dtype=np.float64).ravel()
    if arr.size < 10:
        return na("φ の標本が少なすぎます")
    if (arr <= 0).any():
        return na("φ に非正の値があります")
    if kind == "sigma":
        val = float((arr**2).mean())
        target = "mean(phi^2) = 1"
    elif kind == "lambda":
        val = float(arr.mean())
        target = "mean(phi) = 1"
    else:
        return na(f"kind は sigma / lambda のいずれかです: {kind!r}")
    return ok(
        num(val),
        normalization_value=num(val),
        abs_error=num(abs(val - 1.0)),
        target=target,
        kind=kind,
        passed_1e3=bool(abs(val - 1.0) < 1e-3),
    )
