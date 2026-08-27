"""L0: カレンダー層。

最終系での役割
--------------
日内 U 字の活動度季節性 phi(t)、寄付・引けの特異点、オーバーナイト・ギャップ。
これらは「見かけの統計性質」を大量に作る (日内ボラの U 字は、それを除去しないと
長期記憶や多重フラクタルに見えてしまう)。したがって L0 は最下層に置き、
上の層はすべて phi(t) で伸縮された時間の上で動く。

S4 での実装 (対象市場: 米国株相当の連続単一セッション 6.5 時間)
--------------------------------------------------------------
- ``phi_sigma(u)``: 観測ボラへの**乗法変調** ``sigma_obs = phi_sigma(u) sigma_stoch``。
  確率ボラ成分 (MSM / 緩慢 OU / ラフ) 自体には季節性を掛けない — 掛けると
  S2 のラフ成分の粗さ H が時間変形で歪み、S3 との比較も成立しなくなる。
  決定論的な乗法変調に限定することで、**phi で割れば S3 の系列が完全に復元できる**
  (これが S4 のゲートの検定力の源)。
- ``phi_lambda(u)``: 出来高・注文流の強度変調。**S4 では定義のみで消費されない**
  (L1 がスタブのため)。S7 で Hawkes のベースラインに組み込む。
- オーバーナイト: 引けと翌日の寄付の間に単一のギャップ・リターンを挿入する。

**正規化条件が phi_sigma と phi_lambda で違う** ことに注意:

- ``phi_sigma``: ``(1/T) ∫ phi^2 du = 1`` — 加算されるのは分散なので**二乗**の平均。
  phi の平均を 1 にすると Jensen の不等式で日次積分分散が目標を超える。
- ``phi_lambda``: ``(1/T) ∫ phi du = 1`` — 強度なので一乗。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..config import SESSION_SECONDS, Config
from ..rng import RNGRegistry

__all__ = [
    "ConstantCalendar",
    "SeasonalCalendar",
    "PerpCalendar",
    "build_calendar",
    "SESSION_SECONDS",
    "fourier_profile",
    "normalize_phi_sigma",
    "normalize_phi_lambda",
]


# ---------------------------------------------------------------------------
# phi の表現と正規化
# ---------------------------------------------------------------------------
def fourier_profile(
    u: np.ndarray | float,
    cos_coeffs: Sequence[float],
    sin_coeffs: Sequence[float],
    slope: float = 0.0,
) -> np.ndarray:
    """``1 + s(u-1/2) + sum_k (a_k cos 2pi k u + b_k sin 2pi k u)`` (正規化前)。

    定数項を 1 に固定してあるので、係数は「平均からの相対的な起伏」を表す。

    ★線形項 ``s(u-1/2)`` が必要な理由: **周期 Fourier だけでは
    ``phi(0) = phi(1)``、つまり寄付と引けの水準が必ず等しくなる**
    (cos(2πk·0) = cos(2πk·1) = 1、sin はどちらも 0)。実証的にはボラは寄付が最大で
    引けはそれより低く、出来高は逆に引けが最大なので、非周期の項がないと
    どちらの形も作れない。線形項は平均 0 なので定数項の解釈を壊さない。

    正の値を保つため、結果は下限 0.05 でクリップする (係数を強くしすぎたときに
    負の phi ができて sqrt や log が壊れるのを防ぐ)。
    """
    uu = np.asarray(u, dtype=np.float64)
    out = np.ones_like(uu)
    if slope:
        out = out + slope * (uu - 0.5)
    for k, (a, b) in enumerate(zip(cos_coeffs, sin_coeffs), start=1):
        ang = 2.0 * math.pi * k * uu
        if a:
            out = out + a * np.cos(ang)
        if b:
            out = out + b * np.sin(ang)
    return np.clip(out, 0.05, None)


def _grid(n: int = 20001) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def normalize_phi_sigma(
    cos_coeffs: Sequence[float], sin_coeffs: Sequence[float], slope: float = 0.0
) -> float:
    """``(1/T) ∫ (c*g)^2 du = 1`` を満たす正規化定数 c を返す。

    **二乗の平均を 1 にする** (指示書 §4.1)。加算されるのは分散なので二乗。
    phi の平均を 1 にすると Jensen の不等式で日次積分分散が目標を超える。
    数値積分は台形則で行い、生成側が診断に実測値を残す。
    """
    u = _grid()
    g = fourier_profile(u, cos_coeffs, sin_coeffs, slope)
    mean_sq = float(np.trapezoid(g**2, u))
    if mean_sq <= 0:
        raise ValueError("phi_sigma の二乗平均が 0 以下です")
    return 1.0 / math.sqrt(mean_sq)


def normalize_phi_lambda(
    cos_coeffs: Sequence[float], sin_coeffs: Sequence[float], slope: float = 0.0
) -> float:
    """``(1/T) ∫ c*g du = 1`` を満たす正規化定数 c を返す (強度なので一乗)。"""
    u = _grid()
    g = fourier_profile(u, cos_coeffs, sin_coeffs, slope)
    mean = float(np.trapezoid(g, u))
    if mean <= 0:
        raise ValueError("phi_lambda の平均が 0 以下です")
    return 1.0 / mean


# ---------------------------------------------------------------------------
class ConstantCalendar:
    """時間構造を持たないカレンダー (S0〜S3)。

    セッションを等間隔に分割した通し時刻グリッドを供給するだけで、季節性も
    ギャップも入れない。
    """

    name = "l0.constant"
    has_seasonality = False
    has_overnight = False

    def __init__(self, config: Config) -> None:
        self._config = config
        # ★時間軸の単一情報源 (S0-perp §4): equity では config.seconds_per_day ==
        # SESSION_SECONDS (23400.0) と同一 float なので経路は 1 bit も変わらない。
        # perp_clob では 86400.0 (24/7)。
        self._session_seconds = config.seconds_per_day
        self._step_seconds = self._session_seconds / config.steps_per_day

    # ------------------------------------------------------------------
    def session_seconds(self) -> float:
        return self._session_seconds

    def step_seconds(self) -> float:
        """シミュレーション格子の刻み (秒)。"""
        return self._step_seconds

    def n_days(self) -> int:
        return self._config.n_days

    def simulation_grid(self) -> np.ndarray:
        """L2 が価格を生成する時刻グリッド。

        ``t_i = i * step_seconds`` で ``i = 0 .. n_days * steps_per_day``。
        セッション境界の時刻は前日の最終点と翌日の始点が同一点として共有される
        (オーバーナイトが無いため)。S4 でギャップを入れるときは、価格グリッドは
        このまま (取引時間のみを刻む) にして、**日境界に単一のギャップ・リターンを
        挿入する** — グリッドに物理時間 17.5 時間を足すのではない。
        """
        n_points = self._config.total_steps + 1
        # 本番設定では 1 配列 936MB。`arange(...) * step` は中間配列をもう 1 本
        # 作るので、in-place で掛ける (値は同一)。
        grid = np.arange(n_points, dtype=np.float64)
        grid *= self._step_seconds
        return grid

    def intraday_position(self, t: np.ndarray) -> np.ndarray:
        """セッション内の相対位置 u ∈ [0, 1)。

        ★セッションをまたいで連続にしない。後場の寄付は「日の中盤」ではなく
        「セッションの開始」なので、u は必ずセッション内で 0 から始める
        (分割セッションを入れるときにここが効く)。
        """
        return np.mod(t / self._session_seconds, 1.0)

    def phi_sigma(self, t: float | np.ndarray) -> float | np.ndarray:
        """ボラの季節係数。S0〜S3 では恒等的に 1。"""
        if np.isscalar(t):
            return 1.0
        return np.ones_like(np.asarray(t, dtype=np.float64))

    def phi_lambda(self, t: float | np.ndarray) -> float | np.ndarray:
        """活動度の季節係数。S0〜S3 では恒等的に 1。"""
        return self.phi_sigma(t)

    # 後方互換 (S0〜S3 のコードとテストは phi() を使う)
    phi = phi_sigma

    def overnight_gaps(self) -> np.ndarray:
        """各セッション境界のギャップ (対数)。S0〜S3 では常にゼロ。"""
        return np.zeros(max(self._config.n_days - 1, 0), dtype=np.float64)

    def diagnostics(self) -> dict[str, Any]:
        return {"seasonality": False, "overnight": False}


class SeasonalCalendar(ConstantCalendar):
    """S4: 日内季節性 phi(u) とオーバーナイトを持つカレンダー。

    価格グリッドは ``ConstantCalendar`` と同一 (取引時間のみを刻む)。季節性は
    L2 が観測ボラに乗法で掛け、オーバーナイトは L2 が日境界に単一リターンとして
    挿入する。カレンダーはその**係数とギャップの供給元**に徹する。
    """

    name = "l0.seasonal"

    def __init__(self, config: Config, rng: RNGRegistry) -> None:
        super().__init__(config)
        self._rng = rng
        self.has_seasonality = config.enable_seasonality
        self.has_overnight = config.enable_overnight

        self._c_sigma = (
            normalize_phi_sigma(
                config.phi_sigma_cos, config.phi_sigma_sin, config.phi_sigma_slope
            )
            if config.enable_seasonality
            else 1.0
        )
        self._c_lambda = (
            normalize_phi_lambda(
                config.phi_lambda_cos, config.phi_lambda_sin, config.phi_lambda_slope
            )
            if config.enable_seasonality
            else 1.0
        )

    # ------------------------------------------------------------------
    def phi_sigma_of_u(self, u: np.ndarray | float) -> np.ndarray:
        """相対位置 u から phi_sigma を評価する (正規化済み)。"""
        if not self.has_seasonality:
            return np.ones_like(np.asarray(u, dtype=np.float64))
        cfg = self._config
        return self._c_sigma * fourier_profile(
            u, cfg.phi_sigma_cos, cfg.phi_sigma_sin, cfg.phi_sigma_slope
        )

    def phi_lambda_of_u(self, u: np.ndarray | float) -> np.ndarray:
        """相対位置 u から phi_lambda を評価する (正規化済み)。

        **S4 では消費されない** — L1 がスタブのため。S7 で Hawkes のベースラインに
        ``lambda(t) = phi_lambda(u_t) [mu Z_t + Hawkes 項]`` として組み込む。
        季節性を除去せずに Hawkes を当てると、活発な時間帯へのイベント集中を
        自己励起と誤認して**分岐比 n が系統的に過大推定**される
        (Filimonov-Sornette)。その対策の供給元がこれ。
        """
        if not self.has_seasonality:
            return np.ones_like(np.asarray(u, dtype=np.float64))
        cfg = self._config
        return self._c_lambda * fourier_profile(
            u, cfg.phi_lambda_cos, cfg.phi_lambda_sin, cfg.phi_lambda_slope
        )

    def phi_sigma(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.phi_sigma_of_u(self.intraday_position(np.asarray(t, dtype=np.float64)))

    def phi_lambda(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.phi_lambda_of_u(self.intraday_position(np.asarray(t, dtype=np.float64)))

    phi = phi_sigma

    def diagnostics(self) -> dict[str, Any]:
        cfg = self._config
        u = _grid()
        phi2 = self.phi_sigma_of_u(u) ** 2
        lam = self.phi_lambda_of_u(u)
        return {
            "seasonality": self.has_seasonality,
            "overnight": self.has_overnight,
            "session_type": cfg.session_type,
            "phi_sigma_norm_const": self._c_sigma,
            "phi_lambda_norm_const": self._c_lambda,
            # 正規化の検証値 (ゲート phi_normalization が見る)。
            "phi_sigma_sq_mean": float(np.trapezoid(phi2, u)),
            "phi_lambda_mean": float(np.trapezoid(lam, u)),
            "phi_sigma_sq_max_min_ratio": float(phi2.max() / phi2.min()),
            "phi_lambda_max_min_ratio": float(lam.max() / lam.min()),
            "phi_sigma_at_open": float(self.phi_sigma_of_u(0.0)),
            "phi_sigma_at_mid": float(self.phi_sigma_of_u(0.5)),
            "phi_sigma_at_close": float(self.phi_sigma_of_u(0.999)),
            "phi_lambda_at_open": float(self.phi_lambda_of_u(0.0)),
            "phi_lambda_at_close": float(self.phi_lambda_of_u(0.999)),
            "overnight_variance_share_target": cfg.overnight_variance_share,
        }


class PerpCalendar(ConstantCalendar):
    """perp_clob: 24/7 カレンダー (S0-perp)。

    1 日 = 86,400 秒の連続市場。セッション概念が無いので:

    - オーバーナイト・ギャップは**存在しない** (config が併用を弾く)
    - 「日」境界は UTC 0 時の集計上の区切りにすぎず、価格グリッドは
      ConstantCalendar と同じ通し等間隔グリッド (境界点を共有)
    - 季節性は S4-perp で日内 (UTC) + 週次 (168h) の 2 周期として実装する。
      S0-perp では φ ≡ 1 (weekly_profile_flat ゲートが平坦を確認する)

    実装は ConstantCalendar と同一で、セッション長だけが config 経由で
    86,400 秒になる — これが §1.2 の「分岐はレイヤー差し替えで行い、共通
    コアは分岐しない」の実体 (時間換算は全て session_seconds() を通る)。
    """

    name = "l0.perp_24_7"

    def weekly_position(self, t: np.ndarray) -> np.ndarray:
        """週内の相対位置 w ∈ [0, 1) (S4-perp の週次 φ が使う)。"""
        period = self._config.weekly_period_hours * 3600.0
        return np.mod(np.asarray(t, dtype=np.float64) / period, 1.0)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "seasonality": False,
            "overnight": False,
            "market_type": "perp_clob",
            "seconds_per_day": self._session_seconds,
            "weekly_period_hours": self._config.weekly_period_hours,
        }


def build_calendar(config: Config, rng: RNGRegistry) -> ConstantCalendar:
    """設定に応じた L0 を組み立てる (market_type で実装を差し替える §1.2)。"""
    if config.market_type == "perp_clob":
        del rng  # S0-perp の L0 は乱数を使わない
        return PerpCalendar(config)
    if config.enable_seasonality or config.enable_overnight:
        return SeasonalCalendar(config, rng)
    del rng  # S0〜S3 の L0 は乱数を使わない
    return ConstantCalendar(config)
