"""L1: 潜在活動度層 (S0 ではスタブ)。

最終系での役割
--------------
``lambda(t) = phi_lambda(t) * mu * Z_t + 多変量 Hawkes``。取引の到来速度そのものを
生成し、L3 の注文流を駆動する。カオス成分 chi_1 (強度変調) と chi_3 (分岐比変調)
の注入点でもある。

S0 での実装
-----------
定数強度。イベント生成は行わない (S0 には板もイベントも無いため)。
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..rng import RNGRegistry
from .l0_calendar import ConstantCalendar

__all__ = ["ConstantActivity", "HawkesActivity", "build_activity"]

#: S0 の名目強度 (イベント/秒)。値そのものは S0 では一切使われない。
#: S6 で L3 がイベント駆動になった時点で意味を持つ。
DEFAULT_INTENSITY: float = 1.0


class ConstantActivity:
    """自己励起のない定数強度。"""

    name = "l1.constant"

    def __init__(self, config: Config, calendar: ConstantCalendar) -> None:
        self._config = config
        self._calendar = calendar
        self._mu = DEFAULT_INTENSITY

    def intensity(self, t: float | np.ndarray) -> float | np.ndarray:
        """時刻 t での強度 ``phi(t) * mu``。S0 では phi = 1 なので定数。"""
        phi = self._calendar.phi(t)
        return self._mu * phi

    def branching_ratio(self) -> float | None:
        """Hawkes の分岐比。S0 には自己励起が無いので ``None``。

        ``0.0`` ではなく ``None`` を返すのは、自己励起が無効と「分岐比を
        推定したらゼロだった」を取り違えないため。検証側はこれを見て
        ``not_applicable`` を返す。
        """
        return None

    def event_times(self, t_start: float, t_end: float) -> np.ndarray:
        """区間内のイベント時刻。

        S0 の L3 はイベント駆動ではないため、この経路は使われない。呼ばれた場合は
        黙って空配列を返さず停止する。空を返すとイベントが 0 件だったという
        測定結果と区別がつかなくなるため。
        """
        raise NotImplementedError(
            f"L1 のイベント生成は S6 (板層の導入) で使い始め、S7 で Hawkes 化します。"
            f" 追加先: simchart/layers/l1_activity.py"
            f" (要求区間: [{t_start}, {t_end}])"
        )


class HawkesActivity:
    """S7: 符号対称な 6 次元 Hawkes 注文流の仕様の保持者。

    生成そのものは板カーネル (:mod:`.book_engine`) に融合されている。取消強度が
    板の生存注文数 ``δ0·N(t)`` に依存するため、板から切り離した「イベント時刻の
    事前生成」は原理的にできない (N(t) は板を進めないと判らない)。この層は
    パラメータの単一の出所であり、分岐比などの導出量をここで計算する。

    符号対称制約 (設計要件: Φ[買X→買Y] = Φ[買X→売Y]) の下では、6 次元系は
    型レベル 3 次元 Hawkes + 独立な等確率符号と厳密に等価で、6×6 ブロック
    行列 [[A/2, A/2], [A/2, A/2]] のスペクトル半径は 3×3 の A のそれに一致する。
    実装もその表現を使う (符号の相関構造 (11) は S8 のメタオーダーの仕事)。
    """

    name = "l1.hawkes"

    def __init__(self, config: Config, calendar: ConstantCalendar) -> None:
        self._config = config
        self._calendar = calendar

    # -- 導出量 --------------------------------------------------------
    def matrix(self) -> np.ndarray:
        """型レベル励起行列 a (3×3, 行=源, 列=先)。∫カーネル = a[x,y]。"""
        return np.asarray(self._config.hawkes_a, dtype=np.float64)

    def branching_ratio(self) -> float:
        """分岐比 n = ρ(a) (スペクトル半径)。"""
        return float(np.max(np.abs(np.linalg.eigvals(self.matrix()))))

    def betas_per_day(self) -> np.ndarray:
        """カーネル減衰率 [1/日]。日 = 1 立会 (session_seconds 秒)。"""
        session = self._calendar.session_seconds()
        tau = np.asarray(self._config.hawkes_tau_seconds, dtype=np.float64)
        return session / tau

    def weights(self) -> np.ndarray:
        return np.asarray(self._config.hawkes_weights, dtype=np.float64)

    def stationary_rates(self) -> np.ndarray:
        """定常イベントレート r = (I − aᵀ)⁻¹ μ [件/日] (MO計, LO計, CX計)。

        CX ベースラインは δ0·N̄ref。N̄ref は S6 本番の実測平均生存注文数
        (config.hawkes_nbar_ref = 取消数/日/δ ≈ 239)。2α/δ (= 600) を使っては
        ならない — あれは容量見積用の粗い上限で、実測の 2.5 倍ある。板の実際の
        N(t) が揺らぐぶんは実測との差になる — 目安であり厳密な予言ではない。
        """
        cfg = self._config
        mu_vec = np.array(
            [
                2.0 * cfg.hawkes_mu_mo,
                2.0 * cfg.hawkes_mu_lo,
                cfg.hawkes_delta0 * cfg.hawkes_nbar_ref,
            ]
        )
        return np.linalg.solve(np.eye(3) - self.matrix().T, mu_vec)

    def intensity(self, t: float | np.ndarray) -> float | np.ndarray:
        """ベースライン強度 φ_λ(t)·μ_total [件/日]。

        これは励起項を含まない。完全な λ(t) はイベント履歴の関数であり、
        シミュレーション本体か検証側の再構成 (validation/hawkes.py) でしか
        評価できない。
        """
        cfg = self._config
        mu_total = 2.0 * cfg.hawkes_mu_mo + 2.0 * cfg.hawkes_mu_lo
        return self._calendar.phi_lambda(t) * mu_total

    def event_times(self, t_start: float, t_end: float) -> np.ndarray:
        raise NotImplementedError(
            "S7 の Hawkes 生成は板カーネルに融合されている (取消強度が板の"
            " N(t) に依存するため分離不能)。イベント列は L3 の EventLog を参照。"
            f" (要求区間: [{t_start}, {t_end}])"
        )


def build_activity(
    config: Config, rng: RNGRegistry, calendar: ConstantCalendar
) -> ConstantActivity | HawkesActivity:
    # S12: χ₁/χ₃ は L1 の仕様 (活動度ベースライン・分岐比) を変調するが、
    # 実装は板カーネル側 (χ₁ は Z_total への畳み込み、χ₃ は n_t の sigmoid) —
    # S7 以降の生成は板カーネルに融合と同じ配置 (README の作法)。
    del rng  # 乱数は L3 側がレジストリから直接引く (l1.hawkes ストリーム)
    if config.enable_hawkes:
        return HawkesActivity(config, calendar)
    return ConstantActivity(config, calendar)
