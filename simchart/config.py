"""実行設定。

設計方針
--------
**全段階 (S0〜S13) のフラグを S0 の時点で宣言する。** 後から項目を足すと過去段階の
設定ファイルが読めなくなり、「どの段階までは正常だったか」を遡る回帰テストが成立
しなくなるため。未実装のフラグに ``True`` を渡した場合は暗黙 no-op にせず、
どの段階で実装されるかを明記した :class:`NotImplementedError` を送出する。

``Config`` 本体のフラグ一覧は指示書 §4 と一対一に対応する。検証スイートの推定器
設定 (バンド幅・スケール集合など) は :class:`ValidationConfig` に分離してある。
これはモデルの挙動ではなく「測定器の設定」であり、段階間で比較可能に保つために
やはり S0 で全部宣言しておく。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "Config",
    "ValidationConfig",
    "STAGES",
    "IMPLEMENTED_STAGES",
    "TRADING_DAYS_PER_YEAR",
    "UNIMPLEMENTED_FLAGS",
]

#: S0 から S13 までの段階名。
STAGES: tuple[str, ...] = tuple(f"S{i}" for i in range(14))

#: 現時点で実装が存在する段階。段階を進めるたびにここへ追加する。
IMPLEMENTED_STAGES: tuple[str, ...] = (
    "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11",
)

#: 年率ボラを 1 ステップ分に落とすときの営業日数。
TRADING_DAYS_PER_YEAR: int = 252

#: 1 立会日の長さ (秒)。6.5 時間。steps_per_day はこの長さを何分割するかを表す。
SESSION_SECONDS: float = 6.5 * 3600.0


# ---------------------------------------------------------------------------
# 未実装フラグの表: フラグ名 -> (実装段階, 内容, 追加先)
# ---------------------------------------------------------------------------
UNIMPLEMENTED_FLAGS: dict[str, tuple[str, str, str]] = {
    "enable_chaos_lambda": ("S12", "カオス的強度変調 chi_1", "simchart/layers/l1_activity.py"),
    "enable_chaos_branching": ("S12", "カオス的分岐比変調 chi_3", "simchart/layers/l1_activity.py"),
    "enable_jump_hawkes": ("S11d", "ジャンプ自己励起 (任意 — S11a〜c 完了後に要否判断 §7)", "simchart/layers/l2_price.py"),
}

#: 実装済みの機能フラグ。UNIMPLEMENTED_FLAGS から行を移すときはこちらに追記する。
#: (テストが「全 bool フラグはどちらかの台帳に載っている」ことを強制し、
#:  新フラグの登録漏れ = 暗黙 no-op を構造的に防ぐ)
IMPLEMENTED_FLAGS: tuple[str, ...] = (
    "enable_msm",  # S1
    "enable_slow_ou",  # S1
    "enable_rough",  # S2
    "enable_jump",  # S3
    "enable_leverage",  # S3
    "enable_seasonality",  # S4
    "enable_overnight",  # S4
    "enable_chaos_vol",  # S5
    "enable_book",  # S6
    "book_allow_inspread",  # S6 (板の従属 bool — improvement の許可)
    "book_debug_invariants",  # S6 (板の従属 bool — 毎イベント検証)
    "enable_hawkes",  # S7
    "enable_metaorder",  # S8
    "meta_sequential",  # S8 (従属 bool — 逐次版の相互検証モード)
    "enable_iceberg",  # S8
    "book_iceberg_refill_tail",  # S8 (従属 bool — 補充の優先順位規則)
    "enable_queue_reactive",  # S9
    "enable_uncertainty_zones",  # S9 (fallback — 常用禁止 §8.2)
    "enable_feedback",  # S11
)

#: フラグ以外 (数値パラメータ) の未実装条件。
_UNIMPLEMENTED_NUMERIC: dict[str, tuple[str, str, str, float]] = {}


def _not_implemented(name: str, stage: str, what: str, where: str, value: Any) -> NotImplementedError:
    return NotImplementedError(
        f"{name}={value!r} は段階 {stage} で実装される予定であり、現在の実装段階 "
        f"{IMPLEMENTED_STAGES[-1]} では未実装です。"
        f" 内容: {what} / 追加先: {where}。"
        f" 未実装フラグを暗黙に無視すると結果が静かに嘘になるため、ここで停止します。"
    )


@dataclass(frozen=True)
class ValidationConfig:
    """検証スイートの推定器設定。

    ここに入るのは「測定器の設定」であってモデルの設定ではない。段階をまたいで
    同じ設定で測ることで metrics.json が回帰テストとして機能する。

    Attributes
    ----------
    primary_bar_sec:
        ゲート判定に用いる基準リターンの粒度 (秒)。1 分を既定とする。理由は 3 つ。
        (1) マイクロストラクチャー研究の慣行的な解析粒度である。
        (2) 既定設定 (500 日 x 23400 秒) で約 195,000 本のリターンが取れ、
            尖度の標準誤差が約 0.011 になるため §8 の閾値 [2.7, 3.3] が
            「偶然では落ちない・壊れていれば落ちる」本物の検定になる。
        (3) シミュレーション格子 (1 秒) から十分離れているので、S9 で
            uncertainty zones を入れても粒度そのものが壊れない。
    scales_sec:
        スケール依存量 (尖度・zeta_q・signature plot) を測る粒度の集合 (秒)。
    min_obs_for_gate:
        スケール別指標をゲート判定に使う最小標本数。これを下回るスケールは
        「記録のみ」とする。有限標本誤差で閾値を割るのを防ぐため。
    gph_bandwidth_exponent:
        GPH 推定量のバンド幅 m = N**exp。0.65 を既定とする。古典的な 0.5 では
        既定設定で d の標準誤差が 0.030 となり、§8 のゲート |d| < 0.05 が
        1.6 標準誤差しかない実質無意味な検定になってしまうため。
        0.65 なら標準誤差は約 0.011 で、閾値は約 4.6 標準誤差に相当する。
    """

    primary_bar_sec: int = 60
    scales_sec: tuple[int, ...] = (1, 2, 5, 10, 30, 60, 120, 300, 600, 900, 1800, 23400)
    min_obs_for_gate: int = 10_000

    acf_max_lag: int = 100
    acf_abs_max_lag: int = 1000
    ljung_box_lags: tuple[int, ...] = (1, 5, 10, 20, 50)
    ljung_box_primary_lag: int = 20

    gph_bandwidth_exponent: float = 0.65
    gph_bandwidth_profile: tuple[float, ...] = (0.5, 0.6, 0.65, 0.7, 0.8)
    lw_bandwidth_exponent: float = 0.65

    hill_k_fracs: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10)
    hill_primary_k_frac: float = 0.05
    qq_n_points: int = 201

    vr_qs: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    zeta_qs: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    adf_maxlag: int = 10

    micro_max_lag: int = 200
    micro_fit_lag_range: tuple[int, int] = (5, 200)
    rng_probe_draws: int = 1000

    # --- S1 追加: 日次系列の測定器設定 ---
    # S1 の長期記憶ゲート (gph_d / absr_acf_powerlaw) は日次集計 |r| で測る。
    # MSM の成分レンジ (1〜500 日) が日内スケールより長いため、日内粒度では
    # 記憶もマルチフラクタル性も見えない。
    daily_acf_max_lag: int = 150
    daily_powerlaw_lag_range: tuple[int, int] = (1, 100)
    daily_gph_bandwidth_exponent: float = 0.65
    daily_scales_days: tuple[int, ...] = (1, 2, 5, 10, 20, 50)
    daily_min_obs_for_gate: int = 250
    daily_zeta_qs: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

    # --- S1 追加: ボラ正規化のアンサンブル断面検証 ---
    # E[sigma^2] や Var(log sigma) は遅い成分の自己相関のせいで 1 経路の時間平均では
    # ±15〜20% ゆらぐ (5000 日でも実効独立標本 ~30)。期待値 E[·] の検証は定常初期化
    # した独立標本の断面で行う。20 万本で E[sigma^2] の標準誤差 ≈ 0.22%。
    ensemble_n_paths: int = 200_000

    # --- S2 追加: 粗さ指数 H の測定器設定 ---
    # ★H の測定窓 (5 分〜4 時間) と GPH の測定窓 (1〜100 日) を**重ねない**こと。
    # 重ねると互いに汚染し、どちらの推定も信用できなくなる (S2 指示書 §7)。
    # H はラフ成分が支配する帯域、GPH は MSM/OU が支配する帯域で測る。
    rough_h_scales_seconds: tuple[int, ...] = (300, 600, 1200, 1800, 3600, 7200, 14400)
    rough_h_qs: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    #: RV 側の H 推定に使う実現分散の窓 (秒)。30 分ビン。
    rv_window_seconds: int = 1800
    #: ボラ増分 ACF の最大ラグ (ラフグリッドのステップ数)。
    vol_incr_acf_max_lag: int = 60

    # --- S2 追加: S1 からの不変性トレランス (compare S1 S2 がゲート) ---
    # S2 の合否は「何が増えたか」ではなく「何が変わらなかったか」で決まる。
    inv_tol_gph_d_abs: float = 0.03  # ★最重要。動いたらスケール分離失敗 (指示書 §9)
    inv_tol_powerlaw_gamma_rel: float = 0.10
    inv_tol_acf_profile_mean_abs: float = 0.02  # 日次ラグ 10〜100 の平均 |Δrho|
    inv_tol_kurtosis_daily_increase: float = 0.5
    inv_tol_zeta_c2_abs: float = 0.005

    # --- S1 追加: 時間スケール不変性の対照解像度とトレランス ---
    # 同一シードなら MSM 切替過程は解像度に依存せず完全一致するので、残る差は
    # 拡散乱数と OU 乱数の実現差のみ。トレランスはそのペア差の実測分布
    # (5000 日 x 6 シード: 尖度相対差 max 0.19 / gph max 0.044 / acf1 max 0.051 /
    # var max 0.009) に余裕を載せた値。per-step 切替確率型の欠陥は切替頻度が
    # 解像度比 (60 倍) で変わり統計が桁で動くため、この幅でも検出力は落ちない。
    scale_invariance_steps_per_day: int = 390
    si_tol_kurtosis_rel: float = 0.25
    si_tol_gph_d_abs: float = 0.10
    si_tol_acf1_abs: float = 0.07
    si_tol_var_logvol_abs: float = 0.025

    def __post_init__(self) -> None:
        if self.primary_bar_sec <= 0:
            raise ValueError("primary_bar_sec は正整数である必要があります")
        if not self.scales_sec:
            raise ValueError("scales_sec が空です")
        if any(s <= 0 for s in self.scales_sec):
            raise ValueError("scales_sec の要素は正整数である必要があります")
        if self.min_obs_for_gate < 1:
            raise ValueError("min_obs_for_gate は 1 以上である必要があります")
        if self.ljung_box_primary_lag not in self.ljung_box_lags:
            raise ValueError("ljung_box_primary_lag は ljung_box_lags に含まれている必要があります")
        if not (0.0 < self.gph_bandwidth_exponent < 1.0):
            raise ValueError("gph_bandwidth_exponent は (0, 1) の範囲である必要があります")
        if not (0.0 < self.hill_primary_k_frac < 1.0):
            raise ValueError("hill_primary_k_frac は (0, 1) の範囲である必要があります")
        lo, hi = self.micro_fit_lag_range
        if not (1 <= lo < hi):
            raise ValueError("micro_fit_lag_range は 1 <= lo < hi である必要があります")
        dlo, dhi = self.daily_powerlaw_lag_range
        if not (1 <= dlo < dhi):
            raise ValueError("daily_powerlaw_lag_range は 1 <= lo < hi である必要があります")
        if dhi > self.daily_acf_max_lag:
            raise ValueError("daily_powerlaw_lag_range の上限が daily_acf_max_lag を超えています")
        if self.ensemble_n_paths < 1000:
            raise ValueError("ensemble_n_paths は 1000 以上である必要があります (それ未満では検定力が無い)")
        if self.scale_invariance_steps_per_day < 1:
            raise ValueError("scale_invariance_steps_per_day は正整数である必要があります")


@dataclass(frozen=True)
class Config:
    """シミュレータ全体の設定。指示書 §4 のフラグ一覧をそのまま保持する。"""

    # --- 実行 ---
    seed: int = 42
    n_days: int = 500
    steps_per_day: int = 23400  # 1 秒刻み・6.5 時間
    stage: str = "S0"

    # --- L0 ---
    enable_seasonality: bool = False  # S4
    enable_overnight: bool = False  # S4

    # --- L2 ---
    sigma_bar: float = 0.20  # 年率
    mu_drift: float = 0.0
    p0: float = 100.0  # 初期価格水準。リターン系の統計には一切影響しない
    enable_msm: bool = False  # S1 (実装済み)
    enable_slow_ou: bool = False  # S1 (実装済み)
    enable_rough: bool = False  # S2

    # --- L2 / S1: 確率ボラのパラメータ ---
    # m0 は直接指定しない。分散配分 vol_var_target_msm から solve_m0() で逆算する
    # (S2/S5 で予算を再配分するときに手計算し直さないため)。見かけの長期記憶 d も
    # (m0, b, k) から創発する量であり、パラメータには存在しない。
    msm_k: int = 10  # MSM 成分数
    msm_b: float = 2.0  # 周波数比 (gamma_i = gamma_1 * b^(i-1))
    msm_gamma1_per_day: float = 0.002  # 最遅成分の切替強度 [1/日]。物理時間定義 (§7)
    vol_var_target_msm: float = 0.125  # Var(log sigma) の MSM 配分 (最終予算 0.25 の 50%)
    vol_var_target_slow: float = 0.050  # Var(log sigma) の緩慢 OU 配分 (20%)
    ou_half_life_days: float = 30.0  # 緩慢 OU の半減期 [日] (推奨 20〜60)
    #: Var(log sigma) の最終予算 (全 13 段階の合計)。分散シェアの分母はこれ。
    #: S2 時点の配分後の残り 0.050 は S5 chi_2 の枠。
    vol_var_budget_total: float = 0.25

    # --- L2 / S2: ラフボラティリティ (fractional OU, H ~ 0.1) ---
    # 非定常な fBm/Volterra ではなく**定常な fOU** を使う (非定常だと分散の増大が
    # 低周波の見かけの長期記憶として GPH に混入する — S2 最頻の事故)。
    # eta_r は直接指定せず、分散配分 vol_var_target_rough から solve_eta_rough()
    # で逆算する。ラフ成分は専用の物理グリッド (rough_grid_seconds) 上で生成され、
    # steps_per_day と独立 — 時間スケール不変性が構造的に保たれる。
    #: Hurst 指数 (指示書の許容範囲 0.08〜0.15)。既定 0.08 の理由: H_latent ゲートは
    #: **合成後の log sigma** で測るが、測定窓 (5 分〜4 時間) には MSM 最速成分の
    #: 切替も混入し、H の測定値を +0.05〜0.06 系統的に押し上げる (実測)。入力 0.10
    #: だと測定 0.150〜0.154 でゲート上限に乗るため、範囲内の 0.08 を入力して
    #: 測定 ~0.13 に置く。gph_d はこの選択に依存しない (ペア実験で ±0.01 不変)。
    rough_hurst: float = 0.08
    #: fOU の平均回帰半減期 [日]。**1 日より長くしないこと** — MSM 最速成分
    #: (時定数 ~1 日) の帯域に食い込み、長スケールの記憶 (gph_d) を汚染する。
    rough_half_life_days: float = 0.75
    vol_var_target_rough: float = 0.025  # Var(log sigma) 配分 (最終予算 0.25 の 10%)
    rough_grid_seconds: float = 60.0  # ラフ成分の解像度。これ以下ではボラは一定

    # --- L2 / S3: ジャンプ (Kou 二重指数、ボラ変調強度) ---
    # サイズは**べき則ではなく指数** (Kou): べき則は集計してもテール指数が変わらず
    # 集計正規性 (⑱) を阻害する。指数なら集計で相対テールが薄れ、Hill α が
    # スケールとともに上昇する。α ≈ 3〜5 は有限標本の性質として狙う (§3.2)。
    # ★λ_jump_per_year の指示書目安 (20〜60/年) は η_d 15〜25 と JV share 5〜15%
    # ゲートと算術的に両立しない (λ=40, η_d=18 → JV 83%)。critical ゲートを優先し
    # λ を下げてある。経緯は README。
    jump_p_up: float = 0.42  # 上昇ジャンプの確率 (p < 0.5 で負の歪度)
    jump_eta_up: float = 35.0  # 上昇側の減衰率 (平均 +1/η_u)。**>1 必須** (E[e^J] 発散)
    jump_eta_down: float = 22.0  # 下落側の減衰率 (η_d < η_u で下落が大きい)
    jump_lambda_per_year: float = 2.5  # 基準強度 [回/年]
    jump_vol_exponent: float = 1.0  # λ(t) = λ0 (σ_t/σ̄)^ρ_J — ボラ状態でクラスター
    jump_intensity_cap: float = 10.0  # λ(t)/λ0 の上限 (対数正規 σ の裾で発散させない)
    #: 総二次変動に占めるジャンプ寄与の設計値。拡散側は σ̄_diff = σ̄ √(1-share)
    #: に縮小し、年率の総 QV を σ̄² に保つ (§7)。Var(log σ) 予算とは独立。
    jump_qv_share_target: float = 0.10

    # --- L2 / S3: レバレッジ (2 チャンネル) ---
    # 短期 (0〜1 日) はラフ成分の駆動 fGn と、長期 (5〜30 日) は緩慢 OU の駆動と
    # 相関させる。MSM とは相関させない (純ジャンプ過程で相関の定義が不自然)。
    # 実装は Brownian bridge 分解 (§6) — 共通ショックの単純加算はセル内に正の
    # 自己相関を作り ② を壊すため厳禁。
    leverage_rho_rough: float = -0.70  # corr(セル集計 z, fGn innovation) [推奨 -0.6〜-0.8]
    leverage_rho_slow: float = -0.30  # corr(z, OU 駆動) [推奨 -0.2〜-0.4]
    #: 中速レバレッジ成分 — **既定は無効 (var=0)** (2026-08-20 オペレータ裁定)。
    #: 経緯: per-step 相関では corr(r_t, RV_{t+1}) の理論上限が ~-0.06 で指示書の
    #: 帯 [-0.28, -0.16] に届かず、中速成分 (日次グリッド OU、駆動を前日の日次
    #: 集計リターンと相関) を追加しても実測上限は -0.14。しかも中速の分散は
    #: vol_var_target_slow の内数の取り合いで、**lev を強めるほど 10〜100 日帯域の
    #: 記憶が削れて gph_d が S2 から最大 -0.15 動く** (③ の破壊)。③ の保全を優先し
    #: 中速は無効化、lev ゲートは実測整合帯 [-0.08, -0.005] に変更。レバレッジ
    #: 水準の残りは S10 の板側チャンネルに委ねる (§5.3 の「弱い側を狙う」)。
    #: 機構は実装済みなので、将来再配分する場合は leverage_mid_var > 0 にする。
    leverage_mid_half_life_days: float = 5.0
    leverage_mid_var: float = 0.0  # 0 = 無効。>0 で slow の内数を再配分
    leverage_rho_mid: float = -0.80

    # --- L0 / S4: 日内季節性 (phi) ---
    # 対象市場は**米国株相当の連続単一セッション** (6.5 時間) — 2026-08-20 指定。
    # S0 以来 SESSION_SECONDS = 23400 なので、既存 S0〜S3 の基準値がそのまま使える。
    # 昼休みのある市場 (W 字) を扱うときは session_type を "split" にして
    # セグメント定義を足す (l0_calendar.py に拡張点を用意してある)。
    session_type: str = "continuous"  # "continuous" | "split" | "24h"
    #: ★線形項の傾き。**周期 Fourier だけでは phi(寄付) = phi(引け) が強制される**
    #: (cos(2πk·0) = cos(2πk·1))。寄付と引けで水準を変えるには非周期の項が要る。
    #: phi_sigma は負 (寄付 > 引け)、phi_lambda は正 (引けのクロージング・
    #: オークションが最大) にする。
    phi_sigma_slope: float = -0.2219
    phi_lambda_slope: float = 0.9374
    #: phi_sigma(u) の Fourier 係数。u はセッション内の相対位置 (0〜1)。
    #: ★正規化は **(1/T)∫phi^2 du = 1** (phi ではなく phi^2 の平均)。加算されるのは
    #: 分散なので、phi の平均を 1 にすると Jensen で日次積分分散が目標を超える。
    #: 正規化定数は生成時に数値積分で求める (係数を直接いじらない)。
    #: 既定は寄付最大 (1.484)・日中最小 (0.711)・引け中高 (1.269) の U 字。
    #: 係数は「phi^2 の起伏比 = 4.5 (指示書 §4.4 の 3〜6 の中央)」を満たすよう
    #: 数値的に逆算した値 (cos 1 次で U 字、cos 2 次で形を整え、slope で寄付>引け)。
    phi_sigma_cos: tuple[float, ...] = (0.3439, 0.0777, 0.0)
    phi_sigma_sin: tuple[float, ...] = (0.0, 0.0, 0.0)
    #: phi_lambda(u) の Fourier 係数。**phi_sigma と同一にしないこと** — 実証的に
    #: 出来高は引け (クロージング・オークション) が最大でボラより顕著。
    #: 正規化は (1/T)∫phi_lambda du = 1 (強度なので一乗)。
    #: ★S4 では**定義のみ**で消費されない (L1 がスタブのため)。S7 で使う。
    #: 既定は寄付 1.406・日中最小 0.375・引け最大 2.344 (起伏比 7.0、§4.4 の
    #: 4〜10 の中央)。ボラより起伏が大きく、最大が引けに来るのが出来高の特徴。
    phi_lambda_cos: tuple[float, ...] = (0.7499, 0.1250, 0.0)
    phi_lambda_sin: tuple[float, ...] = (0.0, 0.0, 0.0)

    # --- L0 / S4: オーバーナイト ---
    # ★物理時間比例 (17.5h/6.5h) では作らない。取引の無い時間帯は情報時計が
    # ほとんど進まないので、**別レジームとして c_ON を直接指定**する (指示書 §6.3)。
    #: クローズ・トゥ・クローズ分散に占める ON ギャップの寄与 (設計値)。
    #: 手元に実データが無いため文献値 (米国株 15〜25%) の中央を設定値として使い、
    #: ゲートは「実測が**この設定値**と一致するか」を見る (実装の検定であって
    #: 市場の真値の主張ではない — 2026-08-20 オペレータ承認)。
    overnight_variance_share: float = 0.20
    #: ON ジャンプの発生確率 (日あたり)。ギャップは実質ジャンプ的なので日中
    #: (実効 ~4.2/年 = 1.7%/日) より高い。
    overnight_jump_prob: float = 0.06
    #: ON 分散に占めるジャンプ寄与。**サイズ倍率ではなく分散シェアで指定する** —
    #: 倍率で指定すると Kou の E[J²] が ON の分散予算と噛み合わず、実測シェアが
    #: 目標の 3 倍になる (実際にそうなった)。S3 の Kou 形状 (p_up, eta 比) を保った
    #: まま、このシェアを満たすよう eta のスケールを逆算する。
    overnight_jump_variance_share: float = 0.35
    enable_jump: bool = False  # S3
    enable_leverage: bool = False  # S3

    # --- L2 / S5: 決定論的カオス成分 chi_2 (Mackey-Glass) ---
    # ★目的は統計的リアリズムではなく**ボラ・レジームの決定論的な再現性と制御** —
    # 同一のカオス初期値なら、シード (確率成分) が違っても同じレジーム構造が現れる。
    # 乱数を一切消費しない (RNG ストリームは S4 から不変)。
    enable_chaos_vol: bool = False  # S5  (chi_2)
    chaos_system: str = "mackey_glass"
    #: 遅延 tau。17 で相関次元 ~2.1 の低次元カオス (実測: Lyapunov +0.0071/単位、
    #: D2 1.85、0-1 test K 0.97)。滑らかな不規則振動 = 緩慢成分の搬送波という
    #: 役割に合う (指示書 §3.1 の推奨既定)。
    chaos_tau_delay: float = 17.0
    chaos_beta: float = 0.2
    chaos_gamma: float = 0.1
    chaos_n_exponent: float = 10.0
    #: RK4 の固定ステップ (系固有単位)。tau/dt は整数必須 (遅延値を履歴グリッドに
    #: 載せるため)。適応ステップは禁止 — 局所誤差推定が軌道を環境依存にする (§7)。
    chaos_dt: float = 0.1
    #: 初期条件 = t<=0 の**定数履歴**。遅延方程式の初期値は関数 (履歴全体) なので、
    #: スカラー 1 つで完全に指定できる形にしてある (§7: config に明示)。
    chaos_ic: float = 1.2
    chaos_burn_in_units: float = 1000.0
    #: ★時間スケール写像: 1 系固有単位 = 何市場日か。MG(17) のスペクトルピークは
    #: 実測 49.65 単位なので、ピークを P 日に置くには s = P/49.65。値はアブレーション
    #: (同一シードで chi on/off の gph_d 差) で確定する — 指示書 §4.1 の訂正どおり
    #: 日次スケールは厳禁 (S4 の季節性と区別がつかなくなる)。
    chaos_days_per_unit: float = 0.6042
    #: chi_2 の分散配分。S1 から予約されていた枠 (最終予算 0.25 のうち 0.050 = 20%)。
    #: ★既存成分 (0.125/0.050/0.025) は変更しない — S5 は加算であって再配分ではない。
    vol_var_target_chaos: float = 0.050
    #: 周辺分布の扱い (§3.2)。"standardize" = 平均 0 分散 1 のみ (案 A、軌道保持)。
    #: "ecdf_normal" = 経験 CDF で正規に写像 (案 B、A が周辺分布ゲートで落ちたら)。
    #: どちらも決定論的で再現性を損なわない。
    chaos_normalization: str = "standardize"
    #: 生成物キャッシュの置き場 (再現性の証拠。ロード時に必ずハッシュ照合される)。
    chaos_cache_dir: str = "cache"

    # --- L1 / S7: 多変量 Hawkes 注文流 ---
    # ★6 次元 {成行,指値,取消}×{買,売} に**符号対称制約** Φ[買X→買Y]=Φ[買X→売Y] を
    # 課す (指示書 §3.1)。⑪ 符号 ACF は S8 のメタオーダー分割の担当で、Hawkes の
    # 交差励起に符号相関を持たせると二重計上になり γ・β の整合が崩れる。この制約の
    # もとで 6D は「型レベル 3D Hawkes + iid 符号」と厳密に等価 (6×6 ブロック行列
    # [[A/2,A/2],[A/2,A/2]] のスペクトル半径 = ρ(A)) — 実装は 3D で行う。
    enable_hawkes: bool = False  # S7
    #: 励起行列 a[X][Y] = 型 X のイベント 1 件が生む型 Y の子孫の期待数
    #: (両サイド合算)。ρ(a) = 0.830 (目標帯 0.80〜0.85、S12 で χ₃ が変調するまで固定)。
    #: 構造は指示書 §4.3 の目安 (対角支配 + 成行→取消 = 約定を見て指値を引く挙動 =
    #: ⑬ の源)。値は scripts/calibrate_s7_hawkes.py の出力 — 定常レートを **S6 の
    #: 実測** r = (1800, 3000, 1195)/日 に保つ mu = (I-a^T)r > 0 の制約下で較正。
    #: ★取消レートを δ·N̄ = 4500/日 と仮定した初版較正は誤り (取消数は指値流入を
    #: 超えられない。実測 N̄ = 239)。誤った r で走らせると励起駆動の取消が板を
    #: 食い尽くし、42% の時間で片側が空になりミッドが窓から逸脱した (実測)。
    hawkes_a: tuple[tuple[float, ...], ...] = (
        (0.6722, 0.4608, 0.1149),
        (0.0739, 0.4733, 0.1077),
        (0.0807, 0.2491, 0.2154),
    )
    #: 指数和カーネルの時定数 [秒]。最遅 300 秒 (指示書 §3.2 の上限 1 時間の内側) —
    #: 日次に伸ばすと S10 で MSM の帯域と競合して ③ を壊す。Hawkes は秒〜分の
    #: 反応連鎖の担当で、日次以上は L2 (と S10 の Z_t) の担当。
    hawkes_tau_seconds: tuple[float, ...] = (0.5, 10.0, 300.0)
    hawkes_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    #: ベースライン [件/日/側]。mu = (I - a^T) r から逆算 (r = S6 実測 1800/3000/1195)。
    #: ベースラインシェアは MO/LO 15.1%・CX 34.1% (残りが励起由来)。
    hawkes_mu_mo: float = 135.95
    hawkes_mu_lo: float = 226.49
    #: 取消のベースラインは**各注文独立ハザード δ0·N(t) を維持**する (S6 の構造)。
    #: 定数ベースラインに置き換えると板の復元力が消えて N がランダムウォークし
    #: 500 日で板が漂流する。励起は加法なので n の会計は厳密のまま。これは S9 が
    #: 禁じる「板の状態への戦略的依存」ではなく S6 から継続する簿記構造 (README)。
    #: δ0 = mu_cx / N̄ref。N̄ref は S6 本番 500 日の実測平均生存注文数。
    hawkes_delta0: float = 1.7061
    hawkes_nbar_ref: float = 238.95  # = 597386 取消 / 500 日 / (δ=5)
    #: バーストガード (指示書 §5.3): 総強度の上限 (非励起ベースライン最大値の倍数) と
    #: 1 日あたり最大イベント数。発動はカウンタに記録され、ゲートが頻度 < 0.01% を課す。
    hawkes_intensity_cap_mult: float = 50.0
    hawkes_daily_event_cap: int = 500_000
    enable_chaos_lambda: bool = False  # S12 (chi_1)
    enable_chaos_branching: bool = False  # S12 (chi_3)

    # --- L3 / S6: ZI 板 (zero-intelligence、Smith et al. 2003 ベースライン) ---
    # ★κ=0 で L2 と完全に切り離す。この段階の価格には ①③④⑧⑯⑱ は現れない —
    # それが正しい状態 (指示書 §0)。L2 の性質が観測に現れるのは S10 の結合から。
    enable_book: bool = False  # S6
    #: ティックサイズ (価格単位)。p0=100 で 0.01 = 1bp。**small tick レジーム**
    #: (スプレッド 2〜5 ティック、板構造が観測可能)。large tick は板の別世界
    #: (キュー長が全て) なので S9 で別 config として分岐する (指示書 §7)。
    tick_size: float = 0.01
    #: 成行注文の片側到着率 [件/日]。符号は 50/50 iid (S6)。
    #: ★レートはスイープで確定した (2026-08-21): スプレッド中央値 3 tick・P95 8・
    #: デプスピーク lvl 4・枯渇ゼロ・60 日安定。μ/α や δ を上げすぎると板が
    #: **崩壊する** (μ/α=1, δ=10 でスプレッド 43,000 tick を実測 — 補充が枯渇に
    #: 追いつかない相転移がある)。Smith の次元解析 (spread ∝ μ/α, depth ∝ α/δ) は
    #: 出発点で、水準は実測で合わせた。
    book_mu_mo: float = 900.0
    #: 指値注文の片側到着率 [件/日]。
    book_alpha_lo: float = 1500.0
    #: **板上の各注文が独立に**取り消される率 [1/日]。
    book_delta_cancel: float = 5.0
    #: 配置べき則 P(Δ) ∝ (Δ+Δ0)^-(1+μ) の指数 (指示書 §6.2 の帯 [0.6, 1.5])。
    book_mu_place: float = 0.9
    book_place_offset: float = 3.0  # Δ0
    #: 配置の最大距離 (打ち切り)。裾を切らないと遠方に無駄な注文が溜まる。
    book_max_place_ticks: int = 200
    #: スプレッド内 improvement (Δ<0) を許すか。小ティック板ではこれが無いと
    #: スプレッドを狭める機構が存在せず発散する (成行が best を食う一方になる)。
    book_allow_inspread: bool = True
    #: improvement の最大深さ (実際は spread-1 で必ず打ち切られる)。
    book_inspread_cap: int = 10
    #: サイズ分布: w·ラウンドロット + (1-w)·Pareto (指示書 §6.3)。単位はロット。
    book_size_round_weight: float = 0.7
    book_size_pareto_alpha: float = 2.3
    book_lot_values: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    book_lot_probs: tuple[float, ...] = (0.45, 0.25, 0.20, 0.10)
    #: 板スナップショットの間隔 (秒) と記録レベル数。
    book_snapshot_interval_sec: float = 60.0
    book_snapshot_levels: int = 10
    #: イベントごとに記録する best±N ティックのデプス (指示書 §5.3)。
    book_depth_ticks: int = 10
    #: ウォームアップ (統計収集から除外する日数、指示書 §8.1)。
    book_burn_in_days: float = 5.0
    #: 初期化: best±init_levels の各レベルに init_size ロットの種注文を置く。
    book_init_levels: int = 30
    book_init_size: float = 20.0
    #: 板の絶対ティック窓の半幅。ZI ミッドはランダムウォークするので、端に達したら
    #: 黙って詰まらず明示的に失敗する。**窓は価格が正に留まる範囲に制限する**
    #: (p0=100, tick=0.01 で ±8000 tick = ±$80。健全な ZI 板の 500 日ウォークの
    #: SD ~数百 tick に対し十分で、板が崩壊した場合は窓逸脱として顕在化する)。
    book_window_half_ticks: int = 8000
    #: 毎イベントの完全検証 (テスト用。本番は 5 万イベントごとの抜き取り §9)。
    book_debug_invariants: bool = False

    # --- L3 / S8: メタオーダー分割 (⑪ 約定符号の長期記憶) ---
    # Hawkes が「いつ」を決め、メタオーダーは「どちらの符号か」だけを決める
    # (指示書 §3.1 の役割分離 — タイミング統計と分岐比は S7 から不変であること)。
    enable_metaorder: bool = False  # S8
    #: Pareto 長さの裾指数。★定義域は開区間 (1, 2) — α≤1 は E[N] 発散で
    #: プールが定常にならず、α≥2 は Var(N) 有限で長期記憶が消える (指示書 §4.3)。
    #: 符号 ACF の減衰指数は γ = α − 1 (§4.2)。
    meta_alpha: float = 1.6
    #: 最小子注文数。γ の理論対応が最も素直な 1 を既定とする (§11 診断)。
    #: 長さは N = floor(N_min·u^{-1/α}) で引く — 離散裾 P(N≥n) = (N_min/n)^α が厳密。
    meta_n_min: int = 1
    #: 成行のうちメタオーダー由来の割合 (残り 1−ψ は iid 50/50 のノイズ)。
    #: ψ は C(1) の**水準**を動かし、減衰指数 γ は動かさない (§4.4 — 別々に同定)。
    meta_psi: float = 0.6
    #: Poisson 到着の供給率 ρ = λ_meta·E[N] / (ψ·λ_MO)。
    #: ★厳密に 1 にしてはならない: 子需要と供給が釣り合う臨界負荷では、E[N²]=∞
    #: (α<2) のプール占有が帰無再帰的にさまよい「定常」ゲートが成立しない
    #: (M/G/1 の標準的事実)。1 未満に置き、不足分は空プール時の即時生成
    #: (下記) が需要駆動で埋める — 実現フローは ψ·λ_MO に厳密一致する。
    #: ★ρ は C(1) の水準レバー (プール平均 A を決め、C(1) ≈ ψ²·f(A) — §4.4)。
    #: 60 日プローブ実測: ρ=0.85 → A中央値17・C(1)=0.075 / ρ=0.7 → 8・0.105 /
    #: **ρ=0.5 → 3・0.136 (採用: 帯 [0.05,0.20] の中央、複数執行者の実態も維持)** /
    #: ρ=0.3 → 1・0.167 (プールが逐次版に退化するので不採用)。
    meta_supply_ratio: float = 0.5
    #: 子注文を出そうとしてプールが空だったとき、新規メタオーダーを即時生成して
    #: それを使う (発生数はカウンタに記録)。これが供給の調整弁になる。
    meta_pool_cap: int = 100_000
    #: 相互検証モード (§3.4): Poisson 到着を止め、常に 1 本だけを最後まで実行して
    #: から次を生成する逐次版。γ = α−1 はこちらでも成立する — プール実装の
    #: バグ検出用で、本番は必ず False (プール版)。
    meta_sequential: bool = False

    # --- L3 / S8: アイスバーグ (表示量 + 隠れ量) ---
    #: γ の主要因は分割であり、アイスバーグは補助 (§6.2 — 二重計上禁止)。
    #: on/off アブレーションで γ が ±0.05 以内に収まる規模に留めること。
    enable_iceberg: bool = False  # S8
    #: 表示上限 (ロット) を超える指値がアイスバーグ候補になり、この確率で採用。
    book_iceberg_frac: float = 0.15
    #: 表示量 (ロット)。約定で表示が尽きるたび隠れ量から補充する。
    book_iceberg_display_lots: float = 2.0
    #: 補充時にキュー末尾へ回る (時間優先を失う — 標準的な取引所ルール)。
    #: False で時間優先を保持する取引所の変種 (指示書 §6.1)。
    book_iceberg_refill_tail: bool = True

    # --- L3 / S9: queue-reactive (状態依存の意思決定層) ---
    # ★強度は状態依存にしない (指示書 §3.2 の採用案)。Hawkes が「いつ・種別・
    # サイド」を決め、S9 は種別を所与として「どこに置くか・どれを取り消すか」
    # だけを板の状態から決める — 分岐比 n̂ は構造的に S7 のまま。
    # §3.3 の平均 1 乗法変調 g_m は**使っていない** (意思決定層で足りることを
    # 実測で確認してから、必要になったときに限り導入する)。
    enable_queue_reactive: bool = False  # S9
    #: スプレッド依存の板内配置 (§5 — small tick レジームの主役):
    #: in-spread 側の総重みを m(s) = min(1 + slope·max(0, s − ref), cap) 倍する。
    #: スプレッドが広がるほど内側への指値が増える = 平均回帰の主要な源。
    #: これが S8 のインパクト赤字を縮める本体 (縮め切らないこと — §9.2)。
    #: ★ref はスプレッド中央値 (2) では**なく 1** に置く: 中央値を基準にすると
    #: 変調が半分の時間しか効かず、η・VR が動かない (較正グリッドで実測)。
    #: ref=1 なら s=2 (中央値) でも m=1+slope が掛かり、常時の復元力になる。
    qr_inspread_slope: float = 3.0
    qr_spread_ref: float = 1.0
    qr_inspread_cap: float = 8.0
    #: h(Δ, s) の形状 (§5): 板内配置の距離分布を「べき則 (0)」から「d=1 の重みで
    #: 平坦 (1)」へ線形ブレンドする。平坦側ほど改善が深く入り (反対 best 近くまで)、
    #: 広がったスプレッドを 1 イベントで大きく閉じる = 平均回帰が強くなる。
    qr_inspread_flat: float = 0.0
    #: キュー依存の取消選択 (§6): 一様選択を重み付きに置き換える。
    #: w = [floor + (1−floor)·exp(−dist·Δ)] · L^len_pow · (1 + back·b)
    #: (Δ = own best からの距離 [tick]、L = そのレベルのキュー長、
    #:  b = キュー内の相対後方度 ∈ [0,1] — seq の先頭/末尾比から O(1) で近似)。
    #: ★既定は**中立** (decay=0, floor=1, len=0, back=0 — 一様選択と選択・消費
    #: ともに厳密同値)。理由 (250 日 × 6 シード実測): 前方傾斜の取消は small tick
    #: レジームでは板の前面を薄くして**赤字指標を全て悪化**させる (ノイズ約定の
    #: 戻り率 0.29→0.12、β −0.25→−0.27)。ハンプ (⑳) は前方消耗 + 板内配置だけで
    #: tick 距離 4 に立つ。傾斜は §4 の表どおり large tick レジームの主役 —
    #: 機構は実装済みで、そのレジームの config が有効化する。
    #: ★傾斜を使う場合 floor > 0 は安定性の要 (実測事故): 遠方重みを 0 に潰すと
    #: 遠方注文が永久滞留して N が増え、取消レート (δ0·N) の増加分が全て前方に
    #: 落ちて前面が殲滅される正帰還 (スプレッド中央値 3,718 tick を実測)。
    qr_cx_dist_decay: float = 0.0
    qr_cx_w_floor: float = 1.0
    qr_cx_len_pow: float = 0.0
    qr_cx_back: float = 0.0
    #: 成行サイズのデプス依存 (§3.2 表の 3 行目)。0 = 無効。
    #: ★既定で無効: サイズ分布ゲート (仕様適合) と衝突するため、まず配置と
    #: 取消だけで η・OBI・赤字方向が届くかを測る (届いた — README)。
    qr_mo_depth_frac: float = 0.0
    #: OBI による成行符号バイアス (§7.2)。0 = 無効。
    #: ★まず機構的創発 (§7.1) で corr(I, Δm) を測る。使う場合は on/off で
    #: γ ±0.05 のアブレーションが必須 (⑪ の二重計上リスク)。
    qr_obi_bias: float = 0.0

    #: S9 fallback (§8.2): 明示的な uncertainty zones 層。板は既にティック
    #: 格子上で動くので**常用しない** — queue-reactive の較正で η が範囲に
    #: 入らない場合の非常口。有効化したら「板の動学が実証と合っていない」
    #: 警告として README に記録すること。
    enable_uncertainty_zones: bool = False  # S9 (fallback)
    uz_eta: float = 0.15

    # --- L3 / S10: p* との結合 (工程最大の山場) ---
    #: メタオーダー**生成時**の符号バイアス強度 (指示書 §2.1):
    #:   P(sign=+1) = 0.5 + 0.5·tanh(κ·d/s),  d = log p* − log mid,
    #:   s = σ_t·√τ_meta (σ_t は L2 の現在ボラ — 高ボラ期に反応が鈍らないため)。
    #: ★子注文は親の符号を継承 (§2.2 — 子レベルで掛けると run length が壊れ
    #: γ = α−1 が崩れる)。0 = 切断 (S6〜S9 の状態、経路はビット単位不変)。
    kappa: float = 0.0
    #: s の正規化スケール τ_meta [秒]。S9 本番 (120 日) の完走メタオーダー
    #: (N≥2) の平均実行スパン実測 430 秒。κ と積で効くので規約として固定し、
    #: 強度の探索は κ 側で行う。
    kappa_tau_meta_sec: float = 430.0
    #: S10c: L2 の緩慢ボラ → L1 活動度のリンク (指示書 §7):
    #:   Z_t = exp(c_vol·V_t − c_vol²/2),  V_t = (MA_w(log σ − log φ_σ) − m_V)/s_V
    #: Z は Hawkes の**ベースラインのみ**に乗る (カーネル不変 → n 保存、φ と同じ
    #: 扱い)。取消ベースライン δ0·N にも乗せ、フローの釣り合いを保つ。
    #: ★V にラフ成分を含めない (§7.1) — 移動平均 (既定 3 日) がラフ (半減期
    #: 0.75 日) を落とし、MSM 遅い成分・緩慢 OU・χ₂ を通す。
    #: ★正規化は理論定数 (Jensen 補正 exp(−c²/2)) — 実行全体の標本平均で割ると
    #: 早い時刻の Z が遅い時刻の情報を知る (因果性)。E[Z]=1 は近似で、実現
    #: レートのずれは hawkes_realized_rates ゲート (±5%) が監視する。
    c_vol: float = 0.0
    c_vol_ma_days: float = 3.0
    #: V の標準化スケール s_V。較正実測 (MA_3d 系列の**定常** SD — 価格層のみ
    #: 1000〜5000 日 × 5 シードで中央値 0.41〜0.46、5000 日 0.461) を定数として
    #: 固定 — 実行内標本で標準化すると同じ因果性問題が出る。短ホライズンでは
    #: 実行内 Var(V) < 1 で E[Z] が数 % 下振れし、さらに緩慢成分の実現平均
    #: (エポック効果、250 日でも SD ≈ 0.22) で実行ごとに ±20% 揺れる —
    #: どちらも実測済みの物理で、ゲートは多シード中央値で判定する。
    c_vol_v_scale: float = 0.45

    # --- S11: 実現ボラ → L1/L3 の負のフィードバック (内生的危機) ---
    #: 信号は**水準ではなく驚き** (指示書 §2.1): u_t = log(RV_short/RV_long)。
    #: 両 RV は脱季節化 (φ_σ² で除算) したミッドリターンの EWMA — L2 の σ_t は
    #: 参照しない (§2.2、板が観測不能な情報を使わない)。定常状態で u ≈ 0 になり
    #: フィードバックが自然に無効化される。L2 へは戻さない (§2.3 — 事前生成の維持)。
    #: 経路 (全て tanh(u/u_s) で内側を飽和 §3.2 — 乗数は [e^-b, e^+b] に有界):
    #:   δ_t     = δ0·exp(b_δ·tanh(u/u_s))                 取消強度 (L3)
    #:   Δ_scale =    exp(b_Δ·tanh(u/u_s))                 指値配置距離 (L3)
    #:   n_t     = n_min + (n_max−n_min)·sigmoid(b_n·tanh(u/u_s))  分岐比 (L1)
    enable_feedback: bool = False  # S11
    #: S11d (任意): ジャンプ自己励起。S11a〜c 完了後に要否判断 (§7.2 —
    #: 危機カスケードが既にジャンプ的変動を出すため、必要と実測されるまで未実装)。
    enable_jump_hawkes: bool = False
    fb_b_delta: float = 0.0  # b_δ (S11a で確定)
    fb_b_place: float = 0.0  # b_Δ (S11a で確定)
    fb_b_n: float = 0.0  # b_n (S11b で確定)
    #: tanh の飽和スケール u_s ≈ 1.5×SD(u) (実測 SD 1.25〜1.6)。規約として固定し
    #: 強度の探索は b 側で行う (κ/τ_meta と同じ役割分担)。
    fb_u_scale: float = 2.0
    #: ★u の中心化定数 (指示書 §2.1 との整合措置): クラスタしたフローでは
    #: 大半の短期窓が 2 日平均より静かで、E[log(RV_s/RV_l)] は **−1.08 ± 0.15**
    #: (8 ラン実測、シード/ホライズンに安定) — 生の式は自らの要件「定常で u≈0」を
    #: 満たさない (Jensen: E[log RV_s] < log E[RV_s])。m_V (S10c) と同じく実測
    #: 定数で中心化する。事後平均 ≈ 0 は signal_is_surprise ゲートが確認する。
    fb_u_center: float = -1.05
    #: n_t のレンジ (指示書 §3.3)。ハード上限 0.97 — S12 の χ₃ 変調の余地
    #: 0.07 を残すため n_max は 0.90 を推奨値とする (0.97 に張り付けない)。
    fb_n_min: float = 0.75
    fb_n_max: float = 0.90
    #: RV EWMA の半減期 (指示書 §2.1 の目安: 短 5〜30 分、長 1〜5 日)。
    fb_rv_short_halflife_min: float = 15.0
    fb_rv_long_halflife_days: float = 2.0
    #: 危機検出 (§6.1): |5分リターン| > k·σ_t、スプレッド > m×通常、デプス < 通常/m。
    crisis_k_sigma: float = 5.0
    crisis_spread_mult: float = 3.0
    #: 危機頻度の目標帯 [件/年] (ゲート crisis_frequency — S11c の実測で確定)。
    crisis_freq_per_year_lo: float = 0.5
    crisis_freq_per_year_hi: float = 50.0

    # --- 多資産 ---
    n_assets: int = 1  # S13

    # --- 検証スイートの測定器設定 (モデル設定ではない) ---
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # ------------------------------------------------------------------
    # 検証
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._check_basic()
        self._check_stage()
        self._check_unimplemented()
        self._check_s1_params()

    def _check_basic(self) -> None:
        if self.n_days <= 0:
            raise ValueError("n_days は正整数である必要があります")
        if self.steps_per_day <= 0:
            raise ValueError("steps_per_day は正整数である必要があります")
        if self.sigma_bar <= 0:
            raise ValueError("sigma_bar は正である必要があります (年率ボラ)")
        if self.n_assets < 1:
            raise ValueError("n_assets は 1 以上である必要があります")
        if not isinstance(self.validation, ValidationConfig):
            raise TypeError("validation は ValidationConfig である必要があります")
        if self.validation.primary_bar_sec > SESSION_SECONDS:
            raise ValueError(
                f"primary_bar_sec ({self.validation.primary_bar_sec}s) が"
                f" セッション長 ({SESSION_SECONDS:.0f}s) を超えています"
            )

    def _check_stage(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"未知の段階名 {self.stage!r} です。有効な値: {', '.join(STAGES)}")
        if self.stage not in IMPLEMENTED_STAGES:
            raise NotImplementedError(
                f"段階 {self.stage} はまだ実装されていません。"
                f" 実装済みの段階: {', '.join(IMPLEMENTED_STAGES)}。"
                f" 段階を進めるときは simchart/config.py の IMPLEMENTED_STAGES に追加すること。"
            )

    def _check_unimplemented(self) -> None:
        for name, (stage, what, where) in UNIMPLEMENTED_FLAGS.items():
            if getattr(self, name):
                raise _not_implemented(name, stage, what, where, True)
        for name, (stage, what, where, neutral) in _UNIMPLEMENTED_NUMERIC.items():
            value = getattr(self, name)
            if value != neutral:
                raise _not_implemented(name, stage, what, where, value)
        if self.n_assets > 1:
            raise _not_implemented(
                "n_assets", "S13", "多資産 (共通因子と Hayashi-Yoshida 共分散)",
                "simchart/pipeline.py", self.n_assets,
            )

    #: フラグごとの従属パラメータ。フラグが False のままこれらを既定値から動かしても
    #: 何も起きない (暗黙 no-op) ため、その組み合わせを構成エラーとして弾く。
    _S1_MSM_PARAMS = ("msm_k", "msm_b", "msm_gamma1_per_day", "vol_var_target_msm")
    _S1_SLOW_PARAMS = ("ou_half_life_days", "vol_var_target_slow")
    _S2_ROUGH_PARAMS = (
        "rough_hurst", "rough_half_life_days", "vol_var_target_rough", "rough_grid_seconds",
    )
    _S3_JUMP_PARAMS = (
        "jump_p_up", "jump_eta_up", "jump_eta_down", "jump_lambda_per_year",
        "jump_vol_exponent", "jump_intensity_cap", "jump_qv_share_target",
    )
    _S3_LEVERAGE_PARAMS = (
        "leverage_rho_rough", "leverage_rho_slow",
        "leverage_mid_half_life_days", "leverage_mid_var", "leverage_rho_mid",
    )
    _S4_SEASONALITY_PARAMS = (
        "session_type", "phi_sigma_cos", "phi_sigma_sin", "phi_sigma_slope",
        "phi_lambda_cos", "phi_lambda_sin", "phi_lambda_slope",
    )
    _S4_OVERNIGHT_PARAMS = (
        "overnight_variance_share", "overnight_jump_prob",
        "overnight_jump_variance_share",
    )
    _S5_CHAOS_PARAMS = (
        "chaos_system", "chaos_tau_delay", "chaos_beta", "chaos_gamma",
        "chaos_n_exponent", "chaos_dt", "chaos_ic", "chaos_burn_in_units",
        "chaos_days_per_unit", "vol_var_target_chaos", "chaos_normalization",
    )
    _S7_HAWKES_PARAMS = (
        "hawkes_a", "hawkes_tau_seconds", "hawkes_weights",
        "hawkes_mu_mo", "hawkes_mu_lo", "hawkes_delta0", "hawkes_nbar_ref",
        "hawkes_intensity_cap_mult", "hawkes_daily_event_cap",
    )
    _S8_META_PARAMS = (
        "meta_alpha", "meta_n_min", "meta_psi", "meta_supply_ratio",
        "meta_pool_cap", "meta_sequential",
    )
    _S8_ICEBERG_PARAMS = (
        "book_iceberg_frac", "book_iceberg_display_lots", "book_iceberg_refill_tail",
    )
    _S9_QR_PARAMS = (
        "qr_inspread_slope", "qr_spread_ref", "qr_inspread_cap",
        "qr_inspread_flat",
        "qr_cx_dist_decay", "qr_cx_w_floor", "qr_cx_len_pow", "qr_cx_back",
        "qr_mo_depth_frac", "qr_obi_bias",
    )
    _S9_UZ_PARAMS = ("uz_eta",)
    _S11_FB_PARAMS = (
        "fb_b_delta", "fb_b_place", "fb_b_n", "fb_u_scale", "fb_u_center",
        "fb_n_min", "fb_n_max",
        "fb_rv_short_halflife_min", "fb_rv_long_halflife_days",
        "crisis_k_sigma", "crisis_spread_mult",
        "crisis_freq_per_year_lo", "crisis_freq_per_year_hi",
    )
    _S6_BOOK_PARAMS = (
        "tick_size", "book_mu_mo", "book_alpha_lo", "book_delta_cancel",
        "book_mu_place", "book_place_offset", "book_max_place_ticks",
        "book_allow_inspread", "book_inspread_cap", "book_size_round_weight",
        "book_size_pareto_alpha", "book_lot_values", "book_lot_probs",
        "book_snapshot_interval_sec", "book_snapshot_levels", "book_depth_ticks",
        "book_burn_in_days", "book_init_levels", "book_init_size",
        "book_window_half_ticks", "book_debug_invariants",
    )

    def _check_s1_params(self) -> None:
        defaults = {f.name: f.default for f in dataclasses.fields(type(self))}

        for flag, params in (
            ("enable_msm", self._S1_MSM_PARAMS),
            ("enable_slow_ou", self._S1_SLOW_PARAMS),
            ("enable_rough", self._S2_ROUGH_PARAMS),
            ("enable_jump", self._S3_JUMP_PARAMS),
            ("enable_leverage", self._S3_LEVERAGE_PARAMS),
            ("enable_seasonality", self._S4_SEASONALITY_PARAMS),
            ("enable_overnight", self._S4_OVERNIGHT_PARAMS),
            ("enable_chaos_vol", self._S5_CHAOS_PARAMS),
            ("enable_book", self._S6_BOOK_PARAMS),
            ("enable_hawkes", self._S7_HAWKES_PARAMS),
            ("enable_metaorder", self._S8_META_PARAMS),
            ("enable_iceberg", self._S8_ICEBERG_PARAMS),
            ("enable_queue_reactive", self._S9_QR_PARAMS),
            ("enable_uncertainty_zones", self._S9_UZ_PARAMS),
            ("enable_feedback", self._S11_FB_PARAMS),
        ):
            if not getattr(self, flag):
                changed = [n for n in params if getattr(self, n) != defaults[n]]
                if changed:
                    raise ValueError(
                        f"{flag}=False のまま {', '.join(changed)} が既定値から変更されています。"
                        f" フラグが無効な成分のパラメータは無視される (暗黙 no-op) ため、"
                        f" 意図があるならフラグを有効にし、無いなら既定値に戻してください。"
                    )

        if self.enable_msm:
            if self.msm_k < 1:
                raise ValueError("msm_k は 1 以上である必要があります")
            if self.msm_b <= 1.0:
                raise ValueError("msm_b は 1 より大きい必要があります (成分の時定数を分離するため)")
            if self.msm_gamma1_per_day <= 0:
                raise ValueError("msm_gamma1_per_day は正である必要があります")
            if self.vol_var_target_msm <= 0:
                raise ValueError(
                    "enable_msm=True なのに vol_var_target_msm が 0 以下です。"
                    " 分散配分 0 の MSM は暗黙 no-op になるため許可しません。"
                )
        if self.enable_slow_ou:
            if self.ou_half_life_days <= 0:
                raise ValueError("ou_half_life_days は正である必要があります")
            if self.vol_var_target_slow <= 0:
                raise ValueError(
                    "enable_slow_ou=True なのに vol_var_target_slow が 0 以下です。"
                    " 分散配分 0 の OU は暗黙 no-op になるため許可しません。"
                )
        if self.enable_rough:
            if not (0.0 < self.rough_hurst < 0.5):
                raise ValueError(
                    "rough_hurst は (0, 0.5) の範囲である必要があります"
                    " (H >= 0.5 は「ラフ」ではなく持続的で、長スケールの記憶を汚染する)"
                )
            if self.rough_half_life_days <= 0:
                raise ValueError("rough_half_life_days は正である必要があります")
            if self.rough_half_life_days > 1.0:
                raise ValueError(
                    "rough_half_life_days は 1 日以下である必要があります。"
                    " MSM 最速成分 (時定数 ~1 日) の帯域に食い込むと gph_d が動き、"
                    " スケール分離が壊れます (S2 指示書 §4)。"
                )
            if self.vol_var_target_rough <= 0:
                raise ValueError(
                    "enable_rough=True なのに vol_var_target_rough が 0 以下です。"
                    " 分散配分 0 のラフ成分は暗黙 no-op になるため許可しません。"
                )
            if self.rough_grid_seconds <= 0:
                raise ValueError("rough_grid_seconds は正である必要があります")
            if SESSION_SECONDS % self.rough_grid_seconds != 0:
                raise ValueError(
                    "rough_grid_seconds はセッション長を割り切る必要があります"
                    " (日境界でグリッドが揃わないと物理時間定義が壊れる)"
                )
        if self.enable_jump:
            if self.jump_eta_up <= 1.0:
                raise ValueError(
                    f"jump_eta_up ({self.jump_eta_up}) は 1 より大きい必要があります。"
                    f" eta_u <= 1 では E[e^J] = p eta_u/(eta_u-1) + ... が発散し、"
                    f" マルチンゲール補償項 k が定義できません (S3 指示書 §4.3)。"
                )
            if self.jump_eta_down <= 0:
                raise ValueError("jump_eta_down は正である必要があります")
            if not (0.0 < self.jump_p_up < 1.0):
                raise ValueError("jump_p_up は (0, 1) の範囲である必要があります")
            if self.jump_lambda_per_year <= 0:
                raise ValueError(
                    "enable_jump=True なのに jump_lambda_per_year が 0 以下です。"
                    " 強度 0 のジャンプは暗黙 no-op になるため許可しません。"
                )
            if self.jump_intensity_cap < 1.0:
                raise ValueError("jump_intensity_cap は 1 以上である必要があります")
            if not (0.0 <= self.jump_qv_share_target < 1.0):
                raise ValueError("jump_qv_share_target は [0, 1) の範囲である必要があります")
        if self.enable_leverage:
            if not self.enable_rough:
                raise ValueError(
                    "enable_leverage=True には enable_rough=True が必要です"
                    " (短期チャンネルはラフ成分の駆動 fGn と相関させる)"
                )
            if not self.enable_slow_ou:
                raise ValueError(
                    "enable_leverage=True には enable_slow_ou=True が必要です"
                    " (長期チャンネルは緩慢 OU の駆動と相関させる)"
                )
            rho_sq = self.leverage_rho_rough**2 + self.leverage_rho_slow**2
            if rho_sq >= 1.0:
                raise ValueError(
                    f"rho_rough^2 + rho_slow^2 = {rho_sq:.3f} >= 1。"
                    f" 相関の合成が正定値でなくなります (S3 指示書 §5.1)。"
                )
            if not (-1.0 < self.leverage_rho_mid < 1.0):
                raise ValueError("leverage_rho_mid は (-1, 1) の範囲である必要があります")
            if self.leverage_mid_half_life_days <= 0:
                raise ValueError("leverage_mid_half_life_days は正である必要があります")
            if not (0.0 <= self.leverage_mid_var < self.vol_var_target_slow):
                raise ValueError(
                    f"leverage_mid_var ({self.leverage_mid_var}) は"
                    f" vol_var_target_slow ({self.vol_var_target_slow}) の内数"
                    f" (0 <= mid < slow、0 は無効) である必要があります。中速成分は"
                    f" 緩慢 OU の予算からの再配分であり、総予算を増やしてはならない。"
                )
        if self.enable_seasonality:
            if self.session_type not in ("continuous", "split", "24h"):
                raise ValueError(
                    f"session_type は continuous / split / 24h のいずれかです: {self.session_type!r}"
                )
            if self.session_type != "continuous":
                raise NotImplementedError(
                    f"session_type={self.session_type!r} は未実装です。S4 の対象市場は"
                    f" 連続単一セッション (米国株相当) と決定されています。分割セッション"
                    f" (W 字) を扱うには SESSION_SECONDS の変更が必要で、S0〜S3 の基準値を"
                    f" すべて再生成することになります。"
                )
            if not any(abs(v) > 0 for v in self.phi_sigma_cos + self.phi_sigma_sin):
                raise ValueError(
                    "enable_seasonality=True なのに phi_sigma の係数が全て 0 です。"
                    " 平坦な phi は暗黙 no-op になるため許可しません。"
                )
        if self.enable_overnight:
            if not (0.0 < self.overnight_variance_share < 1.0):
                raise ValueError("overnight_variance_share は (0, 1) の範囲である必要があります")
            if not (0.0 <= self.overnight_jump_prob <= 1.0):
                raise ValueError("overnight_jump_prob は [0, 1] の範囲である必要があります")
            if not (0.0 <= self.overnight_jump_variance_share < 1.0):
                raise ValueError(
                    "overnight_jump_variance_share は [0, 1) の範囲である必要があります"
                )
        if self.enable_chaos_vol:
            if self.chaos_system != "mackey_glass":
                raise NotImplementedError(
                    f"chaos_system={self.chaos_system!r} は未実装です (S5 は Mackey-Glass のみ)"
                )
            if self.vol_var_target_chaos <= 0:
                raise ValueError(
                    "enable_chaos_vol=True なのに vol_var_target_chaos が 0 以下です。"
                    " 分散配分 0 の chi_2 は暗黙 no-op になるため許可しません。"
                )
            if self.chaos_days_per_unit <= 0 or self.chaos_dt <= 0:
                raise ValueError("chaos_days_per_unit と chaos_dt は正である必要があります")
            ratio = self.chaos_tau_delay / self.chaos_dt
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"chaos_tau_delay/chaos_dt = {ratio} が整数ではありません。"
                    f" 遅延値が履歴グリッドに載らず、補間の曖昧さが再現性を壊します。"
                )
            grid_days = self.chaos_dt * self.chaos_days_per_unit
            if grid_days > 0.5:
                raise ValueError(
                    f"カオス格子の間隔 ({grid_days:.3f} 日) が粗すぎます。"
                    f" 20〜40 日の振動を線形補間で運ぶには 0.5 日以下が必要です。"
                )
            if self.chaos_normalization not in ("standardize", "ecdf_normal"):
                raise ValueError(
                    "chaos_normalization は standardize / ecdf_normal のいずれかです"
                )
            if self.chaos_burn_in_units < 10 * self.chaos_tau_delay:
                raise ValueError(
                    f"chaos_burn_in_units ({self.chaos_burn_in_units}) が短すぎます。"
                    f" 過渡が残ると初期の数百日に非定常な水準トレンドが乗ります"
                    f" (tau の 10 倍以上を要求)。"
                )
        if self.enable_book:
            if self.tick_size <= 0:
                raise ValueError("tick_size は正である必要があります")
            if self.p0 / self.tick_size < 10 * self.book_max_place_ticks:
                raise ValueError(
                    "p0/tick_size が小さすぎます (配置範囲に対して価格が 0 に近すぎる)"
                )
            for name in ("book_mu_mo", "book_alpha_lo", "book_delta_cancel"):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} は正である必要があります (0 は暗黙 no-op)")
            if not (0.3 <= self.book_mu_place <= 3.0):
                raise ValueError("book_mu_place は [0.3, 3.0] の範囲を想定しています")
            if self.book_place_offset <= 0 or self.book_max_place_ticks < 10:
                raise ValueError("配置分布のパラメータが不正です")
            if len(self.book_lot_values) != len(self.book_lot_probs):
                raise ValueError("book_lot_values と book_lot_probs の長さが一致しません")
            if abs(sum(self.book_lot_probs) - 1.0) > 1e-9:
                raise ValueError("book_lot_probs の合計が 1 ではありません")
            if self.book_size_pareto_alpha <= 1.0:
                raise ValueError(
                    "book_size_pareto_alpha は 1 より大きい必要があります (平均の存在)"
                )
            if not (0.0 <= self.book_size_round_weight <= 1.0):
                raise ValueError("book_size_round_weight は [0,1] の範囲です")
            if self.book_snapshot_interval_sec <= 0:
                raise ValueError("book_snapshot_interval_sec は正である必要があります")
            if self.book_window_half_ticks < 4 * self.book_max_place_ticks:
                raise ValueError("book_window_half_ticks が配置範囲に対して小さすぎます")
            if self.book_window_half_ticks * self.tick_size >= self.p0:
                raise ValueError(
                    "book_window_half_ticks * tick_size が p0 以上です。"
                    " 窓の下端が非正の価格になり、対数価格が定義できません。"
                )
        if self.enable_hawkes:
            if not self.enable_book:
                raise ValueError(
                    "enable_hawkes=True には enable_book=True が必要です"
                    " (Hawkes は板の注文流の強度 — 板が無いと消費先が無い)"
                )
            import numpy as _np

            a_mat = _np.asarray(self.hawkes_a, dtype=float)
            if a_mat.shape != (3, 3) or (a_mat < 0).any():
                raise ValueError("hawkes_a は非負の 3x3 行列である必要があります")
            rho = float(max(abs(_np.linalg.eigvals(a_mat))))
            if rho >= 1.0:
                raise ValueError(
                    f"分岐比 rho(a) = {rho:.4f} >= 1 (爆発条件)。n < 1 を厳守すること"
                    f" (指示書 §4.2)。"
                )
            if len(self.hawkes_tau_seconds) != len(self.hawkes_weights):
                raise ValueError("hawkes_tau_seconds と hawkes_weights の長さが一致しません")
            if abs(sum(self.hawkes_weights) - 1.0) > 1e-9:
                raise ValueError("hawkes_weights の合計が 1 ではありません (∫Φ = a の規約)")
            if any(t <= 0 for t in self.hawkes_tau_seconds):
                raise ValueError("hawkes_tau_seconds は正である必要があります")
            if max(self.hawkes_tau_seconds) > 3600.0:
                raise ValueError(
                    f"カーネル最遅時定数 {max(self.hawkes_tau_seconds)}s > 1 時間。"
                    f" 日次帯域に食い込むと S10 で MSM と競合して ③ が壊れる (指示書 §3.2)。"
                )
            if self.hawkes_mu_mo <= 0 or self.hawkes_mu_lo <= 0 or self.hawkes_delta0 <= 0:
                raise ValueError("Hawkes のベースラインは正である必要があります")
            if self.hawkes_nbar_ref <= 0:
                raise ValueError("hawkes_nbar_ref (参照定常注文数) は正である必要があります")
            # ★フロー実行可能性: 定常の取消レートは指値流入を超えられない
            # (取消は板に載った注文しか消せず、約定退出のぶん厳密に少ない)。
            # これを破った初版較正 (r_CX=4500 > r_LO=3000) は励起駆動の取消が
            # 板を食い尽くし、42% の時間で片側が空になった (実測)。
            # なお r > 0 自体は mu > 0 と rho < 1 から自動で従うので検査しない。
            mu_vec = _np.array(
                [
                    2.0 * self.hawkes_mu_mo,
                    2.0 * self.hawkes_mu_lo,
                    self.hawkes_delta0 * self.hawkes_nbar_ref,
                ]
            )
            r_vec = _np.linalg.solve(_np.eye(3) - a_mat.T, mu_vec)
            if r_vec[2] >= r_vec[1]:
                raise ValueError(
                    f"定常取消レート r_CX = {r_vec[2]:.0f}/日 >= 指値流入 r_LO ="
                    f" {r_vec[1]:.0f}/日。板に載らない注文は取り消せないので、この"
                    f" 較正は板を枯渇させる。scripts/calibrate_s7_hawkes.py で"
                    f" S6 の実測レートから較正し直すこと。"
                )
        if self.enable_metaorder:
            if not self.enable_book:
                raise ValueError(
                    "enable_metaorder=True には enable_book=True が必要です"
                    " (メタオーダーの符号は板の成行イベントに乗る)"
                )
            if not (1.0 < self.meta_alpha < 2.0):
                raise ValueError(
                    f"meta_alpha = {self.meta_alpha} は開区間 (1, 2) の外です。"
                    f" α ≤ 1 は E[N] が発散してプールが定常にならず、α ≥ 2 は"
                    f" Var(N) 有限で長期記憶が消える (γ = α−1 ≥ 1 で ACF 可和 —"
                    f" 指示書 §4.3)。"
                )
            if self.meta_n_min < 1:
                raise ValueError("meta_n_min は 1 以上である必要があります")
            if not (0.0 < self.meta_psi <= 1.0):
                raise ValueError(f"meta_psi = {self.meta_psi} は (0, 1] の外です")
            if not (0.0 < self.meta_supply_ratio <= 1.0):
                raise ValueError(
                    f"meta_supply_ratio = {self.meta_supply_ratio} は (0, 1] の外です。"
                    f" 1 超は供給過剰でプールが線形発散する (指示書 §3.2)。"
                )
            if self.meta_pool_cap < 1000:
                raise ValueError("meta_pool_cap が小さすぎます (>= 1000)")
        if self.enable_iceberg:
            if not self.enable_book:
                raise ValueError("enable_iceberg=True には enable_book=True が必要です")
            if not (0.0 <= self.book_iceberg_frac < 1.0):
                raise ValueError(f"book_iceberg_frac = {self.book_iceberg_frac} は [0, 1) の外です")
            if self.book_iceberg_display_lots < 1.0:
                raise ValueError("book_iceberg_display_lots は 1 ロット以上である必要があります")
        if self.enable_queue_reactive:
            if not self.enable_book:
                raise ValueError(
                    "enable_queue_reactive=True には enable_book=True が必要です"
                )
            if self.qr_inspread_slope < 0 or self.qr_cx_dist_decay < 0:
                raise ValueError("qr の傾き・減衰係数は非負である必要があります")
            if self.qr_inspread_cap < 1.0:
                raise ValueError("qr_inspread_cap は 1 以上 (1 = 変調なし)")
            if not (0.0 <= self.qr_inspread_flat <= 1.0):
                raise ValueError("qr_inspread_flat は [0, 1]")
            if self.qr_spread_ref < 1.0:
                raise ValueError("qr_spread_ref は 1 tick 以上")
            if self.qr_cx_len_pow < 0 or self.qr_cx_back < 0:
                raise ValueError("qr_cx_len_pow / qr_cx_back は非負である必要があります")
            if not (0.0 < self.qr_cx_w_floor <= 1.0):
                raise ValueError(
                    f"qr_cx_w_floor = {self.qr_cx_w_floor} は (0, 1] の外です。"
                    f" 0 は遠方注文の永久滞留 → 前面殲滅の正帰還を起こす (実測)。"
                )
            if not (0.0 <= self.qr_mo_depth_frac <= 10.0):
                raise ValueError("qr_mo_depth_frac は [0, 10] (0 = 無効)")
            if not (0.0 <= self.qr_obi_bias < 1.0):
                raise ValueError("qr_obi_bias は [0, 1) (0 = 無効。使用時は §7.2 の"
                                 " アブレーションが必須)")
        if self.enable_uncertainty_zones:
            if not self.enable_book:
                raise ValueError(
                    "enable_uncertainty_zones=True には enable_book=True が必要です"
                )
            if not (0.0 < self.uz_eta < 0.5):
                raise ValueError(f"uz_eta = {self.uz_eta} は (0, 0.5) の外です")
        if self.kappa != 0.0:
            if self.kappa < 0:
                raise ValueError(
                    f"kappa = {self.kappa} は負です。負の結合は d を発散させる"
                    f" (指示書 §11「符号の向きを確認」)。"
                )
            if not self.enable_metaorder:
                raise ValueError(
                    "kappa > 0 には enable_metaorder=True が必要です"
                    " (バイアスはメタオーダー生成時の符号に乗る — §2.2)"
                )
            if self.kappa_tau_meta_sec <= 0:
                raise ValueError("kappa_tau_meta_sec は正である必要があります")
        elif self.kappa_tau_meta_sec != 430.0:
            raise ValueError(
                "kappa=0 のまま kappa_tau_meta_sec が既定値から変更されています"
                " (暗黙 no-op — 意図があるなら kappa を設定すること)"
            )
        if self.c_vol != 0.0:
            if self.c_vol < 0:
                raise ValueError(f"c_vol = {self.c_vol} は負です")
            if not self.enable_hawkes:
                raise ValueError(
                    "c_vol > 0 には enable_hawkes=True が必要です"
                    " (Z は Hawkes のベースラインに乗る)"
                )
            if self.c_vol_ma_days <= 0 or self.c_vol_v_scale <= 0:
                raise ValueError("c_vol_ma_days / c_vol_v_scale は正である必要があります")
        elif self.c_vol_ma_days != 3.0 or self.c_vol_v_scale != 0.45:
            raise ValueError(
                "c_vol=0 のまま c_vol_ma_days / c_vol_v_scale が既定値から変更"
                "されています (暗黙 no-op)"
            )
        if self.enable_feedback:
            if not (self.enable_book and self.enable_hawkes):
                raise ValueError(
                    "enable_feedback には enable_book と enable_hawkes が必要です"
                    " (フィードバックは板の RV から L1/L3 へ)"
                )
            if self.fb_b_delta == 0.0 and self.fb_b_place == 0.0 and self.fb_b_n == 0.0:
                raise ValueError(
                    "enable_feedback=True なのに全チャネル (fb_b_*) が 0 です"
                    " (暗黙 no-op — 少なくとも 1 つを正にすること)"
                )
            if min(self.fb_b_delta, self.fb_b_place, self.fb_b_n) < 0:
                raise ValueError("fb_b_* は非負である必要があります")
            if not (0.0 < self.fb_n_min < self.fb_n_max):
                raise ValueError("0 < fb_n_min < fb_n_max が必要です")
            if self.fb_n_max >= 0.97:
                raise ValueError(
                    f"fb_n_max = {self.fb_n_max} ≥ 0.97 (ハード上限、指示書 §3.3)。"
                    " S12 の χ₃ 変調の余地を残すため 0.90 以下を推奨"
                )
            if self.fb_u_scale <= 0:
                raise ValueError("fb_u_scale は正である必要があります")
            if self.fb_rv_short_halflife_min <= 0 or self.fb_rv_long_halflife_days <= 0:
                raise ValueError("fb_rv_*_halflife は正である必要があります")
            if (self.fb_rv_short_halflife_min * 60.0
                    >= self.fb_rv_long_halflife_days * SESSION_SECONDS):
                raise ValueError(
                    "RV_short の半減期が RV_long 以上です (驚き信号 u が定義できない)"
                )
            if self.crisis_k_sigma <= 0 or self.crisis_spread_mult <= 1.0:
                raise ValueError("crisis_k_sigma > 0, crisis_spread_mult > 1 が必要です")
            if not (0 < self.crisis_freq_per_year_lo < self.crisis_freq_per_year_hi):
                raise ValueError("危機頻度帯は 0 < lo < hi が必要です")
        if self.vol_var_budget_total <= 0:
            raise ValueError("vol_var_budget_total は正である必要があります")
        allocated = (
            (self.vol_var_target_msm if self.enable_msm else 0.0)
            + (self.vol_var_target_slow if self.enable_slow_ou else 0.0)
            + (self.vol_var_target_rough if self.enable_rough else 0.0)
            + (self.vol_var_target_chaos if self.enable_chaos_vol else 0.0)
        )
        if allocated > self.vol_var_budget_total + 1e-12:
            raise ValueError(
                f"分散配分の合計 ({allocated:.4f}) が最終予算"
                f" ({self.vol_var_budget_total}) を超えています (指示書 §5.2:"
                f" chi_2 の 25% 超は ③⑱ を薄めるため禁止)。"
            )

    # ------------------------------------------------------------------
    # 導出量
    # ------------------------------------------------------------------
    def without_book(self) -> "Config":
        """板を外した同一設定 (L2 凍結検証の基準ラン用)。

        ★``replace(enable_book=False)`` だけでは足りない: 板パラメータが既定値から
        動いていると「フラグ off + 非既定パラメータ」の暗黙 no-op ガードに当たる。
        板パラメータも既定値へ戻す (L2 には一切影響しない値なので比較は成立する)。
        S7 以降は Hawkes も同時に外す (enable_hawkes は enable_book を要求するため、
        板を外した基準ランでは必ず両方落ちる。L2 は Hawkes を一切読まない)。
        """
        defaults = {f.name: f.default for f in dataclasses.fields(type(self))}
        resets = {name: defaults[name] for name in self._S6_BOOK_PARAMS}
        resets.update({name: defaults[name] for name in self._S7_HAWKES_PARAMS})
        resets.update({name: defaults[name] for name in self._S8_META_PARAMS})
        resets.update({name: defaults[name] for name in self._S8_ICEBERG_PARAMS})
        resets.update({name: defaults[name] for name in self._S9_QR_PARAMS})
        resets.update({name: defaults[name] for name in self._S9_UZ_PARAMS})
        resets["kappa"] = defaults["kappa"]
        resets["kappa_tau_meta_sec"] = defaults["kappa_tau_meta_sec"]
        resets["c_vol"] = defaults["c_vol"]
        resets["c_vol_ma_days"] = defaults["c_vol_ma_days"]
        resets["c_vol_v_scale"] = defaults["c_vol_v_scale"]
        resets.update({name: defaults[name] for name in self._S11_FB_PARAMS})
        return self.replace(
            enable_book=False, enable_hawkes=False, enable_feedback=False,
            enable_metaorder=False, enable_iceberg=False,
            enable_queue_reactive=False, enable_uncertainty_zones=False,
            **resets,
        )

    @property
    def total_steps(self) -> int:
        """全期間のステップ数 (リターン本数)。時点数はこれに 1 を足したもの。"""
        return self.n_days * self.steps_per_day

    @property
    def dt_years(self) -> float:
        """1 ステップの長さ (年)。年率ボラを 1 ステップに落とすときに使う。"""
        return 1.0 / (TRADING_DAYS_PER_YEAR * self.steps_per_day)

    @property
    def sigma_step(self) -> float:
        """1 ステップあたりの対数リターン標準偏差。"""
        return float(self.sigma_bar * self.dt_years**0.5)

    # ------------------------------------------------------------------
    # 直列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON 化できる素の辞書に変換する (tuple は list になる)。"""
        return json.loads(json.dumps(dataclasses.asdict(self), default=list))

    def config_hash(self) -> str:
        """設定の正規化 JSON の SHA-256。manifest / 回帰比較で使う。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def replace(self, **changes: Any) -> "Config":
        return dataclasses.replace(self, **changes)

    # ------------------------------------------------------------------
    # ロード
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        data = dict(data)
        validation_data = data.pop("validation", None)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"未知の設定項目です: {sorted(unknown)}。"
                f" 誤記か、あるいは新しい段階のフラグを config.py に宣言し忘れています。"
            )
        kwargs: dict[str, Any] = _coerce(cls, data)
        if validation_data is not None:
            if not isinstance(validation_data, Mapping):
                raise TypeError("validation セクションは辞書である必要があります")
            vknown = {f.name for f in dataclasses.fields(ValidationConfig)}
            vunknown = set(validation_data) - vknown
            if vunknown:
                raise ValueError(f"未知の validation 設定項目です: {sorted(vunknown)}")
            kwargs["validation"] = ValidationConfig(**_coerce(ValidationConfig, validation_data))
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        # ★重複キーを明示エラーにする。yaml.safe_load は重複キーを黙って後勝ちに
        # するため、コピー元の古い行が残っていると設定が静かに上書きされる
        # (S6 で enable_book: true が 140 行下の false に食われる事故が実際に起きた)。
        class _StrictLoader(yaml.SafeLoader):
            pass

        def _no_dup(loader, node, deep=False):
            seen = set()
            for key_node, _ in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in seen:
                    raise ValueError(
                        f"{path}: キー {key!r} が重複しています (行 "
                        f"{key_node.start_mark.line + 1})。YAML は黙って後勝ちに"
                        f"するため、重複は設定の静かな上書きになる — 削除すること。"
                    )
                seen.add(key)
            return yaml.SafeLoader.construct_mapping(loader, node, deep)

        _StrictLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup
        )
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=_StrictLoader) or {}
        if not isinstance(data, Mapping):
            raise TypeError(f"{path} の内容が辞書ではありません")
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """拡張子から YAML / JSON を判別して読み込む。"""
        suffix = Path(path).suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        if suffix == ".json":
            return cls.from_json(path)
        raise ValueError(f"対応していない設定ファイル形式です: {suffix!r} ({path})")


def _coerce(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """YAML の list を dataclass 側の tuple 既定値に合わせて変換する。

    ★入れ子の list (hawkes_a のような tuple[tuple, ...]) は**再帰的に** tuple 化
    する。外側だけ変換すると (list, list, ...) の tuple になり、既定値との比較や
    「フラグ off + 非既定パラメータ」ガードが誤発火する (to_dict 往復で実際に発火)。
    """
    defaults = {f.name: f for f in dataclasses.fields(cls)}

    def deep(v: Any) -> Any:
        return tuple(deep(x) for x in v) if isinstance(v, list) else v

    out: dict[str, Any] = {}
    for key, value in data.items():
        fld = defaults[key]
        default_value = fld.default if fld.default is not dataclasses.MISSING else None
        if isinstance(default_value, tuple) and isinstance(value, list):
            value = deep(value)
        out[key] = value
    return out
