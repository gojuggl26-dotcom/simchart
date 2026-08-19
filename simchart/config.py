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
IMPLEMENTED_STAGES: tuple[str, ...] = ("S0", "S1", "S2")

#: 年率ボラを 1 ステップ分に落とすときの営業日数。
TRADING_DAYS_PER_YEAR: int = 252

#: 1 立会日の長さ (秒)。6.5 時間。steps_per_day はこの長さを何分割するかを表す。
SESSION_SECONDS: float = 6.5 * 3600.0


# ---------------------------------------------------------------------------
# 未実装フラグの表: フラグ名 -> (実装段階, 内容, 追加先)
# ---------------------------------------------------------------------------
UNIMPLEMENTED_FLAGS: dict[str, tuple[str, str, str]] = {
    "enable_seasonality": ("S4", "日内 U 字の活動度季節性 phi(t)", "simchart/layers/l0_calendar.py"),
    "enable_overnight": ("S4", "オーバーナイト・ギャップと寄引", "simchart/layers/l0_calendar.py"),
    "enable_jump": ("S3", "Hawkes ジャンプ成分", "simchart/layers/l2_price.py"),
    "enable_leverage": ("S3", "レバレッジ効果 (リターンとボラの負相関)", "simchart/layers/l2_price.py"),
    "enable_chaos_vol": ("S5", "カオス的ボラ成分 chi_2", "simchart/layers/l2_price.py"),
    "enable_hawkes": ("S7", "多変量 Hawkes 注文流", "simchart/layers/l1_activity.py"),
    "enable_chaos_lambda": ("S12", "カオス的強度変調 chi_1", "simchart/layers/l1_activity.py"),
    "enable_chaos_branching": ("S12", "カオス的分岐比変調 chi_3", "simchart/layers/l1_activity.py"),
    "enable_book": ("S6", "板 (リミットオーダーブック) 層", "simchart/layers/l3_book.py"),
    "enable_metaorder": ("S8", "メタオーダー分割と符号自己相関", "simchart/layers/l3_book.py"),
    "enable_queue_reactive": ("S9", "queue-reactive な板ダイナミクス", "simchart/layers/l3_book.py"),
    "enable_uncertainty_zones": ("S9", "uncertainty zones による価格離散化", "simchart/layers/l3_book.py"),
    "enable_feedback": ("S11", "RV から L1/L3 へのフィードバック", "simchart/pipeline.py"),
}

#: 実装済みの機能フラグ。UNIMPLEMENTED_FLAGS から行を移すときはこちらに追記する。
#: (テストが「全 bool フラグはどちらかの台帳に載っている」ことを強制し、
#:  新フラグの登録漏れ = 暗黙 no-op を構造的に防ぐ)
IMPLEMENTED_FLAGS: tuple[str, ...] = (
    "enable_msm",  # S1
    "enable_slow_ou",  # S1
    "enable_rough",  # S2
)

#: フラグ以外 (数値パラメータ) の未実装条件。
_UNIMPLEMENTED_NUMERIC = {
    "kappa": ("S10", "潜在情報価格 p* と注文流の結合強度", "simchart/layers/l3_book.py", 0.0),
    "feedback_gain": ("S11", "フィードバック利得", "simchart/pipeline.py", 0.0),
}


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
    enable_jump: bool = False  # S3
    enable_leverage: bool = False  # S3
    enable_chaos_vol: bool = False  # S5  (chi_2)

    # --- L1 ---
    enable_hawkes: bool = False  # S7
    enable_chaos_lambda: bool = False  # S12 (chi_1)
    enable_chaos_branching: bool = False  # S12 (chi_3)

    # --- L3 ---
    enable_book: bool = False  # S6
    enable_metaorder: bool = False  # S8
    enable_queue_reactive: bool = False  # S9
    enable_uncertainty_zones: bool = False  # S9
    kappa: float = 0.0  # S10 p* 結合強度

    # --- フィードバック ---
    enable_feedback: bool = False  # S11
    feedback_gain: float = 0.0

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

    def _check_s1_params(self) -> None:
        defaults = {f.name: f.default for f in dataclasses.fields(type(self))}

        for flag, params in (
            ("enable_msm", self._S1_MSM_PARAMS),
            ("enable_slow_ou", self._S1_SLOW_PARAMS),
            ("enable_rough", self._S2_ROUGH_PARAMS),
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
        if self.vol_var_budget_total <= 0:
            raise ValueError("vol_var_budget_total は正である必要があります")
        allocated = (
            (self.vol_var_target_msm if self.enable_msm else 0.0)
            + (self.vol_var_target_slow if self.enable_slow_ou else 0.0)
            + (self.vol_var_target_rough if self.enable_rough else 0.0)
        )
        if allocated > self.vol_var_budget_total + 1e-12:
            raise ValueError(
                f"分散配分の合計 ({allocated:.4f}) が最終予算"
                f" ({self.vol_var_budget_total}) を超えています。予算を使い切ると"
                f" 後段 (S5 chi_2 など) が入らなくなります (指示書 §6)。"
            )

    # ------------------------------------------------------------------
    # 導出量
    # ------------------------------------------------------------------
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
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
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
    """YAML の list を dataclass 側の tuple 既定値に合わせて変換する。"""
    defaults = {f.name: f for f in dataclasses.fields(cls)}
    out: dict[str, Any] = {}
    for key, value in data.items():
        fld = defaults[key]
        default_value = fld.default if fld.default is not dataclasses.MISSING else None
        if isinstance(default_value, tuple) and isinstance(value, list):
            value = tuple(value)
        out[key] = value
    return out
