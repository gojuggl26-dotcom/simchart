"""ゲート定義と判定。

ゲートは「その段階として正しい状態」を機械的に検査するもので、**通ることを目的に
閾値をいじってはならない。** 落ちたら原因を調べる。閾値が偶然でも落ちるほど厳しい
場合に正しい対処は「閾値を緩める」ではなく「推定量の精度を上げる (標本数の多い
スケールで測る、バンド幅を広げる)」であり、この方針は
:mod:`simchart.validation.scaling` と :mod:`simchart.validation.memory` の
docstring に書いてある。

``critical=False`` のゲートは警告のみで、``all_critical_passed`` には影響しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .base import get_path, jsonable

__all__ = [
    "Gate", "GateResult", "S0_GATES", "S1_GATES", "S2_GATES", "STAGE_GATES",
    "evaluate", "summarize",
]

#: 大きすぎて metrics.json のゲート欄に載せたくないキー。
_BULKY_KEYS = frozenset(
    {
        "values", "lags", "table", "profile", "per_q", "scales", "propagator",
        "probs", "theoretical_quantiles", "empirical_quantiles", "per_stream",
        "per_array", "traceback",
        # S4: 配列を返す枝 (φ の形、スペクトル比、ラグ別 ACF)。
        "ratios", "peak_acf", "excess_by_multiple", "u", "coefficients",
        "raw_profile", "fits",
    }
)


@dataclass(frozen=True)
class Gate:
    """1 つの合否条件。

    Attributes
    ----------
    name:
        ゲート名 (metrics.json に出る識別子)。
    metric_path:
        ``"memory.acf_r.lag1_z"`` のようなドット記法での指標の位置。
    check:
        取り出した値を受け取って合否を返す関数。
    critical:
        ``False`` なら警告のみで全体の合否に影響しない。
    threshold:
        人間が読むための閾値の記述。
    description:
        何を見ているかの説明。
    """

    name: str
    metric_path: str
    check: Callable[[Any], bool]
    critical: bool = True
    threshold: str = ""
    description: str = ""


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: Any
    threshold: str
    critical: bool
    metric_path: str
    error: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "value": jsonable(self.value),
            "threshold": self.threshold,
            "critical": self.critical,
            "metric_path": self.metric_path,
        }
        if self.error:
            out["error"] = self.error
        if self.description:
            out["description"] = self.description
        return out


def _compact(value: Any) -> Any:
    """ゲート欄に載せる値を小さくする (長い配列やテーブルを落とす)。"""
    if isinstance(value, Mapping):
        return {k: _compact(v) for k, v in value.items() if k not in _BULKY_KEYS}
    if isinstance(value, (list, tuple)) and len(value) > 12:
        return f"<{len(value)} 要素>"
    return value


def evaluate(gates: Sequence[Gate], metrics: Mapping[str, Any]) -> list[GateResult]:
    """ゲートを順に判定する。指標が無い・判定中に例外が出た場合は不合格。"""
    results: list[GateResult] = []
    for gate in gates:
        try:
            value = get_path(metrics, gate.metric_path)
        except KeyError as exc:
            results.append(
                GateResult(
                    name=gate.name, passed=False, value=None, threshold=gate.threshold,
                    critical=gate.critical, metric_path=gate.metric_path,
                    error=str(exc), description=gate.description,
                )
            )
            continue
        try:
            passed = bool(gate.check(value))
            error = None
        except Exception as exc:  # noqa: BLE001
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            GateResult(
                name=gate.name, passed=passed, value=_compact(value),
                threshold=gate.threshold, critical=gate.critical,
                metric_path=gate.metric_path, error=error, description=gate.description,
            )
        )
    return results


def summarize(results: Sequence[GateResult]) -> dict[str, Any]:
    critical = [r for r in results if r.critical]
    failed_critical = [r.name for r in critical if not r.passed]
    failed_warning = [r.name for r in results if not r.critical and not r.passed]
    return {
        "n_gates": len(results),
        "n_critical": len(critical),
        "n_passed": sum(1 for r in results if r.passed),
        "all_critical_passed": len(failed_critical) == 0,
        "failed_critical": failed_critical,
        "failed_warning": failed_warning,
    }


# ---------------------------------------------------------------------------
# 判定関数 (lambda ではなく名前を付ける。metrics.json の可読性と再利用のため)
# ---------------------------------------------------------------------------
def _is_true(value: Any) -> bool:
    return value is True


def _abs_lt(limit: float) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        return value is not None and abs(float(value)) < limit

    return check


def _lt(limit: float) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        return value is not None and float(value) < limit

    return check


def _gt(limit: float) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        return value is not None and float(value) > limit

    return check


def _between(lo: float, hi: float) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        return value is not None and lo <= float(value) <= hi

    return check


# ---------------------------------------------------------------------------
S0_GATES: tuple[Gate, ...] = (
    Gate(
        name="pipeline_runs",
        metric_path="runtime.pipeline.completed",
        check=_is_true,
        threshold="例外なく完走",
        description="パイプラインが最後まで走ったか。",
    ),
    Gate(
        name="determinism",
        metric_path="runtime.determinism.bitwise_identical",
        check=_is_true,
        threshold="同一シード 2 回実行でビット単位同一",
        description="同じ設定で 2 回走らせ、全配列とダイジェストが完全一致するか。",
    ),
    Gate(
        name="rng_stability",
        metric_path="runtime.rng_stability.unchanged",
        check=_is_true,
        threshold="ダミーストリーム追加後も既存ストリームが不変",
        description=(
            "後段で新しい乱数ストリームを足しても既存の系列が動かないこと。"
            "これが崩れると段階間の差分が新機能の効果か乱数のずれか区別できなくなる。"
        ),
    ),
    Gate(
        name="rng_streams_distinct",
        metric_path="runtime.rng_stability.streams_distinct",
        check=_is_true,
        critical=True,
        threshold="宣言済みストリームどうしが別系列",
        description="名前ハッシュの衝突や別名化で 2 つの層が同じ乱数を共有していないか。",
    ),
    Gate(
        name="acf_r_lag1",
        metric_path="memory.acf_r.lag1_z",
        check=_abs_lt(2.0),
        threshold="|rho(1)| < 2/sqrt(N)  (|z| < 2)",
        description="リターンに線形の自己相関が無いこと。S0 で出たら実装事故。",
    ),
    Gate(
        name="acf_abs_r_lag1",
        metric_path="memory.acf_abs_r.lag1_z",
        check=_abs_lt(2.0),
        threshold="|rho(1)| < 2/sqrt(N)  (|z| < 2)",
        description=(
            "|リターン| にも記憶が無いこと。ボラティリティ・クラスタリングは S1 から。"
        ),
    ),
    Gate(
        name="ljung_box",
        metric_path="memory.ljung_box_r.pvalue_primary",
        check=_gt(0.01),
        threshold="p > 0.01 (ラグ 20)",
        description="リターンの系列相関をまとめて検定。多重比較を避けて単一ラグで判定する。",
    ),
    Gate(
        name="gph_d",
        metric_path="memory.gph_abs_r.d",
        check=_abs_lt(0.05),
        threshold="|d| < 0.05",
        description="|リターン| に長期記憶が無いこと。S2 でラフ成分を入れると d が動く。",
    ),
    Gate(
        name="variance_ratio",
        metric_path="scaling.variance_ratio.max_abs_dev",
        check=_lt(0.10),
        threshold="全 q で 0.90 <= VR <= 1.10 (max|VR-1| < 0.10)",
        description="分散が時間に比例すること = ランダムウォーク。",
    ),
    Gate(
        name="kurtosis",
        metric_path="tails.moments.kurtosis",
        check=_between(2.7, 3.3),
        threshold="2.7 <= 尖度 <= 3.3",
        description="基準粒度のリターンが正規並みの尖度であること。ファットテールは S1 以降。",
    ),
    Gate(
        name="kurtosis_flat",
        metric_path="scaling.kurtosis_by_scale.max_abs_dev_from_3_gated",
        check=_lt(0.4),
        threshold="ゲート対象の全スケールで 2.6 <= 尖度 <= 3.4",
        description=(
            "集計しても尖度が変わらないこと (単一フラクタル)。"
            "標本数が閾値未満のスケールは記録のみで判定に使わない。"
        ),
    ),
    Gate(
        name="zeta_q_linear",
        metric_path="scaling.zeta_q.r2",
        check=_gt(0.99),
        threshold="zeta_q を q に回帰した R^2 > 0.99",
        description="zeta_q = q/2 の直線であること = 多重フラクタルでない。",
    ),
    Gate(
        name="signature_plot_flat",
        metric_path="scaling.signature_plot.max_rel_dev",
        check=_lt(0.10),
        threshold="中央値からの最大相対乖離 < 0.10",
        description="実現分散が集計スケールに依存しないこと。ノイズは S9 から。",
    ),
    Gate(
        name="adf",
        metric_path="scaling.adf.combined_ok",
        check=_is_true,
        threshold="log P で単位根を棄却せず (p > 0.01)、リターンで棄却 (p < 0.01)",
        description="価格が非定常でリターンが定常であること。",
    ),
    Gate(
        name="validation_callable",
        metric_path="runtime.validation.all_callable",
        check=_is_true,
        threshold="全検証関数が例外なく呼べる",
        description=(
            "該当層が無い関数は not_applicable を返すのが正しく、例外は不合格。"
        ),
    ),
    Gate(
        name="artifacts_written",
        metric_path="runtime.artifacts.metrics_json_ok",
        check=_is_true,
        threshold="results/<stage>/metrics.json が存在し必須項目を含む",
        description="成果物が実際に書き出され、読み直せること。",
    ),
)

# ---------------------------------------------------------------------------
# S1 のゲート
# ---------------------------------------------------------------------------
def _budget_check(value: Any) -> bool:
    """分散予算: MSM シェア 45〜55%、緩慢 OU シェア 15〜25%。

    **分母は最終予算 vol_var_budget_total (0.25) であって、現在の Var(log sigma)
    合計 (S1 で 0.175) ではない** (指示書 §6 の配分表の定義)。現在合計を分母に
    すると 0.716 / 0.287 になり 1.43 倍ずれる。
    """
    if not isinstance(value, Mapping):
        return False
    msm = value.get("msm")
    slow = value.get("slow_ou")
    if msm is None or slow is None:
        return False
    return 0.45 <= float(msm) <= 0.55 and 0.15 <= float(slow) <= 0.25


#: S1 で S0 から**不変**であるべきゲート (確率ボラは方向情報を与えない)。
#: acf / ljung_box は真の sigma で標準化した系列で測る。確率ボラの下では
#: 非標準化系列の検定はサイズが歪み (Q(20) の期待値だけで p < 0.01)、
#: 「実装バグでないのに落ちる」ため。S0 では標準化は定数スケールに退化し
#: 非標準化版と z 統計が同値なので、これは S0 の測定の自然な一般化である。
_S1_INVARIANT_GATES: tuple[Gate, ...] = (
    Gate(
        name="acf_r_lag1",
        metric_path="memory.acf_r_std.lag1_z",
        check=_abs_lt(2.0),
        threshold="|rho(1)| < 2/sqrt(N)  (|z| < 2, 標準化リターン)",
        description="ボラ変動がリターンの自己相関を作っていないこと。S0 から不変。",
    ),
    Gate(
        name="ljung_box",
        metric_path="memory.ljung_box_r_std.pvalue_primary",
        check=_gt(0.01),
        threshold="p > 0.01 (ラグ 20, 標準化リターン)",
        description="同上 (まとめて検定)。S0 から不変。",
    ),
    Gate(
        name="variance_ratio",
        metric_path="scaling.variance_ratio.max_abs_dev",
        check=_lt(0.10),
        threshold="全 q で 0.90 <= VR <= 1.10",
        description="マルチンゲール性の保持。S0 から不変。",
    ),
    Gate(
        name="adf",
        metric_path="scaling.adf.combined_ok",
        check=_is_true,
        threshold="log P で単位根を棄却せず、リターンで棄却",
        description="価格の非定常性とリターンの定常性。S0 から不変。",
    ),
    Gate(
        name="rng_diffusion",
        metric_path="runtime.rng_diffusion.match",
        check=_is_true,
        threshold="l2.diffusion の消費列が S0 相当とビット単位一致",
        description=(
            "MSM/OU のストリームを足しても拡散乱数の系列が変わっていないこと。"
            "名前ハッシュ方式 RNG 設計の実地検証。"
        ),
    ),
)

#: S1 で新規に満たすべきゲート。
_S1_NEW_GATES: tuple[Gate, ...] = (
    Gate(
        name="gph_d",
        metric_path="daily.gph_abs_r.d",
        check=_between(0.30, 0.45),
        threshold="d ∈ [0.30, 0.45] (日次 |r|)",
        description="MSM から創発する見かけの長期記憶。d は直接指定できない量。",
    ),
    Gate(
        name="absr_acf_powerlaw",
        metric_path="daily.acf_abs_r_powerlaw.r2",
        check=_gt(0.95),
        critical=False,
        threshold="log-log R^2 > 0.95 (ラグ 1〜100 日) — 記録のみ (2026-08-19 オペレータ承認)",
        description=(
            "MSM の ACF は有限個の指数の重ね合わせで厳密なべき則ではなく、"
            "理論上限が R^2=0.913 (ノイズゼロでも)。指数減衰との識別も 5000 日では"
            "どの統計量でも不能と実測で確認済み。長期記憶の出現判定は gph_d が担い、"
            "決定的なべき則判定は真のべき則が入る S2 (ラフ成分) に持ち越す。"
        ),
    ),
    Gate(
        name="absr_acf_lag1",
        metric_path="daily.acf_abs_r.lag1",
        check=_gt(0.13),
        threshold="rho(1) > 0.13 (日次 |r|)",
        description=(
            "ボラティリティ・クラスタリングの水準。閾値は実測分布 (16 シードで"
            " 0.138〜0.226、母平均 0.18) の外側に設定 (2026-08-19 オペレータ承認。"
            "指示書の 0.15 は偽陽性 19%)。S0 では ~0。"
        ),
    ),
    Gate(
        name="kurtosis_daily",
        metric_path="daily.moments.kurtosis",
        check=_gt(4.4),
        threshold="日次尖度 > 4.4",
        description=(
            "ボラ混合によるファットテール (部分的 — alpha~3 は S3 の仕事)。"
            "閾値は実測分布 (16 シードで 4.54〜7.14、母平均 5.3) の外側に設定"
            " (2026-08-19 オペレータ承認。指示書の 5.0 は偽陽性 50%)。S0 では 3.0。"
        ),
    ),
    Gate(
        name="kurtosis_decreasing",
        metric_path="daily.kurtosis_decay.decreasing",
        check=_is_true,
        threshold="集計スケール増で尖度が減少 (回帰傾き負 かつ 最細 > 最粗)",
        description="集計正規性の萌芽。外生ファットテールではこれが出ない。",
    ),
    Gate(
        name="zeta_q_nonlinear",
        metric_path="daily.zeta_curvature.c2",
        check=_lt(0.0),
        critical=False,
        threshold="zeta_q の 2 次係数 c2 < 0 — 記録のみ (2026-08-19 オペレータ承認)",
        description=(
            "マルチフラクタル性の出現。指示書の R^2 < 0.99 は S0/S1 の実測分布が"
            "重なり分離不能 (分散予算 0.175 では曲率が小さい)。感度の高い曲率 c2"
            " (q<=4, 1..100 日, 重なり窓) でも 16 シード中 6 本が S0 域と重なる"
            "ため判定はできず、記録して S2 (ラフ成分で曲率が増える) で判定する。"
            "S1 の c2 は [-0.025, -0.001]、S0 は [-0.005, +0.004]。"
        ),
    ),
    Gate(
        name="var_budget",
        metric_path="vol.ensemble.shares_of_budget",
        check=_budget_check,
        threshold="MSM 45〜55% / 緩慢OU 15〜25% (分母 = 最終予算 0.25、断面)",
        description="分散予算の配分。S2/S5 の枠を残す。",
    ),
    Gate(
        name="var_total",
        metric_path="vol.ensemble.var_log_sigma",
        check=_between(0.15, 0.20),
        threshold="Var(log sigma) ∈ [0.15, 0.20] (アンサンブル断面)",
        description="S1 の到達点は最終予算 0.25 の 70%。使い切らない。",
    ),
    Gate(
        name="e_sigma2",
        metric_path="vol.ensemble.e_sigma2_ratio",
        check=_between(0.98, 1.02),
        threshold="|E[sigma^2]/sigma_bar^2 - 1| < 0.02 (アンサンブル断面 20 万本)",
        description=(
            "凸性補正 -Var(X) の検証。1 経路の時間平均は ±17% ゆらぐため、"
            "定常断面のアンサンブル平均で判定する (SE ~ 0.22%)。"
        ),
    ),
    Gate(
        name="scale_invariance",
        metric_path="runtime.scale_invariance.passed",
        check=_is_true,
        threshold="2 解像度 (23400 / 390) で日次統計が一致",
        description=(
            "gamma_i と theta が物理時間定義であることの検証。"
            "per-step 切替確率型の実装はここで落ちる。"
        ),
    ),
)

#: S0 のインフラ系ゲート (S1 でもそのまま維持)。
_S1_INFRA_GATES: tuple[Gate, ...] = tuple(
    g for g in S0_GATES
    if g.name in (
        "pipeline_runs", "determinism", "rng_stability", "rng_streams_distinct",
        "validation_callable", "artifacts_written",
    )
)

S1_GATES: tuple[Gate, ...] = _S1_INFRA_GATES + _S1_INVARIANT_GATES + _S1_NEW_GATES


# ---------------------------------------------------------------------------
# S2 のゲート
# ---------------------------------------------------------------------------
#: S1 のゲートのうち S2 でそのまま維持するもの (var_total だけ範囲が変わる)。
_S2_INHERITED_GATES: tuple[Gate, ...] = tuple(
    g for g in S1_GATES if g.name != "var_total"
)

_S2_NEW_GATES: tuple[Gate, ...] = (
    Gate(
        name="var_total",
        metric_path="vol.ensemble.var_log_sigma",
        check=_between(0.185, 0.215),
        threshold="Var(log sigma) ∈ [0.185, 0.215] (アンサンブル断面)",
        description="S2 の到達点は最終予算 0.25 の 80% (S1 の 0.175 + ラフ 0.025)。",
    ),
    Gate(
        name="h_latent",
        metric_path="rough.h_latent.h",
        check=_between(0.08, 0.15),
        threshold="潜在 log sigma の粗さ指数 H ∈ [0.08, 0.15] (5 分〜4 時間)",
        description=(
            "S2 の中心的検証。判定は潜在パス (真値) で行う — RV 側推定は推定誤差で"
            "下方に偏るため記録のみ (rough.h_rv)。"
        ),
    ),
    Gate(
        name="h_linearity",
        metric_path="rough.h_latent.linearity_r2",
        check=_gt(0.98),
        threshold="zeta_q^vol の q 線形回帰 R^2 > 0.98",
        description="単一フラクタルな粗さであること (ラフ成分の帯域では q に線形)。",
    ),
    Gate(
        name="vol_incr_acf_negative",
        metric_path="rough.increment_acf.lag1",
        check=_lt(0.0),
        threshold="log sigma 増分の 1 ラグ ACF < 0 (60 秒グリッド)",
        description=(
            "反持続性の確認。H < 1/2 の fGn 駆動なら ~2^{2H-1}-1 ≈ -0.43。"
            "持続的な過程を誤って入れるとここが正になる。"
        ),
    ),
    Gate(
        name="var_budget_rough",
        metric_path="rough.share_of_budget_path.value",
        check=_between(0.08, 0.12),
        threshold="ラフ成分の実測分散シェア 8〜12% (経路実測 / 分母 0.25)",
        description=(
            "ラフ成分は半減期 <1 日で経路分散が良く推定できる (SD ~2.5%) ため、"
            "MSM/OU と違い経路実測で判定する。DH + フィルタ + eta 逆算の"
            "パイプライン全体を end-to-end に検証する。"
        ),
    ),
    Gate(
        name="stationarity_y",
        metric_path="rough.stationarity_y.stationary",
        check=_is_true,
        threshold="Y が定常 (前半/後半の平均・分散一致、ADF 棄却)",
        description=(
            "非定常な fBm/Volterra の混入検査。分散が t^{2H} で増大していると"
            "後半の分散が大きくなり、低周波ドリフトが GPH を汚染する (§10-1)。"
        ),
    ),
    Gate(
        name="inv_gph_d",
        metric_path="runtime.baseline_invariance.checks.gph_d.passed",
        check=_is_true,
        threshold="|d(S2) - d(S1)| <= 0.03 (日次 |r|)",
        description=(
            "★S2 で最重要。長スケールの記憶が動いたらスケール分離の失敗であり、"
            "ラフ成分が MSM/OU の帯域に漏れている。診断手順は指示書 §10 "
            "(非定常 fBm → HL_r 過大 → シェア過大 → 測定窓重複 → パラメータ改変 → "
            "チャンク接合、の順に確認)。"
        ),
    ),
    Gate(
        name="inv_absr_powerlaw_gamma",
        metric_path="runtime.baseline_invariance.checks.absr_powerlaw_gamma.passed",
        check=_is_true,
        threshold="|r| ACF べき則指数が S1 から ±10% 以内 (binned R^2 も非劣化)",
        description="長スケールのべき則減衰の形状が保たれていること (③)。",
    ),
    Gate(
        name="inv_absr_acf_profile",
        metric_path="runtime.baseline_invariance.checks.absr_acf_profile.passed",
        check=_is_true,
        threshold="日次 |r| ACF (ラグ 10〜100) の平均 |Δrho| <= 0.02",
        description="長スケールの ACF プロファイルが S1 と一致していること (③)。",
    ),
    Gate(
        name="inv_kurtosis_daily",
        metric_path="runtime.baseline_invariance.checks.kurtosis_daily.passed",
        check=_is_true,
        threshold="日次尖度の増加が +0.5 以内",
        description="ラフ成分の分散混合で微増するのは正しいが、過大な増加は配分過大の兆候。",
    ),
    Gate(
        name="inv_zeta_c2",
        metric_path="runtime.baseline_invariance.checks.zeta_c2.passed",
        check=_is_true,
        critical=False,
        threshold="zeta 曲率 c2 が S1 から悪化していない (記録系)",
        description="マルチフラクタル性の萌芽が保たれていること (⑱)。判定は S1 と同じく警告扱い。",
    ),
    Gate(
        name="inv_rng_s1_streams",
        metric_path="runtime.baseline_invariance.checks.rng_s1_streams.passed",
        check=_is_true,
        threshold="MSM の切替回数・占有率と OU の x0・経路統計が S1 と厳密一致",
        description=(
            "l2.vol_rough を追加しても S1 のストリーム (l2.diffusion / l2.vol_msm / "
            "l2.vol_slow) に 1 draw も触れていないことの実測 (名前ハッシュ RNG の検証)。"
        ),
    ),
)

S2_GATES: tuple[Gate, ...] = _S2_INHERITED_GATES + _S2_NEW_GATES


# ---------------------------------------------------------------------------
# S3 のゲート
# ---------------------------------------------------------------------------
def _z_acf_check(value: Any) -> bool:
    """z の全ラグ ACF が 3.7/sqrt(N) 以内 (§6.2 の直接テスト)。

    指示書の閾値 2/sqrt(N) は 60 ラグの最大値に対しては多重比較で iid でも
    E[max] ~ 2.9/sqrt(N) となり純乱数で落ちる (S0 の ±2σ ゲートと同型の問題)。
    Bonferroni 60 本・両側 5% の 3.66 を丸めた 3.7 を使う。実装欠陥 (共通ショック
    の単純加算) は +rho^2/n ~ 8e-3 >> 3.7/sqrt(117M) = 3.4e-4 なので検出力は保たれる。
    """
    if not isinstance(value, Mapping):
        return False
    m = value.get("max_abs_acf")
    n = value.get("n")
    if m is None or not n:
        return False
    return float(m) < 3.7 / float(n) ** 0.5


def _corr_within(target_key: str, tol: float) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        realized = value.get("realized")
        target = value.get(target_key) if target_key in value else value.get("target")
        if realized is None or target is None:
            return False
        return abs(float(realized) - float(target)) <= tol

    return check


def _neg_between(lo: float) -> Callable[[Any], bool]:
    """lo <= value < 0 (負の方向性検定。0 は含まない)。"""

    def check(value: Any) -> bool:
        return value is not None and lo <= float(value) < 0.0

    return check


def _corr_within_or_disabled(target_key: str, tol: float) -> Callable[[Any], bool]:
    """成分が無効 (target 不在) なら自明成立、有効なら ±tol を要求する。"""
    inner = _corr_within(target_key, tol)

    def check(value: Any) -> bool:
        if isinstance(value, Mapping) and value.get("target") is None and value.get(
            "realized"
        ) is None:
            return True
        return inner(value)

    return check


#: S2 の不変チェックのうち S3 では**性質が変わる**もの:
#: - inv_kurtosis_daily (Δk <= +0.5): ジャンプで尖度が +2〜+13 上がるのが S3 の目的
#:   (①)。絶対ゲート kurtosis_daily と multiseed の hill/skew が管理する
#: - inv_absr_powerlaw_gamma (±10%): ジャンプが |r| ACF の推定にノイズを加え
#:   Δγ/γ が実測 0.02〜0.36 動く。S3 指示書 §10 の要求は R^2 非劣化のみ
#: 観測 |r| の gph_d 絶対ゲートも S3 では差し替える: ジャンプ・レバレッジが
#: スペクトルに加える白色成分が d の測定値を系統的に -0.05 下げる (真の記憶構造
#: は不変 — inv_gph_d が潜在 log sigma で検証する)。帯を下方拡張し、多シード
#: 中央値で判定する (単一シードの d は SD ~0.04 でばらつくため)。
_S3_DROPPED_INVARIANCE = {"inv_kurtosis_daily", "inv_absr_powerlaw_gamma", "gph_d"}
_S3_INHERITED_GATES: tuple[Gate, ...] = tuple(
    g for g in S2_GATES if g.name not in _S3_DROPPED_INVARIANCE
)

_S3_NEW_GATES: tuple[Gate, ...] = (
    Gate(
        name="gph_d",
        metric_path="multiseed.gph_d.median",
        check=_between(0.25, 0.45),
        threshold="観測 |r| の GPH d ∈ [0.25, 0.45] (10 シード中央値。S1 の [0.30,0.45]"
        " から下限を拡張 — 白色混入バイアス -0.05 を記録の上で)",
        description=(
            "ジャンプとレバレッジは |r| のスペクトルに白色成分を加え、真の記憶が"
            "不変でも測定 d を平坦化させる (実測: jump -0.03、leverage -0.02〜-0.03)。"
            "③ の構造の不変性は inv_gph_d が潜在 log sigma で検証する。"
        ),
    ),
    Gate(
        name="hill_alpha",
        metric_path="multiseed.hill_alpha.median",
        check=_between(3.0, 5.0),
        threshold="Hill α ∈ [3.0, 5.0] (日次リターン・上位 5%、10 シード中央値)",
        description=(
            "有限標本の見かけのテール指数 (§3.2 — 漸近べき則ではないのが正しい)。"
            "測定条件 (日次・上位 5%) を固定して報告する。"
        ),
    ),
    Gate(
        name="hill_increasing",
        metric_path="multiseed.hill_scale_slope.median",
        check=_gt(0.0),
        threshold="Hill α が集計スケールで上昇 (log スケール回帰傾き > 0、中央値)",
        description="集計正規性 (⑱)。べき則ジャンプだと α 不変になるので識別でもある。",
    ),
    Gate(
        name="skewness_daily",
        metric_path="multiseed.skewness_daily.median",
        check=_between(-1.5, -0.1),
        threshold="日次歪度 ∈ [-1.5, -0.1] (10 シード中央値)",
        description="非対称ジャンプ (p<0.5, η_d<η_u) 由来の負の歪度。",
    ),
    Gate(
        name="jv_share",
        metric_path="multiseed.jv_share.median",
        check=_between(0.05, 0.15),
        threshold="BNS の JV share ∈ [5%, 15%] (10 シード中央値)",
        description="総二次変動に占めるジャンプ寄与 (§7)。σ̄_diff の縮小と対応する。",
    ),
    Gate(
        name="leverage_corr",
        metric_path="multiseed.leverage_corr.median",
        check=_neg_between(-0.10),
        threshold="corr(r_t, RV_{t+1}) ∈ [-0.10, 0) (10 シード中央値。指示書の"
        " [-0.28, -0.16] から変更 — 2026-08-20 オペレータ裁定)",
        description=(
            "指示書の帯は per-step 相関構成の理論上限 (~-0.06) を超えており達成不能。"
            "中速成分 (実測上限 -0.14) は gph_d を最大 -0.15 動かし ③ を壊すため"
            "無効化 (同じ予算の取り合いと実測確定)。真値 ~-0.017 に対しシード"
            "ゆらぎ ±0.02 なので、方向性の検定 (負であること、median の偽陽性 ~2%)"
            " として判定する。水準は S10 の板側チャンネルが担う。"
        ),
    ),
    Gate(
        name="leverage_shape",
        metric_path="leverage.function.shape_ok",
        check=_is_true,
        threshold="L(1) < 0 かつ mean L(1..20) < 0 (指示書の「全て負」から変更)",
        description=(
            "弱いレバレッジ水準 (裁定後) では個々の L(h) が SE ~0.014 のゼロ近傍に"
            "あり「20 本全て負」は点推定ノイズで確率的に落ちる。方向と形状の検定と"
            "して L(1) と平均で判定する (2026-08-20 裁定の帰結)。"
        ),
    ),
    Gate(
        name="eta_u_valid",
        metric_path="jumps.generator.eta_up",
        check=_gt(1.0),
        threshold="η_u > 1 (E[e^J] の存在条件)",
        description="config 検証と二重化。",
    ),
    Gate(
        name="martingale_compensation",
        metric_path="jumps.generator.compensation_applied",
        check=_is_true,
        threshold="補償項 -λ(t) k dt が適用されている",
        description=(
            "忘れると価格に系統ドリフト (§4.3)。適用量の厳密検証はテスト "
            "(補償 on/off の終端差 = Σ λ k dt) が行う。"
        ),
    ),
    Gate(
        name="corr_rough_realized",
        metric_path="leverage.generator.corr_rough_check",
        check=_corr_within("target", 0.02),
        threshold="実測 corr(セル集計 z, ε) = ρ_rough ± 0.02",
        description="§6.4 の実装検証 (短期チャンネル)。",
    ),
    Gate(
        name="corr_slow_realized",
        metric_path="leverage.generator.corr_slow_check",
        check=_corr_within("target", 0.02),
        threshold="実測 corr(z, ξ) = ρ_slow ± 0.02",
        description="§6.4 の実装検証 (長期チャンネル)。",
    ),
    Gate(
        name="corr_mid_realized",
        metric_path="leverage.generator.corr_mid_check",
        check=_corr_within_or_disabled("target", 0.05),
        threshold="実測 corr(u_d, 中速駆動) = ρ_mid ± 0.05 (中速無効時は自明成立)",
        description=(
            "中速チャンネルの実装検証。既定では無効 (leverage_mid_var=0、"
            "2026-08-20 裁定) なので target が無ければ通す。"
        ),
    ),
    Gate(
        name="z_no_autocorr",
        metric_path="leverage.generator.z_acf",
        check=_z_acf_check,
        threshold="z の ACF (ラグ 1..60) が全て 3.7/√N 以内 (Bonferroni 補正)",
        description=(
            "§6.2 の bridge を正しく実装したかの直接テスト。共通ショックの単純"
            "加算 (+ρ²/n ≈ 8e-3) を確実に検出する。"
        ),
    ),
    Gate(
        name="inv_absr_powerlaw_r2",
        metric_path="runtime.baseline_invariance.checks.absr_powerlaw_gamma.r2_not_degraded",
        check=_is_true,
        threshold="|r| ACF べき則の binned R^2 が S2 から悪化していない (-0.05 まで)",
        description=(
            "S3 指示書 §10 の趣旨 (③ の形状維持)。γ の ±10% はジャンプノイズで"
            "測れないため R^2 非劣化に置き換え (経緯は _S3_DROPPED_INVARIANCE)。"
        ),
    ),
)

S3_GATES: tuple[Gate, ...] = _S3_INHERITED_GATES + _S3_NEW_GATES


# ---------------------------------------------------------------------------
# S4 のゲート
# ---------------------------------------------------------------------------
def _phi_norm_check(value: Any) -> bool:
    """φ_σ は二乗平均 1、φ_λ は一乗平均 1 (規約を取り違えていないか)。"""
    if not isinstance(value, Mapping):
        return False
    a = value.get("phi_sigma_sq_mean_error")
    b = value.get("phi_lambda_mean_error")
    if a is None or b is None:
        return False
    return float(a) < 1e-3 and float(b) < 1e-3


def _phi_shape_check(value: Any) -> bool:
    """日内の起伏比と形状 (§4.3-4.4)。

    - φ_σ² の最大/最小 ∈ [3, 6]、φ_λ の最大/最小 ∈ [4, 10]
    - ボラは寄付が最大で引けはそれより低い / 出来高は引けが最大
    - どちらも最小はセッション中盤 (端が最小なら U 字になっていない)
    """
    if not isinstance(value, Mapping):
        return False
    try:
        rs = float(value["phi_sigma_sq_max_min_ratio"])
        rl = float(value["phi_lambda_max_min_ratio"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        3.0 <= rs <= 6.0
        and 4.0 <= rl <= 10.0
        and bool(value.get("phi_sigma_open_gt_close"))
        and bool(value.get("phi_lambda_close_gt_open"))
        and bool(value.get("phi_sigma_min_interior"))
        and bool(value.get("phi_lambda_min_interior"))
        and bool(value.get("phi_sigma_positive"))
        and bool(value.get("phi_lambda_positive"))
    )


def _abs_median_gt(limit: float) -> Callable[[Any], bool]:
    """多シード指標の |中央値| が下限を超えること。

    符号が構成によって反転しうる量 (季節性の GPH 汚染など) は、符号つきで判定すると
    「向きが違うだけで実装は正しい」ケースを落とす。大きさで判定し、符号は
    記録として残す。
    """

    def check(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        m = value.get("median")
        return m is not None and abs(float(m)) > limit

    return check


def _flat_at_noise_floor(value: Any) -> bool:
    """日内プロファイルの起伏が標本誤差の水準まで落ちていること。

    閾値 1.35 の較正 (帰無 8 シード x 400 日 x 390 バーの実測):
    季節性の無い S3 で比は 0.98〜1.12 (平均 1.05、最大 1.12)。季節性のある生の
    系列は 4.36〜4.83 なので、1.35 は帰無に約 20% の余裕を残しつつ対立仮説とは
    3 倍以上離れている。標本誤差は分割標本から推定しており (理論値は参考値)、
    正規性の仮定に依存しない。
    """
    if not isinstance(value, Mapping):
        return False
    x = value.get("excess_over_se")
    return x is not None and float(x) < 1.35


def _spectral_null_level(value: Any) -> bool:
    """スペクトル高調波比が帰無水準にあること (S3 実測 1.1〜2.3)。"""
    if not isinstance(value, Mapping):
        return False
    x = value.get("mean_ratio")
    return x is not None and float(x) < 5.0


def _phi_recovery_check(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    c = value.get("correlation")
    e = value.get("max_abs_rel_error")
    return c is not None and e is not None and float(c) > 0.99 and float(e) < 0.10


def _gap_null_control(value: Any) -> bool:
    """ギャップが翌日の日中リターンの向きを予測しないこと (帰無対照)。

    ★これは「効果があること」ではなく「無いこと」を要求するゲートなので、
    検出力不足で自動的に通ってしまう。標本数の下限を課して、少なくとも
    ``|corr| > 0.15`` 程度の漏れは捕らえられる状態でのみ合格にする。
    """
    if not isinstance(value, Mapping):
        return False
    c = value.get("corr_gap_next_intraday")
    se = value.get("corr_gap_next_intraday_se")
    n = value.get("n_gaps")
    if c is None or se is None or not n:
        return False
    return int(n) >= 100 and abs(float(c)) <= 3.0 * float(se)


def _on_kurtosis_higher(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    a = value.get("kurtosis_gap")
    b = value.get("kurtosis_intraday_daily")
    return a is not None and b is not None and float(a) > float(b)


#: S4 では S3 の絶対ゲートを 1 つも差し替えない。
#:
#: 当初は「観測 |r| の GPH d が季節性で上振れするから脱季節化系列に差し替える」
#: つもりだったが、``multiseed.gph_d`` が測っているのは**日次**リターンの |r| で
#: あり、季節性は日内現象なので原理的に効かない: 日次分散は
#: ``sum_k phi_k^2 sigma^2 dt = sigma^2 (1 日分)`` で、φ_σ の二乗正規化により
#: **厳密に**不変だからである (潜在側の ``daily.latent_gph_d`` も、日内平均
#: log φ が全日で同一定数なので GPH が厳密に不変)。
#: 汚染が出るのは日内バーの |r| のほうで、そちらは S3 でゲートされていない量なので
#: 「差し替え」ではなく新規ゲート (seasonality_bias_gph) として追加する。
#: 実測で日次 GPH が動かないことは inv_gph_d と gph_d が継承のまま通ることで確認する。
_S4_REPLACED: set[str] = set()

_S4_INHERITED_GATES: tuple[Gate, ...] = tuple(
    g for g in S3_GATES if g.name not in _S4_REPLACED
)

_S4_NEW_GATES: tuple[Gate, ...] = (
    # --- φ の定義そのもの ---
    Gate(
        name="phi_normalization",
        metric_path="seasonality.phi_normalization",
        check=_phi_norm_check,
        threshold="(1/T)∫φ_σ²du = 1 かつ (1/T)∫φ_λdu = 1 (どちらも誤差 < 1e-3)",
        description=(
            "★正規化の規約が σ と λ で違う (§4.1)。分散は加算されるので φ_σ は"
            "**二乗**の平均を 1 に、強度は一乗の平均を 1 にする。片方の規約を"
            "もう片方へ流用すると Jensen の不等式の分だけ日次積分分散が目標を外れる。"
        ),
    ),
    Gate(
        name="intraday_shape",
        metric_path="seasonality.phi_normalization",
        check=_phi_shape_check,
        threshold="φ_σ² の起伏比 ∈ [3,6]、φ_λ ∈ [4,10]、寄付>引け(σ)・引け>寄付(λ)、最小が中盤",
        description="§4.3-4.4 の形状要件。係数は起伏比 4.5 / 7.0 を狙って数値的に逆算した。",
    ),
    # --- 季節性が実際に入っているか (フラグの空振り検出) ---
    Gate(
        name="seasonality_present",
        metric_path="seasonality.deseasonalization.raw.spectral.mean_ratio",
        check=_gt(20.0),
        threshold="生の |r| のスペクトル高調波比 > 20 (帰無水準は 1)",
        description=(
            "★enable_seasonality が黙って空振りしていないことの検出。実測 241 "
            "(1 分足) / 46 (30 分足) に対し S3 の帰無は 1.1〜2.3。"
            "除去側のゲートだけだと「最初から季節性が無い」場合に全部通ってしまう。"
        ),
    ),
    # --- 除去の効き目 (真値経路) ---
    Gate(
        name="deseason_flatness_true",
        metric_path="seasonality.deseasonalization.true_phi_removed.flatness",
        check=_flat_at_noise_floor,
        threshold="真値 φ 除去後の日内プロファイルの sd(log) が標本誤差の 1.35 倍未満",
        description=(
            "除去できたかの主判定。バー粒度に依らず検出力があるのでスペクトルより"
            "こちらを主にする。実測: 生 4.12 倍 → 除去後 1.01 倍。"
        ),
    ),
    Gate(
        name="deseason_spectral_true",
        metric_path="seasonality.deseasonalization.true_phi_removed.spectral",
        check=_spectral_null_level,
        threshold="真値 φ 除去後のスペクトル高調波比 < 5 (帰無水準 1〜2.3)",
        description=(
            "周波数領域での確認。周期成分は離散的な高調波にしか力を持たず、"
            "長期記憶は連続スペクトルなので、両者はここで原理的に分離できる。"
        ),
    ),
    # --- 除去の効き目 (推定経路 = 実務で使えるか) ---
    Gate(
        name="phi_estimation_accuracy",
        metric_path="seasonality.deseasonalization.recovery",
        check=_phi_recovery_check,
        threshold="φ̂ と真の φ の相関 > 0.99 かつ 最大相対誤差 < 10%",
        description=(
            "φ を知らない立場からの推定 (バー別 |r| 中央値 → 対数を Fourier 回帰)。"
            "実測 corr 0.9992 / 最大相対誤差 2.0%。実データへ持っていくのはこの経路。"
        ),
    ),
    Gate(
        name="deseason_flatness_est",
        metric_path="seasonality.deseasonalization.est_phi_removed.flatness",
        check=_flat_at_noise_floor,
        threshold="推定 φ̂ 除去後も日内プロファイルが標本誤差の 1.35 倍未満",
        description=(
            "推定誤差を含めても道具として使えるか。実測 1.00 倍。"
            "真値経路 (1.01) をわずかに下回るのは φ̂ が同じ標本に当てはめられていて"
            "標本ノイズを少し吸収するため — 過学習の兆候だが 8 母数 / 390 ビンなので微小。"
        ),
    ),
    # --- 季節性が長期記憶の測定を汚す量 (S7 への布石) ---
    Gate(
        name="seasonality_bias_gph",
        metric_path="multiseed.gph_bias_intraday",
        check=_abs_median_gt(0.005),
        threshold="日内バー |r| の GPH d の汚染量 |生 − 脱季節化後| > 0.005 (10 シード中央値)",
        description=(
            "★S4 を作る動機そのものの定量化。季節性は |r| のスペクトルの日内高調波に"
            "力を足し、GPH の対数ペリオドグラム回帰を歪める。"
            "★符号は固定されない — **高調波が推定バンドのどこに落ちるかで決まる**。"
            "回帰の説明変数 -log(4sin²(λ/2)) は低周波ほど大きいので、高調波が"
            "バンドの低周波側に来れば傾きは上がり、高周波側なら下がる。実測でも"
            "n_days=400 (高調波 5 本、低周波寄り) では +0.017、本番 n_days=5000 "
            "(高調波 2 本、バンドの 41%/82% 位置) では **-0.022** と反転した。"
            "したがって判定は絶対値で行う。本番 10 シードでは -0.0145〜-0.0242 と"
            "符号も大きさも安定 (|中央値| = 0.022 = 3.8 SE)。"
            "S7 の Hawkes 分岐比の過大推定 (Filimonov-Sornette) と同じ機構であり、"
            "そこでは φ_λ の時間変更で対処する (time_change_by_phi_lambda)。"
            "★日次リターンの GPH (gph_d ゲート) はこの汚染を受けない — φ_σ の"
            "二乗正規化により日次分散が厳密に不変だから。汚染は日内でのみ起きる。"
        ),
    ),
    Gate(
        name="gph_d_intraday_deseason",
        metric_path="multiseed.gph_d_intraday_deseason.median",
        check=_between(0.42, 0.62),
        threshold="脱季節化後の日内バー |r| の GPH d ∈ [0.42, 0.62] (10 シード中央値)",
        description=(
            "汚染を取り除いた側が妥当な範囲にあること。日次の gph_d ゲート"
            "([0.25,0.45]) と水準が違うのは、日内バーは標本が 390 倍で GPH の"
            "測定帯域がまるごと別だから (同じ量ではない)。帯は実測から引いた: "
            "本番 10 シードで 0.501〜0.556 (中央値 0.520)、同一設定の S3 が 0.553。"
        ),
    ),
    # --- オーバーナイト ---
    Gate(
        name="overnight_share",
        metric_path="seasonality.overnight.variance_share",
        check=_between(0.15, 0.27),
        threshold="ON の分散シェア ∈ [0.15, 0.27] (目標 0.20)",
        description=(
            "帯が目標の ±35% と広い理由は 2 つ。(1) 尖度 12〜20 のギャップ 5000 本の"
            "標本分散は標準誤差が大きい。(2) ★この比の**分母**である日中日次分散は"
            "右に歪んだ推定量で、中央値が期待値より下に来るため、比の実測は系統的に"
            "目標より上に出る (6 シードで 0.211〜0.236、平均 0.224 — 6/6 が上振れ)。"
            "分子そのものは設計値どおりで、生成側診断の sample_var / var_on_target が"
            "0.93〜1.19 (平均 1.05) と 1 を挟む。分散設計の証人はそちらである。"
        ),
    ),
    Gate(
        name="overnight_kurtosis",
        metric_path="seasonality.overnight",
        check=_on_kurtosis_higher,
        threshold="ギャップの尖度 > 日中日次リターンの尖度",
        description=(
            "ON は情報が溜まって一度に出るのでテールが厚い (実測 13.8 vs 6.0)。"
            "絶対値でなく日中との**大小**で判定するのは、水準がボラ予算に依存する一方"
            "大小関係は ON をジャンプ主体に設計したことの直接の帰結だから。"
        ),
    ),
    Gate(
        name="overnight_vol_link",
        metric_path="seasonality.overnight.corr_abs_gap_sigma_close",
        check=_gt(0.20),
        threshold="corr(|gap|, σ_close) > 0.20 (SE ≈ 0.05、実測 0.38)",
        description=(
            "★『σ_ON = c_ON σ_close だから相関 1』は構成上自明で検定になっていない"
            "(自動成立するゲートは置かない)。観測できるのは |gap| = |σ_ON z + J| で、"
            "|z| の揺らぎと ON ジャンプで必ず希薄化する。判定は 0 との差 (4 SE) で行う。"
        ),
    ),
    Gate(
        name="overnight_no_lookahead",
        metric_path="seasonality.overnight",
        check=_gap_null_control,
        threshold="|corr(gap_d, 日中_{d+1})| <= 3 SE かつ ギャップ数 >= 100",
        description=(
            "帰無対照。設計上ギャップは翌日の日中方向を予測しない。ここが有意なら"
            "実装が未来を漏らしている。検出力不足で自動的に通らないよう標本下限を課す。"
        ),
    ),
    # --- S3 の予算が動いていないこと ---
    Gate(
        name="inv_jv_share",
        metric_path="runtime.baseline_invariance.checks.jv_share_preserved.passed",
        check=_is_true,
        threshold="ジャンプ QV シェアが S3 から ±0.005 以内",
        description=(
            "S4 の強度補正 (ON 取り分 + φ の Jensen 効果) が効いていることの照合。"
            "補正が抜けると 12.7% → 14.9% に跳ねる。実測 差 +0.00001。"
        ),
    ),
)

S4_GATES: tuple[Gate, ...] = _S4_INHERITED_GATES + _S4_NEW_GATES


# ---------------------------------------------------------------------------
# S5 のゲート
# ---------------------------------------------------------------------------
def _marginal_unimodal_check(value: Any) -> bool:
    """合成 log σ の周辺分布: |超過尖度| < 1 かつ単峰 (§3.2)。

    ゲート対象は**合成後** — chi_2 単体 (MG の周辺は 4 峰) ではない。分散比 1:4 の
    ガウス的成分との合成で滑らかになることを要求している。双峰化すると log RV の
    分布が双峰になり、実証 (log RV はおおむね正規) と乖離する。
    """
    if not isinstance(value, Mapping):
        return False
    k = value.get("excess_kurtosis")
    return k is not None and abs(float(k)) < 1.0 and bool(value.get("unimodal"))


def _spectral_peak_check(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    p = value.get("peak_period_days")
    return p is not None and 20.0 <= float(p) <= 40.0


def _no_daily_peak_check(value: Any) -> bool:
    """日周期帯 (0.8〜1.2 日) のパワーシェア < 1%。

    §4.1 の訂正の検証: 特徴時間を日次にすると決定論的な準周期成分が日周期近傍に
    立ち、S4 で除去した季節性と区別がつかなくなる。正しい写像 (30 日) なら
    この帯のパワーは実質ゼロ (実測 ~1e-15)。
    """
    if not isinstance(value, Mapping):
        return False
    share = value.get("daily_band_power_share")
    return share is not None and float(share) < 0.01


def _hash_recorded_check(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    h = value.get("sha256")
    return isinstance(h, str) and len(h) == 64


#: S4 の絶対ゲートのうち S5 で**範囲だけ変わる**もの。
#: - var_total: chi_2 の 0.050 が加わり 0.200 → 0.250 (予約枠の消化 — 指示書 §5.2)
_S5_INHERITED_GATES: tuple[Gate, ...] = tuple(
    g for g in S4_GATES if g.name != "var_total"
)

_S5_NEW_GATES: tuple[Gate, ...] = (
    Gate(
        name="var_total",
        metric_path="vol.ensemble.var_log_sigma",
        check=_between(0.235, 0.265),
        threshold="Var(log sigma) ∈ [0.235, 0.265] (アンサンブル断面 + chi 周辺分布)",
        description="S5 で最終予算 0.25 を使い切る (MSM 0.125 + OU 0.050 + ラフ 0.025 + chi 0.050)。",
    ),
    # --- chi_2 単体のカオス性 ---
    Gate(
        name="chi2_lyapunov",
        metric_path="chaos.chi_tests.lyapunov.lyapunov_per_unit",
        check=_gt(0.0),
        threshold="chi_2 の最大 Lyapunov 指数 > 0 (Rosenstein 法)",
        description=(
            "決定論的カオスであることの検証。MG(17) の実測 +0.0071/単位 "
            "(文献値 ~0.006)。帰無対照: 正弦波は +0.00006。"
        ),
    ),
    Gate(
        name="chi2_dimension",
        metric_path="chaos.chi_tests.correlation_dimension.d2",
        check=_between(1.5, 5.0),
        threshold="Grassberger-Procaccia 相関次元 ∈ [1.5, 5.0]",
        description="低次元アトラクタであること。MG(17) の実測 1.85 (文献 ~2.1)。",
    ),
    # --- 時間写像 ---
    Gate(
        name="chi2_spectral_peak",
        metric_path="chaos.spectral",
        check=_spectral_peak_check,
        threshold="主要スペクトルピークが 20〜40 日 (§4.2)",
        description=(
            "写像係数 s = 30/49.65 でピークを 30 日に置く。MSM 帯域 (1〜500 日) の"
            "内側、日周期から十分遠く、かつ**副次調波 (2 倍周期 = 62 日) も"
            "潜在 GPH の判定帯 (周期 >= 70 日) の外側**に収まる唯一の配置 "
            "(36〜40 日だと副次調波が帯に入り inv_gph_d が -0.03〜-0.05 動く — 実測)。"
        ),
    ),
    Gate(
        name="chi2_no_daily_peak",
        metric_path="chaos.spectral",
        check=_no_daily_peak_check,
        threshold="日周期帯 (0.8〜1.2 日) のパワーシェア < 1%",
        description="季節性 (S4) と区別がつかなくなる配置の検出 (§4.1 の訂正)。実測 ~1e-15。",
    ),
    # --- 分散予算 ---
    Gate(
        name="var_budget_chi2",
        metric_path="vol.ensemble.shares_of_budget.chaos",
        check=_between(0.18, 0.22),
        threshold="chi_2 の分散シェア ∈ [18%, 22%] (分母 = 最終予算 0.25)",
        description=(
            "窓正規化により構成上 0.200 ちょうどになる — この行は予算算術の文書化で、"
            "経路からの実証は cross_seed_corr が担う (両者の一致が §8 の要求)。"
        ),
    ),
    Gate(
        name="cross_seed_corr",
        metric_path="multiseed.cross_seed_corr.mean",
        check=_between(0.17, 0.23),
        threshold="シード間の corr(log σ_i, log σ_j) ∈ [0.17, 0.23] (φ 除去後、45 対の平均)",
        description=(
            "★S5 の中核ゲート (§8)。chi は全シード共通なので、シード横断相関 = "
            "Var(chi)/Var(log σ) の直接推定になる — 内部状態に触れない実証。"
            "対の値は遅い成分の偶然相関で ±0.09 ばらつく (実測 [0.02, 0.33]) が、"
            "45 対の平均は SE ~0.02 (5 シード予備測定で 0.188)。"
        ),
    ),
    # --- 水準の保存 (数値凸性補正の検証) ---
    Gate(
        name="logvol_marginal",
        metric_path="chaos.marginal_log_vol",
        check=_marginal_unimodal_check,
        threshold="合成 log σ の |超過尖度| < 1 かつ単峰 (§3.2)",
        description=(
            "MG の周辺は 4 峰だが、分散比 1:4 の合成で滑らかになる (実測: 尖度 "
            "-0.09、単峰)。双峰化したら chaos_normalization='ecdf_normal' (案 B) へ。"
        ),
    ),
    # --- 再現性 ---
    Gate(
        name="chi2_hash",
        metric_path="chaos.generator",
        check=_hash_recorded_check,
        threshold="chi_2 の SHA256 が記録されている (再実行一致は determinism が担保)",
        description=(
            "環境間の再現性の証拠。determinism ゲートが同一シード 2 回実行の"
            "ビット単位一致を検証し、その経路には chi が含まれる。"
        ),
    ),
    Gate(
        name="determinism_across_seeds",
        metric_path="multiseed.chi_hash_all_equal",
        check=_is_true,
        threshold="chi_2 の SHA256 が全シードで一致 (決定論成分はシードに依存しない)",
        description="chi が乱数を消費していないことの多シード実証。",
    ),
    # --- 予測一致 (§6) ---
    Gate(
        name="leverage_dilution",
        metric_path="multiseed.dilution_sd_ratio.median",
        check=_between(0.85, 0.95),
        threshold="sd(log σ_S4)/sd(log σ_S5) ∈ [0.85, 0.95] (10 シード中央値、理論 0.894)",
        description=(
            "chi は r と無相関なのでレバレッジ相関は sqrt(Var_S4/Var_S5) 倍に薄まる"
            " (§6)。判定は希釈式が**厳密に**成り立つ log σ の経路 SD 比で行う"
            " (2026-08-21 裁定)。指示書の字義 (相関の比) は |L| ~ 0.02 の水準"
            " (S3 裁定) では SE が信号の 30〜40% あり判定不能 — 3 計器の実測は"
            " multiseed.dilution_corr_* に記録される。シェア過大は検出できる"
            " (25% → 0.87、30% → 0.82)。"
        ),
    ),
    Gate(
        name="chi2_no_direction",
        metric_path="chaos.no_direction.abs_z",
        check=_lt(4.0),
        threshold="|corr(r_daily, chi_daily)| < 4 SE (帰無対照)",
        description=(
            "★§15 の第一禁止事項の検証: chi を価格・リターンの**方向**に入れると"
            "予測可能性 = 裁定機会になり ②⑦⑮ が同時に壊れる。chi は σ にのみ"
            "入るので方向とは無相関のはず。有意なら実装が方向へ漏らしている。"
        ),
    ),
)

S5_GATES: tuple[Gate, ...] = _S5_INHERITED_GATES + _S5_NEW_GATES

#: 段階ごとのゲート。S6 以降を実装するときはここに追加する。
STAGE_GATES: dict[str, tuple[Gate, ...]] = {
    "S0": S0_GATES,
    "S1": S1_GATES,
    "S2": S2_GATES,
    "S3": S3_GATES,
    "S4": S4_GATES,
    "S5": S5_GATES,
}


def gates_for(stage: str) -> tuple[Gate, ...]:
    if stage not in STAGE_GATES:
        raise NotImplementedError(
            f"段階 {stage} のゲートは未定義です。"
            f" 定義済み: {', '.join(sorted(STAGE_GATES))}。"
            f" simchart/validation/gates.py の STAGE_GATES に追加してください。"
        )
    return STAGE_GATES[stage]
