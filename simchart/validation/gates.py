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

#: 段階ごとのゲート。S4 以降を実装するときはここに追加する。
STAGE_GATES: dict[str, tuple[Gate, ...]] = {
    "S0": S0_GATES,
    "S1": S1_GATES,
    "S2": S2_GATES,
    "S3": S3_GATES,
}


def gates_for(stage: str) -> tuple[Gate, ...]:
    if stage not in STAGE_GATES:
        raise NotImplementedError(
            f"段階 {stage} のゲートは未定義です。"
            f" 定義済み: {', '.join(sorted(STAGE_GATES))}。"
            f" simchart/validation/gates.py の STAGE_GATES に追加してください。"
        )
    return STAGE_GATES[stage]
