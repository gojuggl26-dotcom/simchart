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

__all__ = ["Gate", "GateResult", "S0_GATES", "S1_GATES", "STAGE_GATES", "evaluate", "summarize"]

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

#: 段階ごとのゲート。S2 以降を実装するときはここに追加する。
STAGE_GATES: dict[str, tuple[Gate, ...]] = {"S0": S0_GATES, "S1": S1_GATES}


def gates_for(stage: str) -> tuple[Gate, ...]:
    if stage not in STAGE_GATES:
        raise NotImplementedError(
            f"段階 {stage} のゲートは未定義です。"
            f" 定義済み: {', '.join(sorted(STAGE_GATES))}。"
            f" simchart/validation/gates.py の STAGE_GATES に追加してください。"
        )
    return STAGE_GATES[stage]
