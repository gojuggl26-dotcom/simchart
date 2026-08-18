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

__all__ = ["Gate", "GateResult", "S0_GATES", "STAGE_GATES", "evaluate", "summarize"]

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

#: 段階ごとのゲート。S1 以降を実装するときはここに追加する。
STAGE_GATES: dict[str, tuple[Gate, ...]] = {"S0": S0_GATES}


def gates_for(stage: str) -> tuple[Gate, ...]:
    if stage not in STAGE_GATES:
        raise NotImplementedError(
            f"段階 {stage} のゲートは未定義です。"
            f" 定義済み: {', '.join(sorted(STAGE_GATES))}。"
            f" simchart/validation/gates.py の STAGE_GATES に追加してください。"
        )
    return STAGE_GATES[stage]
