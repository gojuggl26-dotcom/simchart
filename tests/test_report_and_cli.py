"""成果物の書き出し・読み直し・段階間比較と CLI の往復。

``artifacts_written`` ゲートは「書いたつもり」で終わらせないためのものなので、
書いた後に読み直す経路そのものを試す。``compare`` は S2 / S4 / S10 のゲート判定で
必要になるため、S0 の時点で動くことを固定しておく。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from simchart.cli import main
from simchart.config import Config
from simchart.report import (
    REQUIRED_METRIC_GROUPS,
    REQUIRED_TOP_LEVEL_KEYS,
    compare_stages,
    load_metrics,
    verify_metrics_file,
)
from simchart.validation import evaluate
from simchart.validation.gates import S0_GATES

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "s0.yaml"

# ゲート判定に足る標本数を保ちつつテストとして現実的な大きさ。
SMALL_RUN = ["--n-days", "400", "--steps-per-day", "390", "--no-plots"]


# ---------------------------------------------------------------------------
# 設定ファイル
# ---------------------------------------------------------------------------
def test_shipped_config_loads_and_matches_defaults() -> None:
    config = Config.from_yaml(CONFIG_PATH)
    assert config.stage == "S0"
    assert config == Config()  # configs/s0.yaml は既定値と一致していること


def test_config_round_trips_through_dict() -> None:
    config = Config.from_yaml(CONFIG_PATH)
    assert Config.from_dict(config.to_dict()) == config
    assert Config.from_dict(config.to_dict()).config_hash() == config.config_hash()


def test_config_hash_changes_with_the_seed() -> None:
    assert Config().config_hash() != Config(seed=43).config_hash()


# ---------------------------------------------------------------------------
# 成果物
# ---------------------------------------------------------------------------
#: 統計量に依存せず、常に合格していなければならないゲート。
#: 統計ゲート (acf_r_lag1 など) は帰無仮説下でも一定の確率で落ちるので、
#: 配管のテストをそれに依存させない (テストが偶然で赤くなるのを避ける)。
INFRASTRUCTURE_GATES = (
    "pipeline_runs",
    "determinism",
    "rng_stability",
    "rng_streams_distinct",
    "validation_callable",
    "artifacts_written",
)


@pytest.fixture(scope="module")
def run_result(tmp_path_factory) -> tuple[Path, int]:
    out = tmp_path_factory.mktemp("results")
    code = main(["run", "--stage", "S0", "--results-dir", str(out), *SMALL_RUN])
    assert code in (0, 1), f"実行そのものが失敗しました (終了コード {code})"
    return out, code


@pytest.fixture(scope="module")
def run_dir(run_result: tuple[Path, int]) -> Path:
    return run_result[0]


def test_infrastructure_gates_always_pass(run_dir: Path) -> None:
    data = load_metrics("S0", root=run_dir)
    verdicts = {g["name"]: g["passed"] for g in data["gates"]}
    for name in INFRASTRUCTURE_GATES:
        assert verdicts[name] is True, f"{name} が不合格です"


def test_metrics_file_has_every_required_key(run_result: tuple[Path, int]) -> None:
    run_dir, code = run_result
    data = load_metrics("S0", root=run_dir)
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in data, key
    for group in REQUIRED_METRIC_GROUPS:
        assert group in data["metrics"], group
    assert data["all_critical_passed"] is (code == 0)
    assert len(data["gates"]) == len(S0_GATES)


def test_metrics_file_is_strict_json(run_dir: Path) -> None:
    """NaN / Infinity が混入していないこと。

    ``json`` は既定でこれらを非標準の literal として書いてしまい、他の言語や
    ツールから読めないファイルになる。
    """
    raw = (run_dir / "S0" / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw, parse_constant=lambda _: pytest.fail("非標準の定数が含まれています"))


def test_saved_metrics_reproduce_the_gate_verdict(run_dir: Path) -> None:
    """保存した指標だけからゲートを再判定しても同じ結論になること。"""
    data = load_metrics("S0", root=run_dir)
    recomputed = {g.name: g.passed for g in evaluate(S0_GATES, data["metrics"])}
    stored = {g["name"]: g["passed"] for g in data["gates"]}
    assert recomputed == stored


def test_verify_detects_a_missing_file(tmp_path: Path) -> None:
    report = verify_metrics_file(tmp_path / "nope" / "metrics.json")
    assert report["metrics_json_ok"] is False
    results = evaluate(S0_GATES, {"runtime": {"artifacts": report}})
    artifacts = [g for g in results if g.name == "artifacts_written"][0]
    assert artifacts.passed is False


def test_verify_detects_a_truncated_file(tmp_path: Path, run_dir: Path) -> None:
    broken = tmp_path / "S0"
    broken.mkdir()
    data = load_metrics("S0", root=run_dir)
    del data["metrics"]["scaling"]
    (broken / "metrics.json").write_text(json.dumps(data), encoding="utf-8")
    report = verify_metrics_file(broken / "metrics.json")
    assert report["metrics_json_ok"] is False
    assert report["missing_metric_groups"] == ["scaling"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_validate_subcommand(run_result: tuple[Path, int]) -> None:
    """保存済み結果の再判定が、実行時と同じ結論を返すこと。"""
    run_dir, code = run_result
    assert main(["validate", "--stage", "S0", "--results-dir", str(run_dir)]) == code


def test_validate_reports_a_missing_stage(run_dir: Path) -> None:
    assert main(["validate", "--stage", "S3", "--results-dir", str(run_dir)]) == 4


def test_compare_subcommand(run_dir: Path, tmp_path: Path) -> None:
    """後段の段階が出来たときと同じ形で比較できること。"""
    shutil.copytree(run_dir / "S0", run_dir / "S1", dirs_exist_ok=True)
    out = tmp_path / "diff.json"
    assert main(
        ["compare", "--stages", "S0", "S1", "--results-dir", str(run_dir), "--json", str(out)]
    ) == 0

    diff = json.loads(out.read_text(encoding="utf-8"))
    assert diff["stages"] == ["S0", "S1"]
    assert diff["config_hashes"]["S0"] == diff["config_hashes"]["S1"]
    # 同じ結果を複製したので、数値指標の差分は全てゼロ。
    numeric = [row for row in diff["metrics"] if row["delta"] is not None]
    assert numeric, "数値の比較対象が 1 つもありません"
    assert all(row["delta"] == 0 for row in numeric)
    assert all(row["S0"] == row["S1"] for row in diff["gates"])


def test_compare_needs_two_stages(run_dir: Path) -> None:
    assert main(["compare", "--stages", "S0", "--results-dir", str(run_dir)]) == 2


def test_cli_refuses_an_unimplemented_stage(run_dir: Path) -> None:
    assert main(["run", "--stage", "S3", "--results-dir", str(run_dir), *SMALL_RUN]) == 3
