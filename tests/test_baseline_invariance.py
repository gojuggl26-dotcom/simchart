"""前段階との不変性照合 (baseline_invariance_check) のロジック検証。

S2 の合否機構そのものなので、「一致すれば通る」「壊せば落ちる」の両方向を固定する。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from simchart import Config, run
from simchart.pipeline import baseline_invariance_check
from simchart.validation.suite import run_all

S1 = dict(stage="S1", enable_msm=True, enable_slow_ou=True)
S2 = dict(stage="S2", enable_msm=True, enable_slow_ou=True, enable_rough=True)
BASE = dict(seed=21, n_days=1200, steps_per_day=390)


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> tuple[Path, Config, dict]:
    """小規模な S1 の metrics.json を基準として書き、S2 の metrics を作る。"""
    root = tmp_path_factory.mktemp("results")
    c1 = Config(**S1, **BASE)
    m1 = run_all(run(c1), c1)
    (root / "S1").mkdir()
    with open(root / "S1" / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"stage": "S1", "git_commit": "test", "config": c1.to_dict(),
             "metrics": _jsonable(m1)},
            fh, ensure_ascii=False,
        )
    c2 = Config(**S2, **BASE)
    m2 = run_all(run(c2), c2)
    return root, c2, m2


def _jsonable(obj):
    from simchart.validation.base import jsonable

    return jsonable(obj)


def test_invariance_passes_for_true_s2(setup) -> None:
    root, c2, m2 = setup
    report = baseline_invariance_check(c2, _jsonable(m2), "S1", results_root=root)
    # 小標本 (1200 日) では gph のペア差が本番よりゆらぐので、個別に確認する。
    assert report["checks"]["rng_s1_streams"]["passed"], report["checks"]["rng_s1_streams"]
    assert report["checks"]["absr_acf_profile"]["passed"]
    assert report["checks"]["gph_d"]["diff"] is not None


def test_invariance_detects_touched_s1_streams(setup) -> None:
    """帰無対照: MSM の切替回数が 1 つでも違えば rng_s1_streams が落ちる。"""
    root, c2, m2 = setup
    broken = copy.deepcopy(_jsonable(m2))
    broken["vol"]["msm"]["table"][0]["n_switches"] += 1
    report = baseline_invariance_check(c2, broken, "S1", results_root=root)
    assert report["checks"]["rng_s1_streams"]["passed"] is False


def test_invariance_detects_moved_gph_d(setup) -> None:
    """帰無対照: d が 0.05 動いたら inv_gph_d が落ちる (スケール分離失敗の想定)。"""
    root, c2, m2 = setup
    broken = copy.deepcopy(_jsonable(m2))
    broken["daily"]["gph_abs_r"]["d"] = broken["daily"]["gph_abs_r"]["d"] + 0.05
    report = baseline_invariance_check(c2, broken, "S1", results_root=root)
    assert report["checks"]["gph_d"]["passed"] is False


def test_invariance_reports_missing_baseline(setup) -> None:
    root, c2, m2 = setup
    report = baseline_invariance_check(c2, _jsonable(m2), "S0", results_root=root)
    assert report["passed"] is False
    assert "error" in report
