"""時間スケール不変性 (指示書 §7)。

MSM の切替強度と OU の theta は物理時間 (1 日) で定義されており、グリッド刻みを
変えてもモデルは変わらないはず。「1 ステップあたり切替確率」型の実装はここで落ちる。
この性質は S4 (季節性) と S7 (Hawkes) で再発しやすいので、テストとして固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, run
from simchart.pipeline import scale_invariance_check
from simchart.validation.scaling import daily_invariance_stats

S1 = dict(stage="S1", enable_msm=True, enable_slow_ou=True)


def test_msm_switch_process_is_resolution_independent() -> None:
    """同一シードなら MSM の切替過程 (時刻・値・個数) が解像度に依らず一致する。

    切替時刻を連続時間で生成しグリッドへ写像するだけ、という実装の直接検証。
    ビット単位の一致であり、統計的一致より遥かに強い。
    """
    a = run(Config(n_days=200, steps_per_day=390, **S1))
    b = run(Config(n_days=200, steps_per_day=1560, **S1))
    assert a.meta["l2"]["msm"]["switch_digest"] == b.meta["l2"]["msm"]["switch_digest"]
    assert a.meta["l2"]["msm"]["n_switches"] == b.meta["l2"]["msm"]["n_switches"]


def test_msm_grid_values_agree_on_shared_timestamps() -> None:
    """粗いグリッドの各時刻での MSM 値が、細かいグリッドの同時刻の値と一致する。"""
    coarse = run(Config(n_days=100, steps_per_day=390, **S1))
    fine = run(Config(n_days=100, steps_per_day=1950, **S1))
    sub_c = coarse.meta["l2"]["vol_subsample"]
    sub_f = fine.meta["l2"]["vol_subsample"]
    # サブサンプルは 60 秒間隔で保存される。両者の共通時刻で MSM 成分を比較する。
    t_c = np.round(sub_c["t_days"] * 86400).astype(np.int64)
    t_f = np.round(sub_f["t_days"] * 86400).astype(np.int64)
    common, idx_c, idx_f = np.intersect1d(t_c, t_f, return_indices=True)
    assert common.size > 1000
    np.testing.assert_array_equal(
        sub_c["half_log_msm"][idx_c], sub_f["half_log_msm"][idx_f]
    )


def test_ou_daily_autocorrelation_is_resolution_independent() -> None:
    """OU の 1 日ラグ自己相関が、どの刻みでも e^{-theta} に一致する。

    厳密離散化の解像度非依存の検証。実現**分散**での比較は遅い過程の標本ゆらぎ
    (数百日では ±50%) に埋もれて検定にならないため、ゆらぎの小さい自己相関で
    比較する。Euler-Maruyama なら減衰係数が (1 - theta*dt) に化けて、粗い刻み
    ほど理論からずれる。
    """
    theta = np.log(2.0) / 30.0
    expected = np.exp(-theta)
    for steps in (39, 390):
        rhos = []
        for seed in range(4):
            cfg = Config(seed=seed, n_days=2000, steps_per_day=steps, **S1)
            sub = run(cfg).meta["l2"]["vol_subsample"]
            x = np.asarray(sub["x_slow"])
            per_day = int(round(1.0 / (sub["t_days"][1] - sub["t_days"][0])))
            rhos.append(float(np.corrcoef(x[:-per_day], x[per_day:])[0, 1]))
        assert float(np.mean(rhos)) == pytest.approx(expected, abs=0.02), (
            steps, rhos,
        )


def test_daily_statistics_agree_across_resolutions() -> None:
    """2 解像度で日次統計 (尖度・GPH d・|r|ACF(1)・Var(log sigma)) が一致する。

    同一シードなら MSM 経路は完全一致するので、残る差は拡散乱数と OU 乱数の
    実現差だけ。トレランスはその実現差の大きさに合わせてある
    (本番判定は cli の scale_invariance ゲートが 23400 vs 390 で行う。
    テストは軽量ペアで同じロジックを固定する)。
    """
    config = Config(n_days=5000, steps_per_day=780, **S1)
    reference = run(config)
    report = scale_invariance_check(config, reference)
    assert report["passed"], report["checks"]
    assert report["checks"]["msm_switch_process_identical"]["passed"]


def test_daily_invariance_stats_shape() -> None:
    result = run(Config(n_days=300, steps_per_day=390, **S1))
    stats = daily_invariance_stats(result)
    assert stats["n_days"] == 300
    assert stats["var_log_vol"] is not None and stats["var_log_vol"] > 0
    assert stats["kurtosis_daily"] is not None


def test_per_step_switch_probability_would_fail() -> None:
    """対照実験: per-step 切替確率型の実装なら切替回数が解像度で 60 倍変わる。

    このテストは「検査が本当に欠陥を検出できるか」の帰無対照。物理時間定義の
    実装では切替回数の期待値が解像度に依らないことを確認し、per-step 型が
    出すはずの値 (解像度比で線形に増える) と区別できることを見る。
    """
    slow = run(Config(seed=7, n_days=500, steps_per_day=390, **S1))
    fast = run(Config(seed=7, n_days=500, steps_per_day=1950, **S1))
    n_slow = np.array(slow.meta["l2"]["msm"]["n_switches"])
    n_fast = np.array(fast.meta["l2"]["msm"]["n_switches"])
    # 物理時間定義: 完全一致。per-step 定義なら 5 倍になっていたはず。
    np.testing.assert_array_equal(n_slow, n_fast)
    assert n_slow.sum() > 0
