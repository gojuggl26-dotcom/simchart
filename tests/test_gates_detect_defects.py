"""ゲートが**落ちる能力を持つ**ことの確認 (帰無対照)。

合格するだけのゲートは無価値である。既知の欠陥を仕込んだ系列を食わせて、
**狙ったゲートだけが落ちる**ことを確かめる。ここが緑でなければ「S0 が全ゲート
合格した」という主張自体に意味が無い。

仕込む欠陥と、落ちるべきゲート:

===============================  ==========================================
欠陥                             落ちるべきゲート
===============================  ==========================================
MA(1) の系列相関                 acf_r_lag1, ljung_box
t 分布革新 (自由度 6)            kurtosis, kurtosis_flat
ボラティリティ・クラスタリング   acf_abs_r_lag1, kurtosis (acf_r_lag1 は通る)
定常 AR(1) の対数価格            adf, variance_ratio
マイクロストラクチャー・ノイズ   signature_plot_flat, acf_r_lag1
検証関数の例外                   validation_callable
===============================  ==========================================

速度のため検証設定は本番より小さい (基準粒度を刻みに一致させ、判定に使う
スケールの標本数下限を下げてある)。確かめているのは推定量とゲートの**接続**で
あって、本番の標本数における検出力ではない。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simchart.config import Config, ValidationConfig
from simchart.types import (
    BookSnapshot,
    EventLog,
    Observation,
    PriceProcess,
    StageResult,
)
from simchart.validation import evaluate
from simchart.validation.gates import S0_GATES
from simchart.validation.suite import collect_errors, run_all

N_DAYS = 200
STEPS_PER_DAY = 390  # 刻み = 23400 / 390 = 60 秒 (基準粒度と一致させる)
SESSION_SECONDS = 23400.0
STEP_SECONDS = SESSION_SECONDS / STEPS_PER_DAY

TEST_VALIDATION = ValidationConfig(
    primary_bar_sec=60,
    scales_sec=(60, 120, 300, 600, 900, 1800),
    min_obs_for_gate=5_000,
    acf_max_lag=50,
    acf_abs_max_lag=200,
    ljung_box_lags=(1, 5, 10, 20),
    ljung_box_primary_lag=20,
    gph_bandwidth_profile=(0.65,),
    vr_qs=(2, 4, 8, 16, 32),
    micro_max_lag=50,
    micro_fit_lag_range=(5, 50),
)

TEST_CONFIG = Config(
    n_days=N_DAYS, steps_per_day=STEPS_PER_DAY, validation=TEST_VALIDATION
)

#: 1 ステップの対数リターン標準偏差 (年率 20% 相当)。
SIGMA_STEP = TEST_CONFIG.sigma_step


def _make_result(returns: np.ndarray, noise: np.ndarray | None = None) -> StageResult:
    """任意のリターン列から検証スイートに食わせられる結果を組み立てる。"""
    n = returns.size
    log_p = np.empty(n + 1, dtype=np.float64)
    log_p[0] = math.log(TEST_CONFIG.p0)
    np.cumsum(returns, out=log_p[1:])
    log_p[1:] += log_p[0]

    t = np.arange(n + 1, dtype=np.float64) * STEP_SECONDS
    observed = log_p if noise is None else log_p + noise
    price = PriceProcess(
        t=t, log_p_star=log_p, log_vol=np.full(n + 1, math.log(TEST_CONFIG.sigma_bar))
    )
    observation = Observation(
        t=t,
        log_price=observed,
        session_seconds=SESSION_SECONDS,
        step_seconds=STEP_SECONDS,
        source="synthetic-defect",
    )
    return StageResult(
        stage="S0",
        config=TEST_CONFIG,
        price=price,
        events=EventLog.empty(),
        book=BookSnapshot.empty(),
        observation=observation,
        runtime_sec=0.0,
    )


def _gate_status(result: StageResult) -> dict[str, bool]:
    """検証 -> ゲート判定を通し、ゲート名 -> 合否を返す。

    実行時系のゲート (決定性など) は今回の関心ではないので、全て合格として与える。
    ここで見たいのは統計量に基づくゲートの検出力だけ。
    """
    metrics = run_all(result, TEST_CONFIG)
    metrics["runtime"] = {
        "pipeline": {"completed": True},
        "determinism": {"bitwise_identical": True},
        "rng_stability": {"unchanged": True, "streams_distinct": True},
        "validation": {"all_callable": not collect_errors(metrics)},
        "artifacts": {"metrics_json_ok": True},
    }
    return {g.name: g.passed for g in evaluate(S0_GATES, metrics)}


def _gbm(rng: np.random.Generator, n: int) -> np.ndarray:
    return SIGMA_STEP * rng.standard_normal(n)


N = N_DAYS * STEPS_PER_DAY


@pytest.fixture(scope="module")
def control() -> dict[str, bool]:
    """欠陥のない対照。これが全合格でなければ以下のテストは意味を持たない。"""
    rng = np.random.default_rng(20260818)
    return _gate_status(_make_result(_gbm(rng, N)))


def test_control_passes_every_gate(control: dict[str, bool]) -> None:
    failed = [name for name, passed in control.items() if not passed]
    assert not failed, f"対照が落ちています: {failed}"


def test_ma1_autocorrelation_is_detected(control: dict[str, bool]) -> None:
    rng = np.random.default_rng(1)
    innovation = _gbm(rng, N + 1)
    returns = innovation[1:] + 0.10 * innovation[:-1]
    status = _gate_status(_make_result(returns))

    assert status["acf_r_lag1"] is False
    assert status["ljung_box"] is False
    # 正規革新のままなのでテール側は影響を受けない。
    assert status["kurtosis"] is True


def test_fat_tailed_innovations_are_detected() -> None:
    """自由度 6 の t 分布 (母集団尖度 6) を入れると尖度ゲートが落ちること。

    S0 でテールを外生的に入れてはならない、という禁止事項が機械的に守られる
    ことの確認でもある。
    """
    rng = np.random.default_rng(2)
    df = 6
    raw = rng.standard_t(df, size=N)
    returns = SIGMA_STEP * raw / math.sqrt(df / (df - 2))
    status = _gate_status(_make_result(returns))

    assert status["kurtosis"] is False
    assert status["kurtosis_flat"] is False
    assert status["acf_r_lag1"] is True  # 系列相関は入れていない


def test_volatility_clustering_is_detected() -> None:
    """|r| の記憶だけが落ち、r の無相関は保たれること。

    「リターンは無相関だがボラは持続する」という実データの性質を、検証スイートが
    2 つの別々のゲートとして切り分けられていることの確認。
    """
    rng = np.random.default_rng(3)
    log_vol = np.empty(N)
    log_vol[0] = 0.0
    innovation = rng.standard_normal(N)
    for i in range(1, N):
        log_vol[i] = 0.995 * log_vol[i - 1] + 0.05 * innovation[i]
    returns = SIGMA_STEP * np.exp(log_vol - log_vol.var() / 2) * rng.standard_normal(N)
    status = _gate_status(_make_result(returns))

    assert status["acf_r_lag1"] is True
    assert status["acf_abs_r_lag1"] is False
    assert status["kurtosis"] is False


def test_stationary_log_price_is_detected() -> None:
    """対数価格が定常 AR(1) になっていると ADF と分散比が落ちること。"""
    rng = np.random.default_rng(4)
    phi = 0.99
    shock = SIGMA_STEP * rng.standard_normal(N + 1)
    level = np.empty(N + 1)
    level[0] = 0.0
    for i in range(1, N + 1):
        level[i] = phi * level[i - 1] + shock[i]
    returns = np.diff(level)
    status = _gate_status(_make_result(returns))

    assert status["adf"] is False
    assert status["variance_ratio"] is False


def test_microstructure_noise_is_detected() -> None:
    """観測にノイズを乗せると signature plot が平坦でなくなること。

    S9 で uncertainty zones を入れたときに現れるべき現象を、S0 の時点で
    検出できる状態にしてある、という確認。
    """
    rng = np.random.default_rng(5)
    returns = _gbm(rng, N)
    noise = 0.5 * SIGMA_STEP * rng.standard_normal(N + 1)
    status = _gate_status(_make_result(returns, noise=noise))

    assert status["signature_plot_flat"] is False
    assert status["acf_r_lag1"] is False


def test_broken_estimator_trips_validation_callable(monkeypatch) -> None:
    """検証関数が例外を投げたら validation_callable が落ちること。

    例外が握り潰されて「測れなかったのに合格」になるのが最悪なので、
    経路そのものを試す。
    """
    from simchart.validation import suite as suite_module

    def explode(*_args, **_kwargs):
        raise RuntimeError("意図的な故障")

    monkeypatch.setattr(suite_module.tails, "basic_moments", explode)

    rng = np.random.default_rng(6)
    result = _make_result(_gbm(rng, N))
    metrics = run_all(result, TEST_CONFIG)
    errors = collect_errors(metrics)
    assert errors and errors[0]["path"] == "tails.moments"

    metrics["runtime"] = {
        "pipeline": {"completed": True},
        "determinism": {"bitwise_identical": True},
        "rng_stability": {"unchanged": True, "streams_distinct": True},
        "validation": {"all_callable": not errors},
        "artifacts": {"metrics_json_ok": True},
    }
    status = {g.name: g.passed for g in evaluate(S0_GATES, metrics)}
    assert status["validation_callable"] is False
    # 指標そのものが取れないゲートも巻き添えで落ちる (静かに合格しない)。
    assert status["kurtosis"] is False


def test_missing_metric_fails_the_gate() -> None:
    """指標のパスが存在しないとき、黙って合格しないこと。"""
    results = evaluate(S0_GATES, {"tails": {}})
    assert all(not g.passed for g in results)
    assert all(g.error for g in results)
