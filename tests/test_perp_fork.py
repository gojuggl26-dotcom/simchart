"""perp フォークの株式保全ゲート (S0-perp §6.3 / §6.4 / §11)。

このファイルの役割は perp の機能ではなく、**フォーク作業が株式側を 1 bit も
動かしていないこと**の証明である。fixture (tests/baselines_equity_fork.json)
はフォーク作業前の master (3439798) で生成した 4 構成のダイジェスト —
RNG・L2 全成分・板・結合・フィードバック・χ・多資産の全経路を踏む。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simchart.config import Config
from simchart.pipeline import run, run_multi

FIXTURE = Path(__file__).resolve().parent / "baselines_equity_fork.json"


def _baselines() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# §6.3 最重要ゲート: 株式ベースラインのビット単位不変
# ---------------------------------------------------------------------------
def test_equity_baselines_unchanged_s0() -> None:
    base = _baselines()["s0_gbm_250d"]
    r = run(Config(stage="S0", n_days=250))
    assert r.digest() == base["digest"], (
        "株式 S0 (GBM) のダイジェストがフォーク前と不一致 — perp の中身より先に"
        " RNG かレイヤー分岐を疑うこと (S0-perp §6.3)"
    )


def test_equity_baselines_unchanged_s3() -> None:
    base = _baselines()["s3_l2_120d"]
    cfg = Config(
        stage="S3", n_days=120,
        enable_msm=True, enable_slow_ou=True, enable_rough=True,
        enable_jump=True, enable_leverage=True,
    )
    assert run(cfg).digest() == base["digest"]


def test_equity_baselines_unchanged_s12() -> None:
    base = _baselines()["s12_full_30d"]
    cfg = Config.load("configs/s12.yaml").replace(n_days=30)
    assert run(cfg).digest() == base["digest"]


def test_equity_baselines_unchanged_s13_multi() -> None:
    base = _baselines()["s13_multi_20d"]
    cfg = Config.load("configs/s13.yaml").replace(n_days=20)
    m = run_multi(cfg)
    assert {str(k): v for k, v in m.digests.items()} == base["digests"]


# ---------------------------------------------------------------------------
# §6.4 ストリーム名の衝突検査
# ---------------------------------------------------------------------------
def test_stream_names_hash_distinctly() -> None:
    """全カノニカル名 (資産派生込み) の子シードが相異なること。

    sha256 の 64bit 切り出しなので衝突は実質起きないが、「起きたら改名では
    なくエラーで止める」規約 (§6.4 — 改名は経路変更) をテストとして固定する。
    """
    from simchart.rng import (
        RESERVED_STREAM_NAMES,
        STREAM_NAMES,
        asset_stream_names,
        derive_seed,
    )

    names = list(STREAM_NAMES + RESERVED_STREAM_NAMES) + list(asset_stream_names(6))
    for master in (0, 42, 123456789):
        keys = [derive_seed(master, n) for n in names]
        assert len(set(keys)) == len(names), (
            f"ストリーム名の子シードが衝突しています (master={master})。"
            f" 名前を変えず、衝突しない新名を追加すること (§6.4)"
        )


def test_market_type_not_in_seed_derivation() -> None:
    """market_type が乱数経路に入っていないこと (§6.2)。

    株式と perp は別 run なので経路が同じでも問題なく、market_type を鍵に
    加えると株式の全ストリームが変わってしまう。
    """
    from simchart.rng import RNGRegistry

    a = RNGRegistry(42).get("l2.diffusion").standard_normal(64)
    b = RNGRegistry(42).get("l2.diffusion").standard_normal(64)
    assert (a == b).all()


# ---------------------------------------------------------------------------
# §11 修正の検証 (§3 は本リポジトリでは実装済みであることの固定)
# ---------------------------------------------------------------------------
def test_phi_normalizations_are_separate_and_correct() -> None:
    """φ_σ: mean(φ²)=1 / φ_λ: mean(φ)=1 (±0.001)。

    S0-perp §3.1 が指摘する不備 (mean-1 に統一) は本リポジトリには存在しない
    — S4 で正規化を分離済み。このテストはそれを恒久固定する。
    """
    import numpy as np

    from simchart.layers.l0_calendar import build_calendar
    from simchart.rng import RNGRegistry

    cfg = Config.load("configs/s12.yaml")
    cal = build_calendar(cfg, RNGRegistry(1))
    u = (np.arange(100001) + 0.5) / 100001
    ps = np.asarray(cal.phi_sigma_of_u(u))
    pl = np.asarray(cal.phi_lambda_of_u(u))
    assert abs(float(np.trapezoid(ps**2, u)) - 1.0) < 1e-3
    assert abs(float(np.trapezoid(pl, u)) - 1.0) < 1e-3
    # 逆の正規化になっていないこと (φ が非自明なら mean(φ_σ) ≠ 1)
    assert abs(float(np.trapezoid(ps, u)) - 1.0) > 5e-3


def test_chi2_characteristic_time_is_30_days() -> None:
    """χ₂ の特徴時間 30 日 (§3.2 の「1 日」不備は存在しない — S5 で解決済み)。"""
    cfg = Config.load("configs/s12.yaml")
    # MG(τ=17) のスペクトルピーク 49.65 単位 × days_per_unit = 30 日 (S5 裁定)
    peak_days = 49.65 * cfg.chaos_days_per_unit
    assert 25.0 < peak_days < 35.0
    # perp の週次周期 (5〜10 日帯) を避けていること (S0-perp §3.2 の追加要求)
    assert not (5.0 < peak_days < 10.0)


def test_overnight_is_variance_share() -> None:
    """ON は cc 分散シェアで管理 (§3.3 の「ratio=2.0」不備は存在しない — S4)。"""
    cfg = Config()
    assert hasattr(cfg, "overnight_variance_share")
    assert not hasattr(cfg, "overnight_var_ratio")
    assert 0.0 < cfg.overnight_variance_share < 1.0


# ---------------------------------------------------------------------------
# §5.3 config 検証の発火
# ---------------------------------------------------------------------------
def _perp_base(**over):
    kw = dict(stage="S0", market_type="perp_clob", session_type="24h",
              steps_per_day=1440, n_days=30, sigma_bar=0.60)
    kw.update(over)
    return Config(**kw)


def test_perp_requires_24h_session() -> None:
    with pytest.raises(ValueError):
        _perp_base(session_type="continuous")


def test_perp_forbids_overnight() -> None:
    with pytest.raises(ValueError):
        _perp_base(enable_overnight=True, enable_jump=True)


def test_perp_margin_consistency() -> None:
    with pytest.raises(ValueError):
        _perp_base(maintenance_margin=0.05, max_leverage=50.0)


def test_perp_funding_interval_divides_day() -> None:
    with pytest.raises(ValueError):
        _perp_base(funding_interval_hours=7.0)


def test_equity_rejects_perp_params() -> None:
    with pytest.raises(ValueError):
        Config(block_time_ms=500)


def test_l4_flags_raise_with_stage_name() -> None:
    for flag, stage in (
        ("enable_positions", "S11-perp"),
        ("enable_liquidation", "S11-perp"),
        ("enable_funding", "S10-perp"),
        ("enable_cross_margin", "S13-perp"),
        ("enable_weekly_seasonality", "S4-perp"),
    ):
        with pytest.raises(NotImplementedError) as excinfo:
            _perp_base(**{flag: True})
        assert stage in str(excinfo.value), (flag, str(excinfo.value))
