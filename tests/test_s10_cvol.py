"""S10c: c_vol (緩慢ボラ → 活動度 Z_t) のテスト。

中心は 3 つ:
1. **リンクの向き**: 高ボラ日ほどイベント数が多い (日次相関 > 0)。
2. **平均レートの保存**: E[Z] ≈ 1 (理論 Jensen 正規化) で総イベント数が
   c_vol=0 と大きくずれない。
3. **thinning の厳密性**: 打ち切り機構が働き、決定論が保たれる。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, run
from test_s10_coupling import S, _s10_cfg

CV = 0.5


def _cvol_cfg(seed: int = 1101, n_days: int = 40, **extra) -> Config:
    extra.setdefault("c_vol", CV)
    return _s10_cfg(seed=seed, n_days=n_days, kappa=0.2, **extra)


@pytest.fixture(scope="module")
def cvol_run():
    cfg = _cvol_cfg()
    return run(cfg), cfg


def _daily_counts(r, cfg):
    t = np.asarray(r.events.t)
    burn = int(cfg.book_burn_in_days)
    day = (t / S).astype(np.int64)
    n_days = int(cfg.n_days)
    cnt = np.bincount(day, minlength=n_days)[burn:n_days]
    return cnt.astype(np.float64)


def test_volume_follows_slow_vol(cvol_run):
    """日次イベント数と日次平均 log σ (3日 MA 相当の緩慢成分) が正相関。"""
    r, cfg = cvol_run
    cnt = _daily_counts(r, cfg)
    obs = r.observation
    spd = int(round(S / obs.step_seconds))
    lv = r.price.log_vol[: spd * int(cfg.n_days)].reshape(int(cfg.n_days), spd)
    lv_day = lv.mean(axis=1)[int(cfg.book_burn_in_days):]
    c = float(np.corrcoef(cnt, lv_day)[0, 1])
    assert c > 0.3, c
    # c_vol=0 の対照: 活動度はボラと独立
    cfg0 = _s10_cfg(seed=1101, n_days=40, kappa=0.2)
    r0 = run(cfg0)
    c0 = float(np.corrcoef(_daily_counts(r0, cfg0), lv_day)[0, 1])
    assert abs(c0) < abs(c), (c0, c)


def test_mean_rate_follows_z(cvol_run):
    """レートは Z に比例する (機構の検証)。

    ★絶対水準 E[Z]=1 は短ホライズンでは成立しない — 緩慢成分の実現平均
    (エポック効果) が実行ごとに ±20% 動く (実測: 250 日でも SD(mean V·s_V)≈0.22)。
    そこで「イベント数比 ≈ 実現 z_mean」という機構をテストし、水準は本番の
    多シード中央値ゲート (hawkes_realized_rates) に委ねる。
    """
    r, cfg = cvol_run
    r0 = run(_s10_cfg(seed=1101, n_days=40, kappa=0.2))
    ratio = len(r.events.t) / len(r0.events.t)
    z_mean = r.meta["l3"]["cvol"]["z_mean"]
    assert 0.90 < ratio / z_mean < 1.10, (ratio, z_mean)
    assert 0.5 < z_mean < 1.6, z_mean


def test_thinning_bound_and_determinism(cvol_run):
    r, cfg = cvol_run
    diag = r.meta["l3"]["cvol"]
    # 打ち切りは稀 (上界有効域 8h ≫ 平均候補間隔)
    assert diag["truncations"] < 1000
    assert diag["z_min"] > 0.0
    r2 = run(cfg)
    assert r.digest() == r2.digest()


def test_config_validation():
    with pytest.raises(ValueError, match="負"):
        _cvol_cfg(c_vol=-0.5)
    with pytest.raises(ValueError, match="enable_hawkes"):
        Config(stage="S10", c_vol=0.5)
    with pytest.raises(ValueError, match="no-op"):
        Config(stage="S10", c_vol_ma_days=5.0)
    base = _cvol_cfg().without_book()
    assert base.c_vol == 0.0 and base.c_vol_ma_days == 3.0
