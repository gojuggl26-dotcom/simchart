"""S11: RV フィードバックのテスト。

中心は 4 つ:
1. **S10 経路のビット単位不変** (enable_feedback=False は乗数 ×1.0 恒等 —
   digest 回帰は test_s10_* が固定。ここでは on の決定論を確認)。
2. **信号が「驚き」であること** (§2.1): u_t は平均 ≈ 0 で、ボラの**水準**とは
   ほぼ無相関 (水準反応だと高ボラ期に恒常的な危機モードになる)。
3. **チャネルの向き**: u が高い局面でスプレッドが広い (δ↑・Δ↑ の帰結)。
4. **有界性** (§3.2/§3.3): n_t ≤ n_max、乗数は [e^-b, e^+b]。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart import Config, run
from test_s10_coupling import S, _s10_cfg


def _s11_cfg(seed: int = 1201, n_days: int = 40, **extra) -> Config:
    kw = dict(
        stage="S11", seed=seed, n_days=n_days, kappa=0.2, c_vol=0.65,
        sigma_bar=0.2217, enable_feedback=True,
        fb_b_delta=0.5, fb_b_place=0.3, fb_u_scale=1.0,
    )
    kw.update(extra)
    return _s10_cfg(**kw)


@pytest.fixture(scope="module")
def fb_run():
    cfg = _s11_cfg()
    return run(cfg), cfg


def test_surprise_signal_stationary(fb_run):
    """u_t: 平均 ≈ 0 (§2.1)、ボラ水準 (log RV_long 相当の緩慢成分) と低相関。"""
    r, cfg = fb_run
    u = np.asarray(r.events.meta["fb_u_grid"], dtype=np.float64)
    step = float(r.events.meta["fb_u_step_sec"])
    burn = int(cfg.book_burn_in_days * S / step)
    u_b = u[burn:]
    # 中心化後の平均 ≈ 0 (±0.5: 定数化した u0 のシード間散らばり ±0.18 +
    # フィードバック自身による分布シフト。本番の多シード中央値ゲートが水準を判定)
    assert abs(float(u_b.mean())) < 0.5, u_b.mean()
    assert 0.05 < float(u_b.std()) < 3.0, u_b.std()
    # 水準盲目性: 潜在 log σ (日次平均) と u (日次平均) の相関が小さい
    spd_u = int(round(S / step))
    n_days_u = u_b.size // spd_u
    u_d = u_b[: n_days_u * spd_u].reshape(n_days_u, spd_u).mean(axis=1)
    spd_g = int(round(S / (r.price.t[1] - r.price.t[0])))
    burn_d = int(cfg.book_burn_in_days)
    lv_d = r.price.log_vol[burn_d * spd_g:][: n_days_u * spd_g]
    lv_d = lv_d.reshape(n_days_u, spd_g).mean(axis=1)
    c = float(np.corrcoef(u_d, lv_d)[0, 1])
    assert abs(c) < 0.45, f"u がボラ水準に反応している (corr={c:.3f})"


def test_channels_widen_spread_on_surprise(fb_run):
    """u 高値域でスプレッドが広い (δ_t↑ で板が薄く、Δ_scale↑ で補充が遠い)。"""
    r, cfg = fb_run
    u = np.asarray(r.events.meta["fb_u_grid"], dtype=np.float64)
    step = float(r.events.meta["fb_u_step_sec"])
    burn = int(cfg.book_burn_in_days * S / step)
    bb = np.asarray(r.book.bid_px[:, 0], dtype=np.float64)
    ba = np.asarray(r.book.ask_px[:, 0], dtype=np.float64)
    bb_c = bb[burn: u.size]
    ba_c = ba[burn: u.size]
    sp = ba_c - bb_c
    u_b = u[burn: burn + sp.size]
    ok = (bb_c >= 0) & (ba_c >= 0) & (sp > 0)
    hi = u_b > np.quantile(u_b[ok], 0.9)
    lo = u_b < np.quantile(u_b[ok], 0.1)
    sp_hi = float(np.median(sp[ok & hi]))
    sp_lo = float(np.median(sp[ok & lo]))
    assert sp_hi > sp_lo, (sp_hi, sp_lo)


def test_nt_bounded_and_diagnostics(fb_run):
    r, cfg = fb_run
    # n チャネル無効 (fb_b_n=0) では n_t ≡ n_design
    fb = r.meta["l3"]["feedback"]
    assert abs(fb["nt_mean"] - 0.8300) < 0.01
    assert fb["nt_max"] < 0.97
    r2 = run(_s11_cfg(fb_b_n=2.0))
    fb2 = r2.meta["l3"]["feedback"]
    assert fb2["nt_max"] <= cfg.fb_n_max + 1e-9
    assert cfg.fb_n_min - 1e-9 <= fb2["nt_mean"] <= cfg.fb_n_max + 1e-9


def test_deterministic(fb_run):
    r, cfg = fb_run
    r2 = run(cfg)
    assert r.digest() == r2.digest()


def test_config_validation():
    with pytest.raises(ValueError, match="enable_book"):
        Config(stage="S11", enable_feedback=True, fb_b_delta=0.5)
    with pytest.raises(ValueError, match="全チャネル"):
        _s11_cfg(fb_b_delta=0.0, fb_b_place=0.0, fb_b_n=0.0)
    with pytest.raises(ValueError, match="0.97"):
        _s11_cfg(fb_n_max=0.97)
    with pytest.raises(ValueError, match="no-op|既定値"):
        _s10_cfg(stage="S11", kappa=0.2, fb_b_delta=0.5)
    base = _s11_cfg().without_book()
    assert base.enable_feedback is False and base.fb_b_delta == 0.0
