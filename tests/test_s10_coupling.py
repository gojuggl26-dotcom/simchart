"""S10: κ 結合のテスト。

中心は 4 つ:
1. **S9 経路のビット単位不変** (κ=0 の乱数消費列は完全一致 — p_up=0.5 の比較は
   定数 0.5 と同一)。
2. **バイアスの向き**: メタオーダー符号が生成時の乖離 d = log p* − log mid と
   正に相関する (κ=0 では無相関)。
3. **d の定常化**: κ>0 で乖離が平均回帰し、κ=0 では漂流する。
4. **子の継承**: バイアスは生成時のみで、γ (run length 構造) が壊れない。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simchart import Config, run

ROOT = Path(__file__).resolve().parent.parent
S = 23400.0


def _s10_cfg(seed: int = 1001, n_days: int = 60, kappa: float = 0.5, **extra) -> Config:
    cfg = Config.load(ROOT / "configs" / "s9.yaml")
    kw = dict(stage="S10", seed=seed, n_days=n_days, kappa=kappa)
    kw.update(extra)
    return cfg.replace(**kw)


def test_s9_production_digest_bit_identical():
    metrics_path = ROOT / "results" / "S9" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("S9 本番の results が無い環境ではスキップ")
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    want = stored["metrics"]["runtime"]["determinism"]["digest_first"]
    r = run(Config.load(ROOT / "configs" / "s9.yaml"))
    assert r.digest() == want, "S9 経路の出力が変わった — S10 改修が前段を汚染"


@pytest.fixture(scope="module")
def coupled():
    cfg = _s10_cfg()
    return run(cfg), cfg


def _sign_gap_corr(r, cfg):
    """メタオーダー符号と生成時乖離の相関 (初子直前ミッドで近似)。"""
    mo = r.events.meta["metaorders"]
    started = (mo["n_exec"] > 0) & (mo["t_first"] >= cfg.book_burn_in_days * S)
    mid = mo["mid_first"][started]
    t_first = mo["t_first"][started]
    sgn = mo["sign"][started]
    tick = r.events.meta["tick_size"]
    base = r.events.meta["base_price"]
    lm = np.log(base + tick * mid)
    obs = r.observation
    idx = np.clip((t_first / obs.step_seconds).astype(np.int64), 0,
                  r.price.log_p_star.size - 1)
    d = r.price.log_p_star[idx] - lm
    return float(np.corrcoef(sgn, d)[0, 1])


def test_sign_biased_toward_pstar(coupled):
    r, cfg = coupled
    c = _sign_gap_corr(r, cfg)
    assert c > 0.10, c
    # κ=0 (S9 相当) では無相関
    r0 = run(Config.load(ROOT / "configs" / "s9.yaml").replace(seed=1001, n_days=60))
    c0 = _sign_gap_corr(r0, Config.load(ROOT / "configs" / "s9.yaml").replace(
        seed=1001, n_days=60))
    assert abs(c0) < 0.05, c0


def test_gap_becomes_stationary(coupled):
    r, cfg = coupled
    obs = r.observation
    burn = int(cfg.book_burn_in_days * S / obs.step_seconds)
    d = (r.price.log_p_star - obs.log_price)[burn:]
    stride = max(1, int(round(60.0 / obs.step_seconds)))
    dm = d[::stride]
    phi = float(np.corrcoef(dm[:-1], dm[1:])[0, 1])
    hl = -np.log(2) / np.log(phi)
    assert hl < 600, hl  # 分オーダーの半減期 (κ=0.5 実測 ~100 分)
    assert d.std() * 1e4 < 120  # 乖離 SD が有界 (κ=0 実測 240bp)


def test_gamma_survives_coupling(coupled):
    """子は親符号を継承 (§2.2) — run length 構造 = γ が残る。

    ★raw の C(ℓ) には情報チャネル (追跡ハーディングのこぶ) が重畳する — それは
    バグではなく結合の物理。分割構造の保存は**生成時バイアス E[ε|d] を引いた
    残差符号**で判定する (S10a 実測: 残差 γ̂ = 0.611 = S8 の 0.614)。
    """
    from simchart.validation.memory import acf_powerlaw_fit

    r, cfg = coupled
    s_arr = np.asarray(r.events.meta["agg_trade_side"], dtype=np.float64)
    t = np.asarray(r.events.meta["agg_trade_t"])
    mt = np.asarray(r.events.meta["agg_trade_meta"], dtype=np.float64)
    keep = t >= cfg.book_burn_in_days * S
    s_k = s_arr[keep]
    # raw: こぶ込みで正の相関が残っていること (存在確認)
    d = s_k - s_k.mean()
    c1 = float(d[:-1] @ d[1:]) / float(d @ d)
    assert 0.10 < c1 < 0.35, c1
    # 残差: 分割構造の保存 (量は本番の中央値ゲート)
    mo = r.events.meta["metaorders"]
    tick = r.events.meta["tick_size"]
    bp = r.events.meta["base_price"]
    obs = r.observation
    idx = np.clip((mo["t_first"] / obs.step_seconds).astype(np.int64), 0,
                  r.price.log_p_star.size - 1)
    sgrid = np.exp(r.price.log_vol) * np.sqrt(
        cfg.kappa_tau_meta_sec / (252.0 * S)
    )
    mu_meta = np.tanh(
        cfg.kappa
        * (r.price.log_p_star[idx] - np.log(bp + tick * mo["mid_first"]))
        / sgrid[idx]
    )
    mt_k = mt[keep].astype(np.int64)
    mu_row = np.where(mt_k >= 0, mu_meta[np.clip(mt_k, 0, mu_meta.size - 1)], 0.0)
    fit = acf_powerlaw_fit(s_k - mu_row, (2, 400), max_lag=400)
    assert fit["status"] == "ok"
    assert 0.35 < fit["gamma"] < 0.95, fit["gamma"]


def test_deterministic(coupled):
    r, cfg = coupled
    r2 = run(cfg)
    assert r.digest() == r2.digest()


def test_config_validation():
    with pytest.raises(ValueError, match="enable_metaorder"):
        Config(stage="S10", kappa=0.5)
    with pytest.raises(ValueError, match="負"):
        _s10_cfg(kappa=-0.1)
    with pytest.raises(ValueError, match="no-op"):
        Config(stage="S10", kappa_tau_meta_sec=100.0)
    base = _s10_cfg().without_book()
    assert base.kappa == 0.0