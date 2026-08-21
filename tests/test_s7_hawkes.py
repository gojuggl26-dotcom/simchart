"""S7: 符号対称 Hawkes 注文流のテスト。

中心は 4 つ:
1. **S6 経路のビット単位不変** — 本番設定を新カーネルで再実行し、保存済み
   ダイジェストと照合する (カーネル改修の主リスクは S6 の乱数消費列のずれ)。
2. **ゼロ励起の帰無対照** — a=0 なら (季節性 off で) 純 Poisson に退化する。
   thinning 実装の単位系 (β の 1/日換算・dt 変換) をレート復元で定量検証する。
3. **クラスタリングの存在** — 既定の較正で Fano・CV² が Poisson から明確に離れる。
4. **板の健全性** — バーストの下でも FIFO/保存則 (完全リプレイ) と流動性が保たれる。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from simchart import Config, run
from simchart.layers.l1_activity import HawkesActivity, build_activity
from simchart.types import EventType
from simchart.validation.engine import engine_invariants

ROOT = Path(__file__).resolve().parent.parent

# S6 のテストと同じ小規模スタック (S5 までの全成分 + 板 + Hawkes)。
S5KW = dict(
    enable_msm=True, enable_slow_ou=True, enable_rough=True,
    enable_jump=True, enable_leverage=True,
    enable_seasonality=True, enable_overnight=True, enable_chaos_vol=True,
    jump_lambda_per_year=5.0, jump_eta_down=35.0, jump_eta_up=56.0,
    jump_qv_share_target=0.12, jump_p_up=0.42,
    leverage_rho_rough=-0.60, leverage_rho_slow=-0.35,
)
SMALL = dict(n_days=20, steps_per_day=390)

#: 誤較正の回帰フィクスチャ: 初版 (v2) のパラメータ。取消の定常レートを
#: δ·N̄ = 4500/日 (実測は 1195/日) と誤仮定して較正したもので、実行すると
#: 励起駆動の取消が板を食い尽くし 42% の時間で片側が空になった。
INFEASIBLE_V2 = dict(
    hawkes_a=(
        (0.5763, 0.2096, 0.2620),
        (0.0524, 0.6078, 0.1572),
        (0.0524, 0.1257, 0.5763),
    ),
    hawkes_mu_mo=184.8, hawkes_mu_lo=116.8, hawkes_delta0=1.07,
    hawkes_nbar_ref=900.0,
)


def _hawkes_cfg(seed: int = 42, **extra) -> Config:
    kw = {**S5KW, **SMALL, **extra}  # extra が SMALL (n_days 等) を上書きできる
    return Config(
        stage="S7", seed=seed, enable_book=True, enable_hawkes=True,
        book_debug_invariants=True, **kw,
    )


@pytest.fixture(scope="module")
def hawkes_result():
    cfg = _hawkes_cfg()
    return run(cfg), cfg


def _nontrade_times(r) -> np.ndarray:
    ev = r.events
    return ev.t[ev.event_type != int(EventType.TRADE)]


def _fano_per_minute(t: np.ndarray, n_days: int, skip_days: int = 1) -> float:
    edges = np.arange(skip_days * 23400.0, n_days * 23400.0 + 60.0, 60.0)
    counts, _ = np.histogram(t[t >= edges[0]], bins=edges)
    return float(counts.var() / counts.mean())


# ---------------------------------------------------------------------------
# 1. S6 経路のビット単位不変 (最重要の回帰)
# ---------------------------------------------------------------------------
def test_s6_production_digest_bit_identical():
    """S6 本番設定 (500 日) を S7 カーネルで再実行 → 保存ダイジェストと一致。

    カーネルに Hawkes 分岐を足しても、use_hawkes=False の乱数消費列は 1 bit も
    変わっていないことの直接証明。これが通る限り S6 の全ゲート結果は有効なまま。
    """
    metrics_path = ROOT / "results" / "S6" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("S6 本番の results が無い環境ではスキップ")
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    want = stored["metrics"]["runtime"]["determinism"]["digest_first"]
    r = run(Config.load(ROOT / "configs" / "s6.yaml"))
    assert r.digest() == want, "S6 経路の出力が変わった — カーネル改修が S6 を汚染"


# ---------------------------------------------------------------------------
# 2. ゼロ励起の帰無対照 (thinning の単位系の定量検証)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def null_result():
    """a=0・季節性 off → MO/LO は厳密に定数レート Poisson。"""
    cfg = Config(
        stage="S7", seed=7, enable_book=True, enable_hawkes=True,
        hawkes_a=((0.0,) * 3,) * 3,
        n_days=5, steps_per_day=390,
    )
    return run(cfg), cfg


def test_zero_excitation_recovers_baseline_rates(null_result):
    """実現レート ≈ 2μ (thinning のレート単位が正しいことの直接検定)。"""
    r, cfg = null_result
    ev = r.events
    for name, code, mu in (
        ("MO", EventType.MARKET, cfg.hawkes_mu_mo),
        ("LO", EventType.LIMIT_ADD, cfg.hawkes_mu_lo),
    ):
        n = int((ev.event_type == int(code)).sum())
        lam = 2.0 * mu * cfg.n_days
        z = (n - lam) / math.sqrt(lam)  # Poisson の標準化残差
        assert abs(z) < 4.0, f"{name}: 件数 {n} vs 期待 {lam:.0f} (z={z:.1f})"


def test_zero_excitation_is_poisson(null_result):
    """MO+LO の分ごとの件数は Fano ≈ 1、イベント間隔は CV² ≈ 1。

    ★t=0 の板初期化 (init_levels×2 = 60 本の LIMIT_ADD 行) を必ず除外する。
    最初の実装はこれを含めて Fano=1.95 を出し「thinning のバグ」を疑ったが、
    実際は 1 ビンに 60 行の初期化スパイクが乗っただけだった (除外後 0.99)。
    """
    r, cfg = null_result
    ev = r.events
    keep = (ev.event_type == int(EventType.MARKET)) | (
        ev.event_type == int(EventType.LIMIT_ADD)
    )
    t = ev.t[keep]
    t = t[t > 0.0]  # 板初期化行の除外
    fano = _fano_per_minute(t, cfg.n_days, skip_days=0)
    assert abs(fano - 1.0) < 0.15, f"Fano={fano:.3f} (Poisson なら 1)"
    d = np.diff(t)
    d = d[d > 0]
    cv2 = float(d.var() / d.mean() ** 2)
    assert abs(cv2 - 1.0) < 0.2, f"CV²={cv2:.3f} (指数なら 1)"


# ---------------------------------------------------------------------------
# 3. クラスタリングの存在 (既定較正)
# ---------------------------------------------------------------------------
def test_overdispersion_present(hawkes_result):
    r, cfg = hawkes_result
    t = _nontrade_times(r)
    fano = _fano_per_minute(t, cfg.n_days)
    assert fano > 2.0, f"Fano={fano:.2f} — 自己励起が効いていない"
    d = np.diff(np.unique(t))
    cv2 = float(d.var() / d.mean() ** 2)
    assert cv2 > 1.5, f"CV²={cv2:.2f} — バーストが見えない"


def test_realized_rates_near_targets(hawkes_result):
    """実現レートは定常目標の ±12% (バーンイン 1 日除外)。

    完全一致は要求しない: CX ベースラインは実現 N(t) に比例し、N̄ref (S6 実測)
    からのずれが定常解ごと僅かにずらす。厳密な実測は本番 500 日で報告する。
    """
    r, cfg = hawkes_result
    targets = {"MO": 1800.0, "LO": 3000.0, "CX": 1195.0}
    ev = r.events
    days = cfg.n_days - 1
    keep = ev.t >= 23400.0
    for name, code in (("MO", EventType.MARKET), ("LO", EventType.LIMIT_ADD),
                       ("CX", EventType.CANCEL)):
        n = int(((ev.event_type == int(code)) & keep).sum())
        rate = n / days
        rel = rate / targets[name] - 1.0
        assert abs(rel) < 0.12, f"{name}: {rate:.0f}/日 vs 目標 {targets[name]:.0f} ({rel:+.1%})"


def test_phi_lambda_modulates_baseline(hawkes_result):
    """日内 U 字: u ビンごとの件数が φ_λ(u) と強く相関する (§3.3 の消費確認)。"""
    r, cfg = hawkes_result
    from simchart.layers.l0_calendar import build_calendar
    from simchart.rng import RNGRegistry

    cal = build_calendar(cfg, RNGRegistry(cfg.seed))
    t = _nontrade_times(r)
    t = t[t >= 23400.0]
    u = (t / 23400.0) % 1.0
    n_bins = 26
    counts, edges = np.histogram(u, bins=np.linspace(0.0, 1.0, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    phi = np.asarray(cal.phi_lambda_of_u(centers))
    c = float(np.corrcoef(counts, phi)[0, 1])
    assert c > 0.8, f"corr(件数, φ_λ) = {c:.3f} — 季節性がベースラインに乗っていない"


def test_side_symmetry(hawkes_result):
    """符号対称制約: 各型の買い/売り件数は二項 (p=1/2) の範囲 (|z| < 4)。"""
    r, _ = hawkes_result
    ev = r.events
    for name, code in (("MO", EventType.MARKET), ("LO", EventType.LIMIT_ADD)):
        m = ev.event_type == int(code)
        n = int(m.sum())
        n_buy = int((ev.side[m] == 1).sum())
        z = (n_buy - 0.5 * n) / math.sqrt(0.25 * n)
        assert abs(z) < 4.0, f"{name}: buy {n_buy}/{n} (z={z:.1f})"


# ---------------------------------------------------------------------------
# 4. 板の健全性 (バーストの下で)
# ---------------------------------------------------------------------------
def test_engine_invariants_under_hawkes(hawkes_result):
    r, _ = hawkes_result
    inv = engine_invariants(r.meta["l3"], r.events.t)
    assert inv["status"] == "ok"
    assert inv["all_passed"], inv


def test_full_replay_under_hawkes(hawkes_result):
    """完全リプレイ (FIFO・価格優先・部分約定・取消) がバースト下でも成立。"""
    from _book_replay import replay_and_verify

    r, _ = hawkes_result
    assert replay_and_verify(r.events) > 10_000


def test_guards_silent_and_liveness(hawkes_result):
    """健全な較正では全ガードが無発動、板の枯渇も実質ゼロ。"""
    r, cfg = hawkes_result
    d = r.meta["l3"]
    h = d["hawkes"]
    assert h["cap_hits"] == 0
    assert h["daycap_hits"] == 0
    assert h["cx_noop"] == 0
    assert 0.2 < h["acceptance_rate"] < 0.98  # 上界が壊れると 1.0 に張り付く
    from simchart.layers.book_engine import C_EMPTY_SIDE_TIME
    frac = d["counters"][C_EMPTY_SIDE_TIME] / (cfg.n_days * 23400.0)
    assert frac < 1e-3, f"片側空の時間比率 {frac:.2%} — 誤較正の兆候"


def test_throughput_maintained(hawkes_result):
    r, _ = hawkes_result
    assert r.meta["l3"]["throughput_events_per_sec"] > 50_000


def test_deterministic(hawkes_result):
    r, cfg = hawkes_result
    r2 = run(cfg)
    assert r.digest() == r2.digest()


# ---------------------------------------------------------------------------
# 設定の検証 (誤較正・爆発・板なしを入口で止める)
# ---------------------------------------------------------------------------
def test_config_rejects_infeasible_cancel_rate():
    """★回帰: 初版 v2 の較正 (r_CX=4500 > r_LO=3000) は入口で止まること。"""
    with pytest.raises(ValueError, match="r_CX"):
        _hawkes_cfg(**INFEASIBLE_V2)


def test_config_rejects_explosive_matrix():
    a = tuple(tuple(v * 1.25 for v in row) for row in Config().hawkes_a)
    with pytest.raises(ValueError, match="rho"):
        _hawkes_cfg(hawkes_a=a)


def test_config_rejects_slow_kernel():
    with pytest.raises(ValueError, match="1 時間"):
        _hawkes_cfg(hawkes_tau_seconds=(0.5, 10.0, 7200.0))


def test_config_rejects_bad_weights():
    with pytest.raises(ValueError, match="合計が 1"):
        _hawkes_cfg(hawkes_weights=(0.5, 0.3, 0.3))


def test_config_requires_book():
    with pytest.raises(ValueError, match="enable_book"):
        Config(stage="S7", enable_hawkes=True)


def test_without_book_strips_hawkes():
    cfg = _hawkes_cfg()
    base = cfg.without_book()
    assert base.enable_book is False and base.enable_hawkes is False
    assert base.hawkes_a == Config().hawkes_a  # 暗黙 no-op ガードに当たらない


def test_activity_layer_derivations():
    cfg = _hawkes_cfg()
    act = build_activity(cfg, None, None)  # rng は使われない
    assert isinstance(act, HawkesActivity)
    n = act.branching_ratio()
    assert 0.80 <= n <= 0.85
    r = act.stationary_rates()
    assert np.allclose(r, [1800.0, 3000.0, 1195.0], rtol=0.005), r
    with pytest.raises(NotImplementedError):
        act.event_times(0.0, 1.0)


def test_l2_bitwise_frozen_under_hawkes():
    """L2 (p*・σ) は Hawkes+板の on/off で 1 bit も動かない (小規模直接照合)。"""
    cfg = _hawkes_cfg(seed=99, n_days=5)
    r_on = run(cfg)
    r_off = run(cfg.without_book())
    assert np.array_equal(r_on.price.log_p_star, r_off.price.log_p_star)
    assert np.array_equal(r_on.price.log_vol, r_off.price.log_vol)
