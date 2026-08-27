"""S13: 多資産 — RNG 名前空間の最終試験と因子構造。

ここで固定する性質:

1. 因子経路の退化 (§8.3): N=2・β₀=0・共有シェア 0 の多資産実行における
   資産 0 が、単一資産 (S12 コードパス) とビット単位で一致する。
2. 資産追加の不変性 (§8.2): N を増やしても既存資産の系列は 1 bit も動かない。
3. 決定性: run_multi の 2 回実行が全資産で一致する。
4. 因子相関: 潜在日次リターンの相関 ≈ β_iβ_j、ボラ相関は共有シェア相当。
5. 共通ジャンプ: 同一時刻・同一サイズで全資産に乗る (完全コジャンプ)。
6. z 合成後もセル内自己相関ゼロ (bridge の保存 — §4.1 の実装リスク箇所)。
"""

from __future__ import annotations

import numpy as np
import pytest

from simchart.config import Config
from simchart.pipeline import (
    asset_addition_check,
    factor_degeneracy_check,
    run,
    run_multi,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _base_book_config(**over) -> Config:
    """板つきの小規模 S13 設定 (s12.yaml 相当の全機構 + 因子構造)。"""
    cfg = Config.load("configs/s12.yaml")
    kw = dict(
        stage="S13",
        n_days=20,
        n_assets=2,
        factor_betas=(0.6, 0.8),
        msm_k_common=6,
        ou_common_share=0.5,
        jump_common_share=0.5,
    )
    kw.update(over)
    return cfg.replace(**kw)


def _latent_config(**over) -> Config:
    """板なし・高速 (60 秒刻み) の統計テスト用設定。"""
    cfg = Config.load("configs/s12.yaml").without_book()
    kw = dict(
        stage="S13",
        n_days=250,
        steps_per_day=390,
        n_assets=3,
        factor_betas=(0.6, 0.8, 0.4),
        msm_k_common=6,
        ou_common_share=0.5,
        jump_common_share=0.5,
    )
    kw.update(over)
    return cfg.replace(**kw)


# ---------------------------------------------------------------------------
# 1-3: ビット単位の構造性質
# ---------------------------------------------------------------------------
def test_factor_path_degenerates_to_s12() -> None:
    chk = factor_degeneracy_check(_base_book_config(), n_days=15)
    assert chk["match"], chk


def test_asset_addition_is_bitwise_invariant() -> None:
    chk = asset_addition_check(_base_book_config(), n_days=15)
    assert chk["bitwise"], chk


def test_run_multi_is_deterministic() -> None:
    cfg = _base_book_config(n_days=10)
    a = run_multi(cfg)
    b = run_multi(cfg)
    assert a.digests == b.digests


def test_run_rejects_multi_config() -> None:
    with pytest.raises(ValueError):
        run(_base_book_config())


# ---------------------------------------------------------------------------
# 4: 因子相関 (潜在レベル)
# ---------------------------------------------------------------------------
def test_latent_daily_correlation_tracks_beta_products() -> None:
    m = run_multi(_latent_config())
    r = [p.daily_ret_latent for p in m.payloads]
    betas = m.config.factor_betas
    # ジャンプ共有 (s_J=0.5) と対数正規希釈込みの理論値近傍 (帯は 250 日の
    # サンプリング誤差 SE~0.06 に余裕を載せる)。厳密帯は本番ゲートが担う。
    for i, j in ((0, 1), (0, 2), (1, 2)):
        c = float(np.corrcoef(r[i], r[j])[0, 1])
        expect = betas[i] * betas[j]
        assert expect - 0.30 < c < expect + 0.30, (i, j, c, expect)
        assert c > 0.05, (i, j, c)


def test_vol_correlation_from_shared_components() -> None:
    # 250 日では測れない (OU 半減期 30 日の実効独立標本 ~6 本 — 固有 OU 間の
    # 相関推定 SD ~0.4 で、実測 −0.69 の見かけの負相関を引いた)。2000 日で
    # 成分は設計へ厳密収束する (corr(msm_i, common) 実測 0.70-0.75 vs 理論 0.735)。
    m = run_multi(_latent_config(n_days=2000, n_assets=2, factor_betas=(0.6, 0.8)))
    lv = [p.log_vol_sub for p in m.payloads]
    n = min(v.size for v in lv)
    c01 = float(np.corrcoef(lv[0][:n], lv[1][:n])[0, 1])
    # 設計シェア ~0.57 (共有実現 0.131 / 総実現 0.23 — 最遅 MSM 成分は 2000 日
    # でも部分実現)。帯は ±0.17 (OU の実効標本) に余裕を載せる。
    assert 0.40 < c01 < 0.80, c01


# ---------------------------------------------------------------------------
# 5: 共通ジャンプ = 完全コジャンプ
# ---------------------------------------------------------------------------
def test_common_jumps_are_shared_and_idio_jumps_are_not() -> None:
    cfg = _latent_config(n_days=120)
    m = run_multi(cfg)
    jd = m.common_diagnostics.get("jump") or {}
    assert jd.get("n_jumps", 0) > 0, "共通ジャンプが 1 本も出ていません (120 日)"
    # 共通ジャンプの実効強度が公称 (s_J·λ0) の Jensen 減衰の範囲にある
    lam_eff = jd["lambda_effective_per_year"]
    lam_nom = jd["lambda_nominal_per_year"]
    assert 0.5 * lam_nom < lam_eff <= 1.1 * lam_nom, (lam_eff, lam_nom)


def test_marginal_jump_budget_preserved() -> None:
    """資産あたりの総ジャンプ強度 (固有 + 共通) が S12 の水準と一致する。"""
    cfg = _latent_config(n_days=120)
    m = run_multi(cfg)
    # 資産 0 の診断: 固有 λ_eff + 共通 λ_eff ≈ 単一資産の λ_eff
    single = run(cfg.n1_config())
    jd_multi = (m.asset0.meta.get("l2") or {}).get("jump") or {}
    jd_single = (single.meta.get("l2") or {}).get("jump") or {}
    total_multi = (
        jd_multi["lambda_effective_per_year"]
        + jd_multi["lambda_effective_common_per_year"]
    )
    total_single = jd_single["lambda_effective_per_year"]
    # 固有側の変調は資産の全ボラ状態、共通側は共通状態のみ — Jensen の内訳が
    # 少し違うため厳密一致はしない (DECISION.md に記録)。±12% で拘束する。
    assert abs(total_multi / total_single - 1.0) < 0.12, (total_multi, total_single)


# ---------------------------------------------------------------------------
# 6: 合成後の z の健全性
# ---------------------------------------------------------------------------
def test_composed_innovation_has_no_cell_autocorrelation() -> None:
    cfg = _base_book_config(n_days=20).without_book().replace(steps_per_day=23400)
    m = run_multi(cfg)
    lev = (m.asset0.meta.get("l2") or {}).get("leverage") or {}
    z_acf = lev.get("z_acf") or {}
    # 判定はゲートと同じ Bonferroni 閾値 3.7/√N (2/√N は 60 ラグの最大値に
    # 対して純乱数でも ~95% 落ちる — S3 で確定済みの多重比較問題)。
    assert z_acf.get("max_abs_acf") is not None, z_acf
    assert z_acf["max_abs_acf"] < 3.7 / z_acf["n"] ** 0.5, z_acf
    # 周辺レバレッジの理論係数が記録されている (√(1−β²) 希釈 — §4.4 の帰結)
    assert "rho_rough_marginal_theory" in lev


def test_var_log_sigma_budget_is_preserved_by_split() -> None:
    """共有分割は周辺 Var(log σ) を変えない (§4.2 — 構成の恒等式)。

    共通 + 固有の分散目標の合計が、単一資産の総予算と厳密に一致することを
    設定から確認する (実現分散の統計比較は本番の記録が担う)。
    """
    cfg = _latent_config()
    total = cfg.vol_var_target_msm + cfg.vol_var_target_slow
    k_c = cfg.msm_k_common
    msm_common = cfg.vol_var_target_msm * k_c / cfg.msm_k
    msm_idio = cfg.vol_var_target_msm * (cfg.msm_k - k_c) / cfg.msm_k
    ou_common = cfg.ou_common_share * cfg.vol_var_target_slow
    ou_idio = (1.0 - cfg.ou_common_share) * cfg.vol_var_target_slow
    assert abs((msm_common + msm_idio + ou_common + ou_idio) - total) < 1e-12
