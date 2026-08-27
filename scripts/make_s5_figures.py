"""S5 (決定論的カオス chi_2) の図を作る。

README に載せる 4 枚:

1. ``s5_regimes.png`` — 同じ chi・違うシードで同じレジーム構造が現れる (S5 の目的)
2. ``s5_attractor.png`` — MG アトラクタと Rosenstein 発散曲線 (カオスの証拠)
3. ``s5_band_design.png`` — 30 日線と GPH 測定帯の位置関係 (2026-08-21 裁定の根拠)
4. ``s5_cross_seed.png`` — シード横断相関 = 分散シェアの実証 (§8)

ラベルは英語 (既存の図と同じ慣習 — matplotlib の既定フォントに日本語グリフが無い)。
数値の正は metrics.json (本番設定)。図は見え方を伝えるためのもの。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simchart import Config, run
from simchart.chaos import mackey_glass
from simchart.layers.l2_price import prepare_chaos_component
from simchart.validation.chaos import lyapunov_rosenstein
from simchart.validation.memory import gph_estimator
from simchart.validation.scaling import cross_seed_correlation

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"
BLUE, ORANGE, GREY, GREEN = "#1f4e79", "#d1701e", "#8a8a8a", "#2e7d32"

S4KW = dict(
    enable_msm=True, enable_slow_ou=True, enable_rough=True,
    enable_jump=True, enable_leverage=True,
    enable_seasonality=True, enable_overnight=True,
    jump_lambda_per_year=5.0, jump_eta_down=35.0, jump_eta_up=56.0,
    jump_qv_share_target=0.12, jump_p_up=0.42,
    leverage_rho_rough=-0.60, leverage_rho_slow=-0.35,
)


def _s5(seed: int, n_days: int, spd: int) -> Config:
    return Config(
        stage="S5", seed=seed, enable_chaos_vol=True,
        n_days=n_days, steps_per_day=spd, **S4KW,
    )


# ---------------------------------------------------------------------------
def fig_regimes(n_days: int, spd: int) -> None:
    """同一の chi・異なるシード → 同じレジーム構造。"""
    seeds = (42, 43, 44)
    daily_vols = {}
    chi_daily = None
    for seed in seeds:
        r = run(_s5(seed, n_days, spd))
        sub = r.meta["l2"]["vol_subsample"]
        lv = np.asarray(sub["log_vol"]) - np.asarray(sub["log_phi_sigma"])
        per_day = spd  # 1 分サブサンプル = spd 点/日 (spd=390 のとき)
        nd = lv.shape[0] // per_day
        daily_vols[seed] = np.exp(lv[: nd * per_day].reshape(nd, per_day).mean(axis=1))
        if chi_daily is None:
            chi = np.asarray(sub["chi_term"])
            chi_daily = chi[: nd * per_day].reshape(nd, per_day).mean(axis=1)
        del r
    show = slice(0, min(756, nd))  # 3 年分
    t = np.arange(nd)[show]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 2.0]})
    ax = axes[0]
    ax.plot(t, chi_daily[show], color="k", lw=1.4)
    ax.set_ylabel(r"$a\,\chi_2(t)$")
    ax.set_title(
        "the deterministic regime carrier is identical for every seed "
        "(same SHA256, zero RNG draws)",
        fontsize=10,
    )
    ax.grid(alpha=0.25)

    ax = axes[1]
    # 21 日移動平均で速い成分 (ラフ・高速 MSM) を落とし、共有される緩慢帯域
    # (chi の注入先) を見えるようにする。生の日次 σ だとシード固有成分 (分散比 4:1)
    # に埋もれて主張が読めない。
    win = 21
    kernel = np.ones(win) / win
    for seed, color in zip(seeds, (BLUE, ORANGE, GREEN)):
        smooth = np.convolve(np.log(daily_vols[seed]), kernel, mode="valid")
        ax.plot(np.arange(smooth.size)[show], np.exp(smooth)[show], color=color,
                lw=1.3, alpha=0.9, label=f"seed {seed}")
    ax.set_yscale("log")
    ax.set_xlabel("trading day")
    ax.set_ylabel(r"21-day mean $\sigma$ (annualised)")
    ax.set_title(
        "three independent seeds share the same regime skeleton (21-day smoothing "
        "reveals the common slow band) — that is what chaos buys, not statistics",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s5_regimes.png", dpi=140)
    plt.close(fig)


def fig_attractor() -> None:
    _, x = mackey_glass(length_units=3000.0, dt=0.1)
    tau_steps = 170
    ly = lyapunov_rosenstein(x, dt=0.1)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(x[tau_steps:], x[:-tau_steps], color=BLUE, lw=0.25, alpha=0.8)
    ax.set_xlabel(r"$x(t)$")
    ax.set_ylabel(r"$x(t-\tau)$")
    ax.set_title(
        "Mackey-Glass attractor (tau=17, correlation dimension 1.85)", fontsize=10
    )

    ax = axes[1]
    tcur = np.asarray(ly["curve_t_units"])
    ycur = np.asarray(ly["curve_mean_log_d"])
    ax.plot(tcur, ycur, color=BLUE, lw=1.5)
    lo, hi = ly["fit_range_units"]
    lam = ly["lyapunov_per_unit"]
    mask = (tcur >= lo) & (tcur <= hi)
    fit = np.polyfit(tcur[mask], ycur[mask], 1)
    ax.plot(tcur[mask], np.polyval(fit, tcur[mask]), color=ORANGE, lw=2.2, ls="--",
            label=rf"slope = $\lambda$ = {lam:+.4f}/unit (lit. ~0.006)")
    ax.set_xlabel("look-ahead time (chaos units)")
    ax.set_ylabel(r"$\langle \ln\, d(t) \rangle$ of neighbour pairs")
    ax.set_title(
        f"Rosenstein divergence: positive Lyapunov exponent (fit R² {ly['fit_r2']:.3f})",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s5_attractor.png", dpi=140)
    plt.close(fig)


def fig_band_design(n_days: int = 5000) -> None:
    """chi の線スペクトルと GPH 測定帯の位置関係 (裁定の根拠を 1 枚で)。"""
    cfg = _s5(42, n_days, 390)
    _, chi, a, _c, diag = prepare_chaos_component(cfg, float(n_days))
    from scipy import signal as sp

    fs = 1.0 / diag["grid_spacing_days"]
    freqs, psd = sp.welch(chi, fs=fs, nperseg=1 << 14)
    keep = (freqs > 1.0 / 600) & (freqs < 0.25)

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.semilogy(1.0 / freqs[keep], psd[keep] / psd[keep].max(), color=BLUE, lw=1.2)
    ax.set_xscale("log")
    ax.invert_xaxis()

    m050 = int(round(n_days**0.50))
    m065 = int(round(n_days**0.65))
    edge050, edge065 = n_days / m050, n_days / m065
    ax.axvspan(600, edge050, color=GREEN, alpha=0.15,
               label=f"GPH band at bw 0.50 (periods ≥ {edge050:.0f}d) — judgement band")
    ax.axvspan(edge050, edge065, color=ORANGE, alpha=0.15,
               label=f"extra band covered at bw 0.65 (down to {edge065:.0f}d) — recorded only")
    ax.axvline(30.0, color="crimson", lw=1.5, ls="--", label="designed peak: 30d")
    ax.axvline(30.0 * 102.4 / 49.65, color="crimson", lw=1.0, ls=":",
               label="subharmonic: 62d (still outside the judgement band)")
    ax.set_xlabel("period (market days, log scale, long periods on the left)")
    ax.set_ylabel("chi spectrum (normalised)")
    ax.set_title(
        "why the invariance gate reads bandwidth 0.50: the mandated 20–40d peak always sits inside\n"
        "the 0.65 band (Δd −0.08…−0.11 by construction), while the 0.50 band stays clean (Δd +0.0006)",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "s5_band_design.png", dpi=140)
    plt.close(fig)


def fig_cross_seed(n_days: int, spd: int) -> None:
    seeds = tuple(range(42, 50))
    paths = []
    for seed in seeds:
        r = run(_s5(seed, n_days, spd))
        sub = r.meta["l2"]["vol_subsample"]
        paths.append(
            (np.asarray(sub["log_vol"]) - np.asarray(sub["log_phi_sigma"]))[::5]
        )
        del r
    csc = cross_seed_correlation(paths)
    mat = np.corrcoef(np.stack([p[: min(len(q) for q in paths)] for p in paths]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set_yticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set_title("corr(log σ_i, log σ_j) across seeds (φ removed)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    iu = np.triu_indices(len(seeds), k=1)
    ax.hist(mat[iu], bins=12, color=BLUE, alpha=0.8)
    ax.axvline(0.20, color="crimson", lw=1.6, ls="--",
               label="theory: Var(χ)/Var(logσ) = 0.05/0.25 = 0.20")
    ax.axvline(csc["mean"], color=ORANGE, lw=1.6,
               label=f"mean of pairs = {csc['mean']:.3f}")
    ax.set_xlabel("pairwise correlation")
    ax.set_ylabel("pairs")
    ax.set_title(
        "off-diagonal correlation estimates the χ variance share\n"
        "without touching internal state (the core S5 gate)",
        fontsize=10,
    )
    ax.legend(fontsize=8.5)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(IMAGES / "s5_cross_seed.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="S5 の図を作る")
    parser.add_argument("--n-days", type=int, default=2000)
    parser.add_argument("--steps-per-day", type=int, default=390)
    args = parser.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)

    print("図 1/4: レジーム構造", flush=True)
    fig_regimes(args.n_days, args.steps_per_day)
    print("図 2/4: アトラクタと Lyapunov", flush=True)
    fig_attractor()
    print("図 3/4: 測定帯の設計", flush=True)
    fig_band_design()
    print("図 4/4: シード横断相関", flush=True)
    fig_cross_seed(args.n_days, args.steps_per_day)

    for name in ("s5_regimes", "s5_attractor", "s5_band_design", "s5_cross_seed"):
        p = IMAGES / f"{name}.png"
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
