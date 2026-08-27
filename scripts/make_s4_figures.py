"""S4 (日内季節性 + オーバーナイト) の図を作る。

README に載せる 4 枚:

1. ``s4_phi_profiles.png`` — φ_σ と φ_λ の形と、正規化の規約が違うこと
2. ``s4_deseasonalize.png`` — 日内ボラ・プロファイルが除去で平らになる
3. ``s4_spectrum.png`` — |r| のスペクトルに立つ日内高調波と、除去後の消失
4. ``s4_overnight.png`` — ギャップの分布・引けボラとの連動・帰無対照

図のラベルは英語で書く (既存の図スクリプトと同じ慣習)。matplotlib の既定フォントに
日本語グリフが無く、環境によって豆腐になるため。説明は README 本文が持つ。

図を作るためだけに本番設定 (5000 日 x 23400) を回さない。季節性は日内の構造なので
1000 日あれば形は十分に決まり、メモリも 1/5 で済む。数値そのものは metrics.json
(本番設定) を正とし、この図は見え方を伝えるためのものと位置づける。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simchart import Config, run
from simchart.layers.l0_calendar import build_calendar
from simchart.rng import RNGRegistry
from simchart.validation import seasonality as sz

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"
BLUE, ORANGE, GREY = "#1f4e79", "#d1701e", "#8a8a8a"


def _build(n_days: int, steps_per_day: int, seed: int) -> tuple[Config, Config]:
    base = dict(
        n_days=n_days,
        steps_per_day=steps_per_day,
        seed=seed,
        enable_msm=True,
        enable_slow_ou=True,
        enable_rough=True,
        enable_jump=True,
        enable_leverage=True,
        jump_p_up=0.42,
        jump_eta_up=56.0,
        jump_eta_down=35.0,
        jump_lambda_per_year=5.0,
        jump_qv_share_target=0.12,
        leverage_rho_rough=-0.60,
        leverage_rho_slow=-0.35,
    )
    cfg4 = Config(stage="S4", enable_seasonality=True, enable_overnight=True, **base)
    cfg3 = Config(stage="S3", **base)
    return cfg4, cfg3


# ---------------------------------------------------------------------------
def fig_phi_profiles(calendar) -> None:
    u = np.linspace(0.0, 1.0, 2001)
    ps = np.asarray(calendar.phi_sigma_of_u(u))
    pl = np.asarray(calendar.phi_lambda_of_u(u))
    hours = 6.5 * u

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.plot(hours, ps, color=BLUE, lw=2.0, label=r"$\varphi_\sigma(u)$")
    ax.plot(hours, ps**2, color=ORANGE, lw=1.4, ls="--", label=r"$\varphi_\sigma^2(u)$")
    ax.axhline(1.0, color=GREY, lw=0.9, ls=":")
    ax.set_title(
        "volatility: the SQUARE is normalised\n"
        rf"$(1/T)\!\int\!\varphi_\sigma^2 du = 1$, peak/trough of $\varphi^2$ = "
        f"{(ps**2).max() / (ps**2).min():.2f}",
        fontsize=10,
    )
    ax.set_ylabel(r"$\varphi$")
    for x, y, txt, dx in ((0.0, ps[0], f"open {ps[0]:.2f}", 6), (6.5, ps[-1], f"close {ps[-1]:.2f}", -58)):
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, 5), fontsize=8)

    ax = axes[1]
    ax.plot(hours, pl, color=BLUE, lw=2.0, label=r"$\varphi_\lambda(u)$")
    ax.axhline(1.0, color=GREY, lw=0.9, ls=":")
    ax.set_title(
        "activity: the LEVEL is normalised\n"
        rf"$(1/T)\!\int\!\varphi_\lambda du = 1$, peak/trough = {pl.max() / pl.min():.2f}",
        fontsize=10,
    )
    for x, y, txt, dx in ((0.0, pl[0], f"open {pl[0]:.2f}", 6), (6.5, pl[-1], f"close {pl[-1]:.2f}", -58)):
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, -13), fontsize=8)

    for ax in axes:
        ax.set_xlabel("hours into the session")
        ax.set_xlim(0, 6.5)
        ax.legend(fontsize=9, loc="upper center")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Two different normalisations: variance adds (square), intensity adds (level)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(IMAGES / "s4_phi_profiles.png", dpi=140)
    plt.close(fig)


def fig_deseasonalize(r_2d: np.ndarray, phi_true: np.ndarray, phi_hat: np.ndarray) -> None:
    n_bars = r_2d.shape[1]
    hours = 6.5 * sz.bin_centers(n_bars)
    raw = np.asarray(sz.intraday_profile(r_2d)["value"])
    d_true = sz.deseasonalize(r_2d, phi_true)
    d_est = sz.deseasonalize(r_2d, phi_hat)
    p_true = np.asarray(sz.intraday_profile(d_true)["value"])
    p_est = np.asarray(sz.intraday_profile(d_est)["value"])

    f_raw = sz.profile_flatness(r_2d)
    f_true = sz.profile_flatness(d_true)
    f_est = sz.profile_flatness(d_est)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.plot(hours, raw / raw.mean(), color=GREY, lw=1.0, alpha=0.85, label="measured (raw)")
    ax.plot(hours, phi_true / phi_true.mean(), color=BLUE, lw=2.0, label=r"true $\varphi_\sigma$")
    ax.plot(hours, phi_hat / phi_hat.mean(), color=ORANGE, lw=1.4, ls="--",
            label=r"estimated $\hat\varphi$")
    ax.set_title(
        "the estimator never sees the true profile\n"
        f"corr = {np.corrcoef(phi_hat, phi_true)[0, 1]:.5f},  "
        f"max relative error = {np.abs(phi_hat / phi_true - 1).max() * 100:.1f}%",
        fontsize=10,
    )
    ax.set_ylabel("normalised intraday level")

    ax = axes[1]
    ax.plot(hours, raw / raw.mean(), color=GREY, lw=1.4,
            label=f"before ({f_raw['excess_over_se']:.2f} x SE)")
    ax.plot(hours, p_true / p_true.mean(), color=BLUE, lw=1.6,
            label=f"true $\\varphi$ removed ({f_true['excess_over_se']:.2f} x SE)")
    ax.plot(hours, p_est / p_est.mean(), color=ORANGE, lw=1.2, ls="--",
            label=f"est. $\\hat\\varphi$ removed ({f_est['excess_over_se']:.2f} x SE)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_title(
        "after removal the profile sits at the sampling-noise floor\n"
        "(x SE = sd of log profile / split-half sampling error)",
        fontsize=10,
    )

    for ax in axes:
        ax.set_xlabel("hours into the session")
        ax.set_xlim(0, 6.5)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)
    fig.suptitle("Estimating and removing the intraday volatility profile (1-minute bars)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGES / "s4_deseasonalize.png", dpi=140)
    plt.close(fig)


def fig_spectrum(r_2d: np.ndarray, phi_true: np.ndarray, r3_2d: np.ndarray) -> None:
    """|r| のスペクトル。日内周期の高調波が離散的に立つことを見せる。"""
    n_days = r_2d.shape[0]

    def spec(arr: np.ndarray) -> np.ndarray:
        a = np.abs(arr)
        a = a - a.mean(axis=1, keepdims=True)
        x = a.ravel()
        return np.abs(np.fft.rfft(x)) ** 2 / x.size

    d_true = sz.deseasonalize(r_2d, phi_true)
    p_raw, p_dsn, p_s3 = spec(r_2d), spec(d_true), spec(r3_2d)
    n_show = 6
    harm = np.array([k * n_days for k in range(1, n_show + 1)])
    idx = np.arange(1, harm[-1] + 2 * n_days)
    cycles = idx / n_days

    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.semilogy(cycles, p_raw[idx], color=GREY, lw=0.5, alpha=0.75, label="S4 raw")
    ax.semilogy(cycles, p_dsn[idx], color=BLUE, lw=0.5, alpha=0.9,
                label=r"S4, true $\varphi$ removed")
    ax.semilogy(cycles, p_s3[idx], color=ORANGE, lw=0.5, alpha=0.55,
                label="S3 (no seasonality)")
    for k in range(1, n_show + 1):
        ax.axvline(k, color="k", lw=0.6, ls=":", alpha=0.45)
    ax.scatter(np.arange(1, n_show + 1), p_raw[harm], color="crimson", zorder=5, s=30,
               label="intraday harmonics")

    raw_stat = sz.spectral_periodicity_test(r_2d)
    dsn_stat = sz.spectral_periodicity_test(d_true)
    ax.set_xlabel("frequency (cycles per trading day)")
    ax.set_ylabel(r"periodogram of $|r|$")
    ax.set_title(
        "Seasonality puts power on discrete harmonics only — long memory is a continuous "
        "spectrum, so the two separate exactly here\n"
        f"harmonic ratio vs local median: raw {raw_stat['mean_ratio']:.1f}  →  "
        f"after removal {dsn_stat['mean_ratio']:.2f}   (null expectation = 1)",
        fontsize=10,
    )
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(IMAGES / "s4_spectrum.png", dpi=140)
    plt.close(fig)


def fig_overnight(gaps: np.ndarray, r_daily: np.ndarray, sigma_close: np.ndarray) -> None:
    stats = sz.overnight_stats(gaps, r_daily, sigma_close)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    ax = axes[0]
    # それぞれ自分の SD で標準化してから比べる。ON は総分散の 2 割しか持たないので
    # 生のままだとギャップのほうが分布が狭いという逆の印象になり、伝えたい
    # 裾が厚いが読めない。ここで見せたいのは水準ではなく形である。
    zg = gaps / gaps.std()
    zi = r_daily / r_daily.std()
    lim = float(np.percentile(np.abs(np.concatenate([zg, zi])), 99.8))
    bins = np.linspace(-lim, lim, 61)
    ax.hist(zi, bins=bins, density=True, color=GREY, alpha=0.55,
            label=f"intraday daily (kurt {stats['kurtosis_intraday_daily']:.1f})")
    ax.hist(zg, bins=bins, density=True, histtype="step", color=BLUE, lw=1.8,
            label=f"overnight gap (kurt {stats['kurtosis_gap']:.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("log return / own standard deviation")
    ax.set_ylabel("density")
    ax.set_title(
        "same scale, different shape: the gap has heavier tails\n"
        f"(gap carries {stats['variance_share'] * 100:.1f}% of the variance, target 20%)",
        fontsize=10,
    )
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.scatter(sigma_close, np.abs(gaps), s=5, alpha=0.35, color=BLUE)
    sl, ic = np.polyfit(sigma_close, np.abs(gaps), 1)
    xs = np.linspace(sigma_close.min(), sigma_close.max(), 50)
    ax.plot(xs, sl * xs + ic, color=ORANGE, lw=1.8)
    ax.set_xlabel(r"$\sigma$ at the close (annualised)")
    ax.set_ylabel(r"$|$gap$|$")
    ax.set_title(
        "the gap scales with closing volatility\n"
        f"corr = {stats['corr_abs_gap_sigma_close']:.3f} "
        f"(SE {stats['corr_abs_gap_sigma_close_se']:.3f})",
        fontsize=10,
    )

    ax = axes[2]
    c = stats["corr_gap_next_intraday"]
    se = stats["corr_gap_next_intraday_se"]
    ax.scatter(gaps, r_daily[1:], s=5, alpha=0.35, color=GREY)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("gap (day d to d+1)")
    ax.set_ylabel("next day's intraday return")
    ax.set_title(
        "null control: the gap must not predict the next day\n"
        f"corr = {c:+.4f} (SE {se:.4f}, {abs(c) / se:.1f} SE)",
        fontsize=10,
    )

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Overnight gaps are a separate regime, not diffusion scaled by 17.5/6.5 hours",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(IMAGES / "s4_overnight.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="S4 の図を作る")
    parser.add_argument("--n-days", type=int, default=1000)
    parser.add_argument("--steps-per-day", type=int, default=23400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)
    cfg4, cfg3 = _build(args.n_days, args.steps_per_day, args.seed)

    print(f"S4 を生成中 ({args.n_days} 日 x {args.steps_per_day} ステップ)...", flush=True)
    r4 = run(cfg4)
    calendar = build_calendar(cfg4, RNGRegistry(cfg4.seed))

    obs = r4.observation
    bar = cfg4.validation.primary_bar_sec
    r_2d = obs.to_bars(bar).returns_2d()
    phi_true = np.asarray(
        sz.true_phi_bars(calendar, r_2d.shape[1], steps_per_day=args.steps_per_day)["value"]
    )
    phi_hat = np.asarray(sz.estimate_phi(r_2d)["value"])
    r_daily = obs.to_bars(obs.session_seconds).returns()
    close_idx = np.arange(1, cfg4.n_days) * args.steps_per_day - 1
    sigma_close = np.exp(r4.price.log_vol[close_idx])
    gaps = r4.price.overnight_gaps
    del r4, obs

    print("図 1/4: phi の形", flush=True)
    fig_phi_profiles(calendar)
    print("図 2/4: 脱季節化", flush=True)
    fig_deseasonalize(r_2d, phi_true, phi_hat)
    print("図 4/4 の材料として S3 を生成中...", flush=True)
    r3 = run(cfg3)
    r3_2d = r3.observation.to_bars(bar).returns_2d()
    del r3
    print("図 3/4: スペクトル", flush=True)
    fig_spectrum(r_2d, phi_true, r3_2d)
    print("図 4/4: オーバーナイト", flush=True)
    fig_overnight(gaps, r_daily, sigma_close)

    print(f"\n{IMAGES} に 4 枚を書き出しました:")
    for name in ("s4_phi_profiles", "s4_deseasonalize", "s4_spectrum", "s4_overnight"):
        p = IMAGES / f"{name}.png"
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
