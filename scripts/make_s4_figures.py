"""S4 (日内季節性 + オーバーナイト) の図を作る。

README に載せる 4 枚:

1. ``s4_phi_profiles.png``   — φ_σ と φ_λ の形と、正規化の規約が違うこと
2. ``s4_deseasonalize.png``  — 日内ボラ・プロファイルが除去で平らになる
3. ``s4_spectrum.png``       — |r| のスペクトルに立つ日内高調波と、除去後の消失
4. ``s4_overnight.png``      — ギャップの分布・引けボラとの連動・帰無対照

★図を作るためだけに本番設定 (5000 日 x 23400) を回さない。季節性は日内の構造なので
1000 日あれば形は十分に決まり、メモリも 1/5 で済む。数値そのものは metrics.json
(本番設定) を正とし、この図は「見え方」を伝えるためのものと位置づける。
"""

from __future__ import annotations

import argparse
import math
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


def _build(n_days: int, steps_per_day: int, seed: int):
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
def fig_phi_profiles(cfg: Config, calendar) -> None:
    u = np.linspace(0.0, 1.0, 2001)
    ps = np.asarray(calendar.phi_sigma_of_u(u))
    pl = np.asarray(calendar.phi_lambda_of_u(u))
    hours = 6.5 * u

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    ax.plot(hours, ps, color=BLUE, lw=2.0, label=r"$\varphi_\sigma(u)$")
    ax.plot(hours, ps**2, color=ORANGE, lw=1.4, ls="--", label=r"$\varphi_\sigma^2(u)$")
    ax.axhline(1.0, color=GREY, lw=0.9, ls=":")
    ax.set_title(
        "ボラの季節係数: 二乗の平均を 1 に正規化\n"
        rf"$(1/T)\int\varphi_\sigma^2du=1$   起伏比 $\varphi^2$: "
        f"{(ps**2).max() / (ps**2).min():.2f}",
        fontsize=10,
    )
    ax.annotate(
        f"寄付 {ps[0]:.2f}", (0.0, ps[0]), textcoords="offset points", xytext=(6, 4), fontsize=8
    )
    ax.annotate(
        f"引け {ps[-1]:.2f}", (hours[-1], ps[-1]), textcoords="offset points",
        xytext=(-42, 4), fontsize=8,
    )

    ax = axes[1]
    ax.plot(hours, pl, color=BLUE, lw=2.0, label=r"$\varphi_\lambda(u)$")
    ax.axhline(1.0, color=GREY, lw=0.9, ls=":")
    ax.set_title(
        "出来高の季節係数: **一乗**の平均を 1 に正規化\n"
        rf"$(1/T)\int\varphi_\lambda du=1$   起伏比: {pl.max() / pl.min():.2f}",
        fontsize=10,
    )
    ax.annotate(
        f"引け {pl[-1]:.2f}", (hours[-1], pl[-1]), textcoords="offset points",
        xytext=(-46, -12), fontsize=8,
    )

    for ax in axes:
        ax.set_xlabel("セッション経過 (時間)")
        ax.set_xlim(0, 6.5)
        ax.legend(fontsize=9, loc="upper center")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "正規化の規約が σ と λ で違う — 分散は加算されるので φ_σ は二乗、強度は一乗",
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

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    ax.plot(hours, raw / raw.mean(), color=GREY, lw=1.0, alpha=0.8, label="実測 (生)")
    ax.plot(hours, phi_true / phi_true.mean(), color=BLUE, lw=2.0, label="真の φ_σ")
    ax.plot(hours, phi_hat / phi_hat.mean(), color=ORANGE, lw=1.4, ls="--", label="推定 φ̂")
    ax.set_title(
        "推定は φ を知らずにリターンだけから作る\n"
        f"相関 {np.corrcoef(phi_hat, phi_true)[0, 1]:.5f} / "
        f"最大相対誤差 {np.abs(phi_hat / phi_true - 1).max() * 100:.1f}%",
        fontsize=10,
    )
    ax.set_ylabel("正規化した日内水準")

    ax = axes[1]
    ax.plot(hours, raw / raw.mean(), color=GREY, lw=1.4,
            label=f"除去前 (SE 比 {f_raw['excess_over_se']:.2f})")
    ax.plot(hours, p_true / p_true.mean(), color=BLUE, lw=1.6,
            label=f"真の φ で除去 (SE 比 {f_true['excess_over_se']:.2f})")
    ax.plot(hours, p_est / p_est.mean(), color=ORANGE, lw=1.2, ls="--",
            label=f"推定 φ̂ で除去 (SE 比 {f_est['excess_over_se']:.2f})")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_title(
        "除去後は標本誤差の水準まで平坦になる\n"
        "(SE 比 = プロファイルの sd(log) ÷ 中央値の標本誤差)", fontsize=10
    )

    for ax in axes:
        ax.set_xlabel("セッション経過 (時間)")
        ax.set_xlim(0, 6.5)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)
    fig.suptitle("日内ボラ・プロファイルの推定と除去 (1 分バー)", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGES / "s4_deseasonalize.png", dpi=140)
    plt.close(fig)


def fig_spectrum(r_2d: np.ndarray, phi_true: np.ndarray, r3_2d: np.ndarray) -> None:
    """|r| のスペクトル。日内周期の高調波が離散的に立つことを見せる。"""
    n_days, n_bars = r_2d.shape

    def spec(arr: np.ndarray) -> np.ndarray:
        a = np.abs(arr)
        a = a - a.mean(axis=1, keepdims=True)
        x = a.ravel()
        return np.abs(np.fft.rfft(x)) ** 2 / x.size

    p_raw, p_dsn, p_s3 = spec(r_2d), spec(sz.deseasonalize(r_2d, phi_true)), spec(r3_2d)
    harm = np.array([k * n_days for k in range(1, 7)])
    # 表示は基本周波数の周り。横軸は「1 日あたり何周期か」。
    hi = harm[-1] + 3 * n_days
    idx = np.arange(1, hi)
    cycles = idx / n_days

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.semilogy(cycles, p_raw[idx], color=GREY, lw=0.6, alpha=0.75, label="S4 生")
    ax.semilogy(cycles, p_dsn[idx], color=BLUE, lw=0.6, alpha=0.9, label="S4 φ 除去後")
    ax.semilogy(cycles, p_s3[idx], color=ORANGE, lw=0.6, alpha=0.6, label="S3 (季節性なし)")
    for k, j in enumerate(harm, start=1):
        ax.axvline(k, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.scatter(np.arange(1, 7), p_raw[harm], color="crimson", zorder=5, s=28,
               label="日内周期の高調波")

    raw_stat = sz.spectral_periodicity_test(r_2d)
    dsn_stat = sz.spectral_periodicity_test(sz.deseasonalize(r_2d, phi_true))
    ax.set_xlabel("周波数 (1 日あたりの周期数)")
    ax.set_ylabel(r"$|r|$ のピリオドグラム")
    ax.set_title(
        "季節性は離散的な高調波にしか力を持たない — 連続スペクトルの長期記憶と分離できる\n"
        f"高調波比 (近傍中央値で基準化): 生 {raw_stat['mean_ratio']:.1f} → "
        f"除去後 {dsn_stat['mean_ratio']:.2f}   (帰無での期待値 1)",
        fontsize=10,
    )
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(IMAGES / "s4_spectrum.png", dpi=140)
    plt.close(fig)


def fig_overnight(gaps: np.ndarray, r_daily: np.ndarray, sigma_close: np.ndarray) -> None:
    from scipy import stats as st

    stats = sz.overnight_stats(gaps, r_daily, sigma_close)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    ax = axes[0]
    lim = float(np.percentile(np.abs(np.concatenate([gaps, r_daily])), 99.5))
    bins = np.linspace(-lim, lim, 61)
    ax.hist(r_daily, bins=bins, density=True, color=GREY, alpha=0.55,
            label=f"日中日次 (尖度 {stats['kurtosis_intraday_daily']:.1f})")
    ax.hist(gaps, bins=bins, density=True, histtype="step", color=BLUE, lw=1.8,
            label=f"ON ギャップ (尖度 {stats['kurtosis_gap']:.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("対数リターン")
    ax.set_title(
        f"ON は情報が溜まって一度に出る → 裾が厚い\n"
        f"分散シェア {stats['variance_share'] * 100:.1f}% (目標 20%)", fontsize=10
    )

    ax = axes[1]
    ax.scatter(sigma_close, np.abs(gaps), s=5, alpha=0.35, color=BLUE)
    sl, ic = np.polyfit(sigma_close, np.abs(gaps), 1)
    xs = np.linspace(sigma_close.min(), sigma_close.max(), 50)
    ax.plot(xs, sl * xs + ic, color=ORANGE, lw=1.8)
    ax.set_xlabel(r"引け時点の $\sigma$ (年率)")
    ax.set_ylabel(r"$|$ギャップ$|$")
    ax.set_title(
        f"引けのボラ水準と連動する\n"
        f"corr = {stats['corr_abs_gap_sigma_close']:.3f} "
        f"(SE {stats['corr_abs_gap_sigma_close_se']:.3f})", fontsize=10
    )

    ax = axes[2]
    nxt = r_daily[1:]
    ax.scatter(gaps, nxt, s=5, alpha=0.35, color=GREY)
    c = stats["corr_gap_next_intraday"]
    se = stats["corr_gap_next_intraday_se"]
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("ギャップ (日 d → d+1)")
    ax.set_ylabel("翌日の日中リターン")
    ax.set_title(
        f"帰無対照: ギャップは翌日を予測しない\n"
        f"corr = {c:+.4f} (SE {se:.4f}, {abs(c) / se:.1f} SE)", fontsize=10
    )

    for ax in axes:
        ax.legend(fontsize=8.5) if ax.get_legend_handles_labels()[0] else None
        ax.grid(alpha=0.25)
    fig.suptitle("オーバーナイト・ギャップ (物理時間比例ではなく別レジームとして生成)", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGES / "s4_overnight.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=1000)
    parser.add_argument("--steps-per-day", type=int, default=23400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
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

    print("図 1/4: φ の形", flush=True)
    fig_phi_profiles(cfg4, calendar)
    print("図 2/4: 脱季節化", flush=True)
    fig_deseasonalize(r_2d, phi_true, phi_hat)
    print("図 4/4 の材料として S3 を生成中...", flush=True)
    r3_2d = run(cfg3).observation.to_bars(bar).returns_2d()
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
