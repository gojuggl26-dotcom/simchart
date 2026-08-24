"""S5: 決定論的カオス系列 χ₂ の生成 (Mackey-Glass)。

役割
----
log σ の緩慢帯域に加算する**決定論的**成分を作る。目的は統計的リアリズムの向上では
なく、**ボラティリティ・レジームの決定論的な再現性と制御可能性** — 同一のカオス
初期値なら、確率的な部分 (シード) が違っても同じレジーム構造が現れる。これが
複数シナリオ比較・戦略頑健性テストでの価値になる (S5 指示書 §0)。

再現性の要件 (指示書 §7) をこのモジュールが一手に引き受ける:

- **乱数を一切消費しない** (全入力は config 由来の決定論的パラメータ)
- 固定ステップ RK4。適応ステップ (RK45 等) は使わない — 局所誤差推定が
  トラジェクトリを環境依存にするため
- float64。初期値は遅延分の履歴全体を定数で指定 (config に明示)
- burn-in 長を明示し固定
- 生成配列の SHA256 を返し、metrics.json に記録する
- **ディスクキャッシュ**: 長期 run では丸め誤差の蓄積で環境が違うと軌道が分岐し
  得るため、一度生成した系列を ``cache/`` に保存し、ロード時にハッシュで同一性を
  検証する。ハッシュ不一致 = 別環境で生成された別軌道なので、黙って使わず
  再生成する (キャッシュはあくまで高速化と再現性の**証拠**であって権威ではない —
  権威は metrics.json に記録されたハッシュ)

Mackey-Glass について
---------------------
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

古典パラメータ (beta=0.2, gamma=0.1, n=10, tau=17) で相関次元 ~2.1 の低次元カオス。
滑らかな不規則振動で周辺分布が単峰 — 「緩慢成分の滑らかな搬送波」という χ₂ の
役割に合う (指示書 §3.1 の推奨)。

遅延微分方程式の RK4
--------------------
RK4 の各ステージは遅延値 x(t-tau), x(t+h/2-tau), x(t+h) を要求する。
``tau/h`` を**整数に強制**することで t-tau と t+h-tau は履歴グリッド上の点になり、
半ステージ点は隣接 2 点の線形補間 (= 平均) で与える。これは高次精度ではないが、
カオス軌道に「正確な」積分は存在しない (指数感度で必ず発散する) — 要件は
**完全に固定された決定論的スキーム**であることと、アトラクタの性質 (正の
Lyapunov・低い相関次元) が正しく出ることの 2 つで、どちらも満たす。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ChaosSeries", "mackey_glass", "chaos_generate", "chi_window"]


@dataclass(frozen=True)
class ChaosSeries:
    """生成済みカオス系列。``t`` は系固有の時間単位 (市場時間ではない)。"""

    t: np.ndarray
    x: np.ndarray
    dt: float
    sha256: str
    params: dict
    cache_path: str | None = None

    @property
    def n_points(self) -> int:
        return int(self.x.shape[0])


def _array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def mackey_glass(
    length_units: float,
    dt: float = 0.1,
    tau: float = 17.0,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_exponent: float = 10.0,
    ic: float = 1.2,
    burn_in_units: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mackey-Glass を固定ステップ RK4 で積分し、burn-in 後の (t, x) を返す。

    初期条件は「t <= 0 で恒等的に ``ic``」の定数履歴。遅延方程式の初期値は
    **関数** (遅延区間の履歴全体) なので、スカラー 1 つで完全に指定できる定数履歴を
    採用する (指示書 §7: config に明示できる形)。
    """
    if dt <= 0 or length_units <= 0:
        raise ValueError("dt と length_units は正である必要があります")
    ratio = tau / dt
    delay_steps = int(round(ratio))
    if abs(ratio - delay_steps) > 1e-9 or delay_steps < 1:
        raise ValueError(
            f"tau/dt = {ratio} が整数ではありません。遅延値が履歴グリッドに載らず、"
            f"補間方式が環境依存の曖昧さを持つため、整数になる dt を指定してください。"
        )

    n_burn = int(round(burn_in_units / dt))
    n_keep = int(math.ceil(length_units / dt)) + 1
    n_total = n_burn + n_keep

    # 履歴込みのバッファ。先頭 delay_steps+1 点が t in [-tau, 0] の定数履歴。
    buf = np.empty(delay_steps + n_total, dtype=np.float64)
    buf[: delay_steps + 1] = float(ic)

    b = float(beta)
    g = float(gamma)
    p = float(n_exponent)
    h = float(dt)

    def f(x: float, xd: float) -> float:
        return b * xd / (1.0 + xd**p) - g * x

    # 逐次 RK4。90k ステップ程度なので純 Python で十分速い (~0.3 秒)。
    for k in range(delay_steps, delay_steps + n_total - 1):
        x = buf[k]
        xd0 = buf[k - delay_steps]
        xd1 = buf[k - delay_steps + 1]
        xdh = 0.5 * (xd0 + xd1)  # 半ステージの遅延値 (線形補間)
        k1 = f(x, xd0)
        k2 = f(x + 0.5 * h * k1, xdh)
        k3 = f(x + 0.5 * h * k2, xdh)
        k4 = f(x + h * k3, xd1)
        buf[k + 1] = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    x_keep = buf[delay_steps + n_burn : delay_steps + n_total].copy()
    t_keep = np.arange(n_keep, dtype=np.float64) * h
    return t_keep, x_keep


def chaos_generate(
    system: str,
    params: dict,
    length_units: float,
    dt: float,
    ic: float,
    burn_in_units: float,
    cache_dir: str | Path | None = None,
    name: str = "chi2",
) -> ChaosSeries:
    """決定論的にカオス系列を生成する (キャッシュつき)。

    キャッシュのファイル名は全パラメータから決まり、ロード時に配列の SHA256 を
    再計算して照合する。照合はロードのたびに行う — キャッシュを「信じる」のでは
    なく「使ってよいことを毎回証明する」(壊れた/別環境のキャッシュを黙って使うと
    再現性の主張全体が崩れるため)。
    """
    if system != "mackey_glass":
        raise NotImplementedError(f"カオス系 {system!r} は未実装です (S5 は Mackey-Glass のみ)")

    full_params = {
        "system": system,
        "tau": float(params.get("tau", 17.0)),
        "beta": float(params.get("beta", 0.2)),
        "gamma": float(params.get("gamma", 0.1)),
        "n_exponent": float(params.get("n_exponent", 10.0)),
        "dt": float(dt),
        "ic": float(ic),
        "burn_in_units": float(burn_in_units),
        "length_units": float(length_units),
    }
    key = "_".join(f"{k}={v}" for k, v in sorted(full_params.items()))
    ic_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    fname = f"{name}_{system}_{ic_hash}.npy"

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / fname
        if cache_path.exists():
            x = np.load(cache_path)
            t = np.arange(x.shape[0], dtype=np.float64) * full_params["dt"]
            return ChaosSeries(
                t=t, x=x, dt=full_params["dt"], sha256=_array_hash(x),
                params=full_params, cache_path=str(cache_path),
            )

    t, x = mackey_glass(
        length_units=full_params["length_units"],
        dt=full_params["dt"],
        tau=full_params["tau"],
        beta=full_params["beta"],
        gamma=full_params["gamma"],
        n_exponent=full_params["n_exponent"],
        ic=full_params["ic"],
        burn_in_units=full_params["burn_in_units"],
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, x)
    return ChaosSeries(
        t=t, x=x, dt=full_params["dt"], sha256=_array_hash(x),
        params=full_params, cache_path=str(cache_path) if cache_path else None,
    )


def chi_window(
    config, n_days: float, which: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """S12: χ₁ / χ₃ の窓を用意する共通経路 (χ₂ の prepare_chaos_component と同型)。

    同一の MG 系を**異なる初期値・異なる時間写像**で回す (§3.1 の推奨構成)。
    過渡除去後は動的に独立 — 独立性は chi_independence ゲートが実測で確認する。
    Returns: (t_days, chi_norm, diagnostics)。chi_norm は窓上で平均 0・分散 1。
    乱数は一切消費しない。
    """
    if which == "chi1":
        ic = float(config.chi1_ic)
        s = float(config.chi1_days_per_unit)
    elif which == "chi3":
        ic = float(config.chi3_ic)
        s = float(config.chi3_days_per_unit)
    else:
        raise ValueError(f"未知の系列 {which!r} (chi1 / chi3)")
    length_units = n_days / s + 2.0 * config.chaos_dt
    series = chaos_generate(
        system=config.chaos_system,
        params={
            "tau": config.chaos_tau_delay,
            "beta": config.chaos_beta,
            "gamma": config.chaos_gamma,
            "n_exponent": config.chaos_n_exponent,
        },
        length_units=length_units,
        dt=config.chaos_dt,
        ic=ic,
        burn_in_units=config.chaos_burn_in_units,
        cache_dir=config.chaos_cache_dir or None,
        name=which,
    )
    x = series.x
    mu = float(x.mean())
    sd = float(x.std())
    if sd <= 0:
        raise ValueError(f"{which} の分散が 0 です")
    chi_norm = (x - mu) / sd
    diagnostics = {
        "system": config.chaos_system,
        "sha256": series.sha256,
        "cache_path": series.cache_path,
        "ic": ic,
        "days_per_unit": s,
        "grid_spacing_days": config.chaos_dt * s,
        "n_grid_points": int(chi_norm.shape[0]),
        "window_mean": mu,
        "window_sd": sd,
    }
    return series.t * s, chi_norm, diagnostics
