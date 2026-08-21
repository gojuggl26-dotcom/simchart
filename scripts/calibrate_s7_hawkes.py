# -*- coding: utf-8 -*-
"""S7 Hawkes パラメータの較正 (励起行列 a・ベースライン μ・δ0)。

設計方針
--------
S6 の**実測**定常フローを保存したまま自己励起を加える。つまり
``r = (I - aᵀ)⁻¹ μ`` が S6 の実測イベントレートに一致するよう μ を決める
(μ = (I - aᵀ) r)。これで板の物理 (スプレッド・デプスのレジーム) は S6 と同じ
土俵に載り、加わるのはクラスタリングだけになる。

★過去の誤り (2026-08-21 に実測で発覚): 初版の較正は S6 の取消レートを
δ·N̄ = 5·900 = 4500/日 と仮定した。しかし取消数は指値流入 3000/日 を超え得ない。
S6 本番 (500 日) の台帳の実測は cancelled 597,386 → **1,195/日**、したがって
N̄ = 1195/5 = **239** である。誤った r で解いた μ は CX 列が負になり、走らせると
励起駆動の取消が板を食い尽くして 42% の時間で片側が空になり、ミッドが
ティック窓から逸脱した。r は必ず実測から取ること。

出所 (S6 本番 500 日 results/S6/metrics.json の order_ledger):
  submitted_mo = 899,954   → r_MO = 1,800/日
  submitted_lo = 1,500,119 → r_LO = 3,000/日
  cancelled    =   597,386 → r_CX = 1,195/日
  (departure の残り 902,552 は完全約定 — 取消ではないので r_CX に入れない)

構造 (指示書 §3.2 の経験的パターン):
  - 対角優位: MO→MO (注文分割・モメンタム)、LO→LO (気配の競り合い)、
    CX→CX (取消カスケード)
  - MO→LO (約定後の流動性補充)、CX→LO (取消→再掲示) は正で大きめ
  - MO→CX (約定を見て気配を引く)、LO→MO (新気配が約定を誘発) は小さめ
  - CX 列は控えめに保つ: ベースライン φ·δ0·N(t) (板の在庫に比例する復元力) に
    3 割強を残す (較正結果 f_cx ≈ 0.34、他型は F_MIN=0.15)。取消レート自体を
    誤って過大 (4500/日) にすると板が食い尽くされる — 上記の実測事故。

実行: uv run python scripts/calibrate_s7_hawkes.py
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# S6 実測 (500 日本番 — 出所は docstring)
# ---------------------------------------------------------------------------
R_TARGET = np.array([1800.0, 3000.0, 1195.0])  # [MO, LO, CX] 件/日
NBAR_REF = 597386.0 / 500.0 / 5.0  # = 238.95 (取消数 / 日 / δ、δ=5 は S6 の設定値)
N_TARGET = 0.83  # 分岐比 (スペクトル半径)。指示書 §3.4: 0.80–0.85
F_MIN = 0.15  # 各型のベースライン最小シェア μ_m / r_m (負・極小 μ の回避)

# 構造行列 (相対パターン)。スケールは ρ = N_TARGET になるよう数値で決める。
STRUCTURE = np.array(
    [
        # → MO     → LO    → CX
        [0.500, 0.370, 0.080],  # MO が源
        [0.055, 0.380, 0.075],  # LO が源
        [0.060, 0.200, 0.150],  # CX が源
    ]
)


def spectral_radius(m: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(m))))


def scale_to_rho(s: np.ndarray, rho: float) -> np.ndarray:
    """ρ(c·S) = c·ρ(S) (非負行列のスペクトル半径は斉次) なので閉形式。"""
    return s * (rho / spectral_radius(s))


def main() -> int:
    s = STRUCTURE.copy()
    for it in range(200):
        a = scale_to_rho(s, N_TARGET)
        mu = R_TARGET - a.T @ R_TARGET
        f = mu / R_TARGET
        bad = f < F_MIN
        if not bad.any():
            break
        # 違反列を必要率まで縮めて再スケール (縮めた分は他列がスケールで受ける)
        for j in np.flatnonzero(bad):
            col_needed = (1.0 - F_MIN) * R_TARGET[j]
            col_actual = float(a[:, j] @ R_TARGET)
            s[:, j] *= 0.98 * col_needed / col_actual
    else:
        raise RuntimeError("較正が収束しませんでした (STRUCTURE を見直すこと)")

    rho = spectral_radius(a)
    mu = R_TARGET - a.T @ R_TARGET
    delta0 = mu[2] / NBAR_REF

    # 丸めた値 (config に載せる形) で全数値を再検算する — 丸めのせいで
    # ρ や μ が要件を割らないことの確認まで含めて較正。
    a_r = np.round(a, 4)
    rho_r = spectral_radius(a_r)
    mu_r = R_TARGET - a_r.T @ R_TARGET
    delta0_r = round(float(mu_r[2] / NBAR_REF), 4)
    mu_mo = round(float(mu_r[0] / 2.0), 2)  # 片側あたり
    mu_lo = round(float(mu_r[1] / 2.0), 2)

    print(f"iterations: {it}")
    print(f"rho(a) = {rho:.6f}  (rounded: {rho_r:.6f}, target {N_TARGET})")
    print("a (rounded, rows=source, cols=target [MO, LO, CX]):")
    for row in a_r:
        print("  (" + ", ".join(f"{v:.4f}" for v in row) + "),")
    print(f"mu (per day, both sides) = {np.round(mu_r, 1)}")
    print(f"baseline shares f = mu/r = {np.round(mu_r / R_TARGET, 3)}")
    print(f"hawkes_mu_mo (per side) = {mu_mo}")
    print(f"hawkes_mu_lo (per side) = {mu_lo}")
    print(f"hawkes_delta0 = {delta0_r}  (NBAR_REF = {NBAR_REF:.2f})")
    # 検算: 丸め値で r を復元
    mu_vec = np.array([2.0 * mu_mo, 2.0 * mu_lo, delta0_r * NBAR_REF])
    r_check = np.linalg.solve(np.eye(3) - a_r.T, mu_vec)
    print(f"r reconstructed from rounded params = {np.round(r_check, 1)}"
          f"  (target {R_TARGET})")
    err = np.abs(r_check - R_TARGET) / R_TARGET
    print(f"relative error = {np.round(err * 100, 3)} %  (max {err.max()*100:.3f}%)")
    assert rho_r < 1.0 and (mu_r > 0).all() and err.max() < 0.02
    # バースト診断: 1 イベントが直後に加える総強度 (w·β の和 × 行和)
    session = 23400.0
    tau = np.array([0.5, 10.0, 300.0])
    w = np.array([0.5, 0.3, 0.2])
    beta_day = session / tau
    wb = float(w @ beta_day)
    for i, name in enumerate(["MO", "LO", "CX"]):
        print(f"instant intensity jump per {name} event: "
              f"{a_r[i].sum() * wb:,.0f} /day (decays at tau=0.5s first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
