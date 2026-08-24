"""S7: 型レベル 3D Hawkes の MLE 推定・分岐比の 3 経路再推定・残差検定。

推定モデル (生成側 config と同一のカーネル形状、β・w は既知として固定):

    λ_y(t) = φ(u_t)·μ_y + Σ_x a[x,y] Σ_k w_k β_k Σ_{t_i^x < t} e^{−β_k (t − t_i^x)}

推定するのは振幅 a[x,y] (9) とベースライン μ_y (3) のみ。対数尤度は
ターゲット型 y ごとに分離し (log(線形) の和 − 線形)、各 4 パラメータの
**凹**最大化になる — 大域最適が一意なので単一開始点の L-BFGS-B で足りる。

★時間の規約: 立会時間を連結した「セッション秒」。エンジン自体が日境界を
特別扱いせず励起状態を持ち越すので、推定側も同じ時間軸を使う。

★CX ベースラインのモデル不一致 (意図的): 生成側は φ·δ0·N(t) (板の生存注文数に
比例)、推定側は φ·μ_cx (定数)。N(t) は N̄ 周りに揺らぐだけなので μ̂_cx ≈ δ0·N̄ を
拾う。合成データの復元テストでこの近似のバイアスが許容内であることを確認する。

分岐比の 3 経路 (指示書の中心ゲート、Filimonov & Sornette 2015 の罠の実証):

- ``n_hat_raw``      — 季節性を除去せずに推定。日内 U 字によるイベント集中を
                       自己励起と誤認し、**系統的に過大**になるはず (> n + 0.03)
- ``n_hat_true_phi`` — 真の φ_λ (生成側の値) をベースラインに与えて推定 (±0.05)
- ``n_hat_est_phi``  — イベント数から推定した φ̂_λ で同じことをする (±0.08)。
                       実データで可能なのはこの経路だけ — S4 の脱季節化機構が
                       このためにある
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from scipy import optimize, stats

from ..types import EventType
from .base import na, num, ok

__all__ = [
    "marks_from_eventlog",
    "excitation_pass",
    "phi_lookup",
    "phi_cumulative",
    "estimate_phi_lambda",
    "hawkes_mle",
    "branching_three_ways",
    "time_rescaling_test",
    "simulate_branching",
]


def marks_from_eventlog(events) -> tuple[np.ndarray, np.ndarray]:
    """EventLog → (times, marks) — MO=0, LO=1, CX=2。

    ★TRADE 行 (約定の記録であって注文流イベントではない) と **t=0 の板初期化行**
    (init_levels×2 本の LIMIT_ADD) を除く。初期化行を含めると 1 ビンに 60 件の
    スパイクが乗り、Fano が 1.95 に見える等の人工物が出る (帰無対照で実測)。
    """
    et = events.event_type
    keep = et != int(EventType.TRADE)
    t = np.asarray(events.t[keep], dtype=np.float64)
    e = et[keep]
    marks = np.full(e.size, -1, dtype=np.int64)
    marks[e == int(EventType.MARKET)] = 0
    marks[e == int(EventType.LIMIT_ADD)] = 1
    marks[e == int(EventType.CANCEL)] = 2
    sel = (marks >= 0) & (t > 0.0)
    return t[sel], marks[sel]


# ---------------------------------------------------------------------------
# 励起項の前計算 (パラメータ非依存 — β・w 固定の恩恵で尤度評価が O(N) になる)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _excitation_kernel(times, marks, betas, w, t_end):
    """1 パスで E (各イベント直前の励起強度)・I (区間積分)・tail を出す。

    E[i, x] = Σ_k w_k s_xk(t_i⁻)  — 源が型 x の励起強度 [1/秒]
    I[i, x] = ∫_{t_{i-1}}^{t_i} (同) dt — 時間再スケーリングと ∫λ 用
    tail[x] = ∫_{t_last}^{t_end} (同) dt
    """
    n = times.size
    n_k = betas.size
    s = np.zeros((3, n_k))
    e_out = np.zeros((n, 3))
    i_out = np.zeros((n, 3))
    t_prev = 0.0
    for i in range(n):
        dt = times[i] - t_prev
        for k in range(n_k):
            dec = np.exp(-betas[k] * dt)
            for x in range(3):
                i_out[i, x] += w[k] * (s[x, k] / betas[k]) * (1.0 - dec)
                s[x, k] *= dec
        for x in range(3):
            acc = 0.0
            for k in range(n_k):
                acc += w[k] * s[x, k]
            e_out[i, x] = acc
        m = marks[i]
        for k in range(n_k):
            s[m, k] += betas[k]
        t_prev = times[i]
    tail = np.zeros(3)
    dt = t_end - t_prev
    if dt > 0.0:
        for k in range(n_k):
            dec = np.exp(-betas[k] * dt)
            for x in range(3):
                tail[x] += w[k] * (s[x, k] / betas[k]) * (1.0 - dec)
    return e_out, i_out, tail


def excitation_pass(
    times: np.ndarray, marks: np.ndarray, betas: np.ndarray, w: np.ndarray, t_end: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """励起の前計算。戻り値は (E, I, C)。C[x] = ∫_0^T S_x dt (= I 列和 + tail)。"""
    times = np.ascontiguousarray(times, dtype=np.float64)
    marks = np.ascontiguousarray(marks, dtype=np.int64)
    betas = np.ascontiguousarray(betas, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    e_out, i_out, tail = _excitation_kernel(times, marks, betas, w, float(t_end))
    c = i_out.sum(axis=0) + tail
    return e_out, i_out, c


# ---------------------------------------------------------------------------
# 季節性テーブル (区分一定 — エンジンの消費形式と同一)
# ---------------------------------------------------------------------------
def phi_lookup(times: np.ndarray, table: np.ndarray, session_seconds: float) -> np.ndarray:
    """各時刻の φ (エンジンと同じ floor 参照)。"""
    m = table.size
    u = np.mod(times / session_seconds, 1.0)
    idx = np.minimum((u * m).astype(np.int64), m - 1)
    return table[idx]


def phi_cumulative(times: np.ndarray, table: np.ndarray, session_seconds: float) -> np.ndarray:
    """Φ(t) = ∫_0^t φ(u_s) ds (区分一定テーブルに対して厳密)。"""
    m = table.size
    csum = np.concatenate([[0.0], np.cumsum(table)])  # csum[j] = 先頭 j ビンの和
    d, frac = np.divmod(times, session_seconds)
    pos = frac / session_seconds * m
    j = np.minimum(pos.astype(np.int64), m - 1)
    within = (csum[j] + (pos - j) * table[j]) * (session_seconds / m)
    return d * session_seconds * float(table.mean()) + within


def z_lookup(times: np.ndarray, z_grid: np.ndarray, z_step_sec: float) -> np.ndarray:
    """各時刻の Z_t (S10c。エンジンと同じ floor 参照 + 端クリップ)。"""
    idx = np.minimum((times / z_step_sec).astype(np.int64), z_grid.size - 1)
    return z_grid[idx]


def phi_z_cumulative(
    times: np.ndarray,
    phi_table: np.ndarray | None,
    session_seconds: float,
    z_grid: np.ndarray,
    z_step_sec: float,
) -> np.ndarray:
    """∫_0^t φ(u_s)·Z_s ds (両者とも区分一定なので厳密)。

    S10c: Z はベースラインに φ と同格で乗る — 脱季節化と同じ扱いで
    ベースライン補償器に入れないと、MLE が Z のクラスタリングを励起と
    誤帰属して n̂ が上振れする (φ の raw 経路と同じ機構)。
    """
    times = np.asarray(times, dtype=np.float64)
    n_z = z_grid.size
    bounds = np.arange(n_z + 1, dtype=np.float64) * z_step_sec
    if phi_table is not None:
        phi_b = phi_cumulative(bounds, phi_table, session_seconds)
        phi_t = phi_cumulative(times, phi_table, session_seconds)
    else:
        phi_b = bounds
        phi_t = times
    seg = np.diff(phi_b) * z_grid  # 各 z ステップの ∫φ·Z
    csum = np.concatenate([[0.0], np.cumsum(seg)])
    j = np.minimum((times / z_step_sec).astype(np.int64), n_z - 1)
    return csum[j] + z_grid[j] * (phi_t - phi_b[j])


def estimate_phi_lambda(
    times: np.ndarray, session_seconds: float, n_bins: int = 52
) -> dict[str, Any]:
    """イベント数から日内プロファイル φ̂_λ(u) を推定する (平均 1 に正規化)。

    ★これは「実データで可能な経路」の再現なので、真の φ を一切参照しない。
    カーネル (≤ 300 秒 ≪ セッション) の平滑化バイアスは小さいが 0 ではない —
    その影響込みで n_hat_est_phi のゲート帯 (±0.08) が切られている。
    """
    u = np.mod(times / session_seconds, 1.0)
    counts, _ = np.histogram(u, bins=np.linspace(0.0, 1.0, n_bins + 1))
    if counts.sum() == 0 or (counts == 0).any():
        return {"table": np.ones(n_bins), "counts": counts, "degenerate": True}
    table = counts / counts.mean()
    return {"table": table, "counts": counts, "degenerate": False}


# ---------------------------------------------------------------------------
# MLE (ターゲット型ごとの凹最大化)
# ---------------------------------------------------------------------------
def _fit_one_target(
    lam_parts: np.ndarray,  # (n_y, 4): [φ_i, E_i0, E_i1, E_i2] (ターゲット y のイベントのみ)
    consts: np.ndarray,  # (4,): [Φ_T, C_0, C_1, C_2]
    x0: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    def negloglik(p):
        lam = lam_parts @ p
        if (lam <= 0).any():
            return 1e300, np.zeros(4)
        inv = 1.0 / lam
        ll = np.log(lam).sum() - consts @ p
        grad = lam_parts.T @ inv - consts
        return -ll, -grad

    res = optimize.minimize(
        negloglik, x0, jac=True, method="L-BFGS-B",
        bounds=[(1e-12, None)] * 4,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
    )
    return res.x, float(-res.fun), bool(res.success)


def hawkes_mle(
    times: np.ndarray,
    marks: np.ndarray,
    t_end: float,
    betas: np.ndarray,
    weights: np.ndarray,
    phi_table: np.ndarray | None = None,
    session_seconds: float | None = None,
    precomputed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    z_grid: np.ndarray | None = None,
    z_step_sec: float | None = None,
) -> dict[str, Any]:
    """振幅 a[x,y] とベースライン μ_y の MLE。β・w は固定 (引数で与える)。

    戻り値: mu_hat [1/秒], a_hat (3,3), n_hat = ρ(â), 対数尤度, 収束フラグ。
    ``z_grid`` (S10c の活動度 Z) を与えるとベースラインを φ·Z で補償する。
    """
    times = np.asarray(times, dtype=np.float64)
    marks = np.asarray(marks, dtype=np.int64)
    order = np.argsort(times, kind="stable")
    times, marks = times[order], marks[order]

    if phi_table is not None:
        if session_seconds is None:
            raise ValueError("phi_table には session_seconds が必要です")
        phi_i = phi_lookup(times, phi_table, session_seconds)
        # Φ_T: 区分一定テーブルの厳密積分
        phi_int = float(phi_cumulative(np.array([t_end]), phi_table, session_seconds)[0])
    else:
        phi_i = np.ones_like(times)
        phi_int = float(t_end)
    if z_grid is not None:
        if z_step_sec is None:
            raise ValueError("z_grid には z_step_sec が必要です")
        phi_i = phi_i * z_lookup(times, z_grid, z_step_sec)
        phi_int = float(
            phi_z_cumulative(
                np.array([t_end]), phi_table,
                float(session_seconds or t_end), z_grid, float(z_step_sec),
            )[0]
        )

    if precomputed is not None:
        e_mat, _, c_vec = precomputed
    else:
        e_mat, _, c_vec = excitation_pass(times, marks, betas, weights, t_end)

    consts = np.concatenate([[phi_int], c_vec])
    mu_hat = np.zeros(3)
    a_hat = np.zeros((3, 3))
    ll_total = 0.0
    converged = True
    for y in range(3):
        sel = marks == y
        n_y = int(sel.sum())
        lam_parts = np.column_stack([phi_i[sel], e_mat[sel]])
        # 開始点: 半分をベースライン、残りを励起に置く素朴な配分
        x0 = np.array([0.5 * n_y / max(phi_int, 1e-9)] + [0.5 * n_y / max(c, 1.0) / 3.0 for c in c_vec])
        p, ll, okflag = _fit_one_target(lam_parts, consts, x0)
        mu_hat[y] = p[0]
        a_hat[:, y] = p[1:]
        ll_total += ll
        converged = converged and okflag

    n_hat = float(np.max(np.abs(np.linalg.eigvals(a_hat))))
    return {
        "mu_hat_per_sec": mu_hat,
        "a_hat": a_hat,
        "n_hat": n_hat,
        "loglik": ll_total,
        "converged": converged,
        "n_events": int(times.size),
    }


# ---------------------------------------------------------------------------
# 分岐比の 3 経路
# ---------------------------------------------------------------------------
def branching_three_ways(
    times: np.ndarray,
    marks: np.ndarray,
    t_end: float,
    betas: np.ndarray,
    weights: np.ndarray,
    session_seconds: float,
    true_phi_table: np.ndarray | None,
    n_design: float,
    n_bins_est: int = 52,
    block_days: float | None = None,
    z_grid: np.ndarray | None = None,
    z_step_sec: float | None = None,
) -> dict[str, Any]:
    """raw / true-φ / est-φ̂ の 3 経路で n を再推定する (指示書の中心ゲート)。

    ``block_days`` を与えると全標本推定に加えてブロック分割の再推定を行い、
    n̂ の標本ばらつき (SD) を付す。ブロックは冷開始 (励起の持ち越し ≤300 秒 ≪
    ブロック長なので端バイアスは無視できる)。

    S10c: ``z_grid`` を与えると true 経路のベースラインを φ·Z で補償する
    (Z を知らない MLE は Z のクラスタリングを励起へ誤帰属し n̂ が上振れ —
    実測 +0.063。est 経路は日内周期しか推定できないので Z は入れない:
    それが「実データで可能な経路」の再現であり、est の帯 ±0.08 が緩い理由)。
    """
    times = np.asarray(times, dtype=np.float64)
    marks = np.asarray(marks, dtype=np.int64)
    if times.size < 1000:
        return na(f"イベント数が足りません (n={times.size})")
    order = np.argsort(times, kind="stable")
    times, marks = times[order], marks[order]

    zkw: dict[str, Any] = {}
    if z_grid is not None and z_step_sec is not None:
        zkw = {"z_grid": np.asarray(z_grid, dtype=np.float64),
               "z_step_sec": float(z_step_sec)}
    pre = excitation_pass(times, marks, betas, weights, t_end)
    fit_raw = hawkes_mle(times, marks, t_end, betas, weights, precomputed=pre)
    fit_true = (
        hawkes_mle(
            times, marks, t_end, betas, weights,
            phi_table=true_phi_table, session_seconds=session_seconds, precomputed=pre,
            **zkw,
        )
        if true_phi_table is not None
        else None
    )
    # S12 §7.2 の B 経路: 真の φ のみ (Z = φ 外のベースライン変調を知らない)。
    # n̂_B − n̂_E が Z (c_vol の V + χ₁) による膨張幅 — S4/S7 の罠の再演を
    # 実測で示す record。Z が無い構成では true と同一なので省略。
    fit_phi_only = (
        hawkes_mle(
            times, marks, t_end, betas, weights,
            phi_table=true_phi_table, session_seconds=session_seconds, precomputed=pre,
        )
        if (true_phi_table is not None and zkw)
        else None
    )
    est = estimate_phi_lambda(times, session_seconds, n_bins=n_bins_est)
    fit_est = hawkes_mle(
        times, marks, t_end, betas, weights,
        phi_table=est["table"], session_seconds=session_seconds, precomputed=pre,
    )

    blocks: dict[str, Any] | None = None
    if block_days is not None:
        block_sec = block_days * session_seconds
        n_blocks = int(t_end // block_sec)
        rows = []
        for b in range(n_blocks):
            lo, hi = b * block_sec, (b + 1) * block_sec
            sel = (times >= lo) & (times < hi)
            if int(sel.sum()) < 1000:
                continue
            tb = times[sel] - lo
            mb = marks[sel]
            kw: dict[str, Any] = {}
            if true_phi_table is not None:
                kw = {"phi_table": true_phi_table, "session_seconds": session_seconds}
            if zkw:
                # ブロック内の Z (グリッドを切り出してブロック原点に平行移動)
                j0 = int(lo / zkw["z_step_sec"])
                j1 = max(j0 + 1, int(np.ceil(hi / zkw["z_step_sec"])))
                kw["z_grid"] = zkw["z_grid"][j0:min(j1, zkw["z_grid"].size)]
                kw["z_step_sec"] = zkw["z_step_sec"]
            f = hawkes_mle(tb, mb, block_sec, betas, weights, **kw)
            rows.append(f["n_hat"])
        arr = np.array(rows)
        blocks = {
            "path": "true_phi" if true_phi_table is not None else "raw",
            "block_days": block_days,
            "n_blocks": int(arr.size),
            "n_hat_mean": num(arr.mean()) if arr.size else None,
            "n_hat_sd": num(arr.std(ddof=1)) if arr.size > 1 else None,
            "n_hat_blocks": [num(v) for v in arr],
        }

    n_raw = fit_raw["n_hat"]
    n_true = fit_true["n_hat"] if fit_true is not None else None
    n_est = fit_est["n_hat"]
    out = {
        "n_design": num(n_design),
        "n_hat_raw": num(n_raw),
        "n_hat_true_phi": num(n_true),
        "n_hat_est_phi": num(n_est),
        "raw_minus_design": num(n_raw - n_design),
        "true_phi_minus_design": num(n_true - n_design) if n_true is not None else None,
        "est_phi_minus_design": num(n_est - n_design),
        "raw_inflation_over_true": num(n_raw - n_true) if n_true is not None else None,
        "n_hat_phi_only": num(fit_phi_only["n_hat"]) if fit_phi_only else None,
        "z_inflation": (
            num(fit_phi_only["n_hat"] - n_true)
            if (fit_phi_only and n_true is not None) else None
        ),
        "converged": bool(
            fit_raw["converged"]
            and fit_est["converged"]
            and (fit_true is None or fit_true["converged"])
        ),
        "n_events": int(times.size),
        "a_hat_true_phi": (
            [[num(v) for v in row] for row in fit_true["a_hat"]] if fit_true else None
        ),
        "mu_hat_true_phi_per_day": (
            [num(v * session_seconds) for v in fit_true["mu_hat_per_sec"]] if fit_true else None
        ),
        "phi_est_degenerate": bool(est["degenerate"]),
        "blocks": blocks,
    }
    return ok(num(n_true if n_true is not None else n_est), **out)


# ---------------------------------------------------------------------------
# 残差検定 (時間再スケーリング)
# ---------------------------------------------------------------------------
def time_rescaling_test(
    times: np.ndarray,
    marks: np.ndarray,
    t_end: float,
    betas: np.ndarray,
    weights: np.ndarray,
    mu_per_sec: np.ndarray,
    a_mat: np.ndarray,
    phi_table: np.ndarray | None = None,
    session_seconds: float | None = None,
    z_grid: np.ndarray | None = None,
    z_step_sec: float | None = None,
) -> dict[str, Any]:
    """Λ(t) = ∫λ_tot による時間変換後のイベント間隔が Exp(1) かの KS 検定。

    モデルが正しければ (Ogata 1988) 変換後は単位 Poisson。ベースライン積分は
    区分一定 φ に対して厳密、励起積分は指数カーネルの閉形式 (前計算 I)。
    S10c: ``z_grid`` を与えるとベースライン補償器に φ·Z を使う (Z を入れないと
    補償器がモデルと違うので間隔が Exp(1) にならないのは当然の帰結)。
    """
    times = np.asarray(times, dtype=np.float64)
    marks = np.asarray(marks, dtype=np.int64)
    order = np.argsort(times, kind="stable")
    times, marks = times[order], marks[order]
    if times.size < 100:
        return na(f"イベント数が足りません (n={times.size})")

    _, i_mat, _ = excitation_pass(times, marks, betas, weights, t_end)
    if z_grid is not None:
        if z_step_sec is None:
            raise ValueError("z_grid には z_step_sec が必要です")
        cum = phi_z_cumulative(
            times, phi_table, float(session_seconds or t_end),
            np.asarray(z_grid, dtype=np.float64), float(z_step_sec),
        )
    elif phi_table is not None:
        if session_seconds is None:
            raise ValueError("phi_table には session_seconds が必要です")
        cum = phi_cumulative(times, phi_table, session_seconds)
    else:
        cum = times.copy()
    base_int = np.diff(np.concatenate([[0.0], cum])) * float(np.sum(mu_per_sec))
    row_sums = np.asarray(a_mat, dtype=np.float64).sum(axis=1)
    exc_int = i_mat @ row_sums
    taus = base_int + exc_int
    ks_stat, ks_p = stats.kstest(taus, "expon")
    return ok(
        num(ks_p),
        ks_stat=num(ks_stat),
        ks_pvalue=num(ks_p),
        mean_tau=num(taus.mean()),  # 正しければ 1
        n_intervals=int(taus.size),
    )


# ---------------------------------------------------------------------------
# 分枝 (クラスタ) 表現による独立生成 — thinning 生成器の相互検証用
# ---------------------------------------------------------------------------
def simulate_branching(
    mu_per_sec: np.ndarray,
    a_mat: np.ndarray,
    betas_per_sec: np.ndarray,
    weights: np.ndarray,
    t_end: float,
    rng: np.random.Generator,
    phi_table: np.ndarray | None = None,
    session_seconds: float | None = None,
    max_events: int = 20_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """immigrant–offspring 表現で 3D Hawkes を生成する (板なし・イベント列のみ)。

    thinning (エンジン) と数学的に等価な過程の**別実装** — 実現レートと
    クラスタリング統計の一致を相互検証に使う。CX ベースラインは定数 μ_cx
    (エンジンの φ·δ0·N(t) の N̄ 近似) なので、比較は近似許容で行うこと。
    """
    mu = np.asarray(mu_per_sec, dtype=np.float64)
    a = np.asarray(a_mat, dtype=np.float64)
    betas = np.asarray(betas_per_sec, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    times_list: list[np.ndarray] = []
    marks_list: list[np.ndarray] = []
    stack_t: list[np.ndarray] = []
    stack_m: list[np.ndarray] = []

    # immigrants (φ は区分一定なのでビンごとの Poisson + ビン内一様が厳密)
    for y in range(3):
        if phi_table is None:
            n_im = rng.poisson(mu[y] * t_end)
            t_im = rng.uniform(0.0, t_end, n_im)
        else:
            if session_seconds is None:
                raise ValueError("phi_table には session_seconds が必要です")
            m = phi_table.size
            bin_len = session_seconds / m
            n_days = int(np.ceil(t_end / session_seconds))
            lam_bin = mu[y] * phi_table * bin_len  # 1 日分の各ビンの期待数
            counts = rng.poisson(np.tile(lam_bin, n_days))
            starts = (np.arange(m * n_days) * bin_len)[counts > 0]
            reps = counts[counts > 0]
            t_im = np.repeat(starts, reps) + rng.uniform(0.0, bin_len, int(reps.sum()))
            t_im = t_im[t_im < t_end]
        times_list.append(t_im)
        marks_list.append(np.full(t_im.size, y, dtype=np.int64))
        stack_t.append(t_im)
        stack_m.append(np.full(t_im.size, y, dtype=np.int64))

    total = sum(t.size for t in times_list)
    # 世代ごとに配列処理 (Python ループはイベント単位でなく世代単位)
    while stack_t:
        pt = np.concatenate(stack_t)
        pm = np.concatenate(stack_m)
        stack_t, stack_m = [], []
        if pt.size == 0:
            break
        for x in range(3):
            src = pt[pm == x]
            if src.size == 0:
                continue
            for y in range(3):
                n_child = rng.poisson(a[x, y], src.size)
                tot = int(n_child.sum())
                if tot == 0:
                    continue
                parent_t = np.repeat(src, n_child)
                k_sel = rng.choice(betas.size, size=tot, p=w)
                delays = rng.exponential(1.0 / betas[k_sel])
                child_t = parent_t + delays
                child_t = child_t[child_t < t_end]
                if child_t.size == 0:
                    continue
                total += child_t.size
                if total > max_events:
                    raise RuntimeError("simulate_branching: max_events を超過 (n が 1 に近すぎないか)")
                cm = np.full(child_t.size, y, dtype=np.int64)
                times_list.append(child_t)
                marks_list.append(cm)
                stack_t.append(child_t)
                stack_m.append(cm)

    times = np.concatenate(times_list)
    marks = np.concatenate(marks_list)
    order = np.argsort(times, kind="stable")
    return times[order], marks[order]
