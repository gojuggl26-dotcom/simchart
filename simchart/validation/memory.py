"""記憶 (自己相関・長期記憶) の検証。

S0 での期待値
-------------
リターンにも |リターン| にも記憶が無い。``rho(1) ~ 0``、Ljung-Box は棄却せず、
GPH / local Whittle の ``d ~ 0``。**ここでボラティリティ・クラスタリングが
出ていたら、S0 に入れてはいけないものが入っている。**

セッション境界の扱い
--------------------
自己相関はセッション内で完結する差分だけから計算する。S0 にはオーバーナイトが
無いので実質的な違いは無いが、S4 でギャップを入れた瞬間に「日をまたぐ差分」が
巨大な外れ値として ACF を汚す。最初からその構造で測る
(:meth:`simchart.types.Observation.to_bars` が 2 次元で返すのはこのため)。
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sp_fft
from scipy import optimize, stats

from .base import na, num, ok

__all__ = [
    "acf",
    "ljung_box",
    "gph_estimator",
    "local_whittle",
    "power_law_fit",
    "acf_power_law",
    "acf_powerlaw_fit",
    "vol_increment_acf",
    "leverage_function",
]


def leverage_function(
    r_daily: np.ndarray,
    rv_daily: np.ndarray | None = None,
    horizons=tuple(range(0, 31)),
) -> dict:
    """レバレッジ関数 L(h) = corr(r_t, |r_{t+h}|) と corr(r_t, RV_{t+h}) (S3)。

    実証的には h=0〜1 日で強く負、10〜30 日かけて減衰する。h<0 側 (過去の |r| と
    今日の r) は ~0 が正しい (因果の向きの確認) ので併せて記録する。
    """
    r = np.asarray(r_daily, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 300:
        return na(f"日次リターンが足りません (n={r.size})")
    a = np.abs(r)
    rows = []
    for h in horizons:
        h = int(h)
        if h == 0:
            c_abs = float(np.corrcoef(r, a)[0, 1])
            c_rv = (
                float(np.corrcoef(r, rv_daily)[0, 1]) if rv_daily is not None else None
            )
        else:
            c_abs = float(np.corrcoef(r[:-h], a[h:])[0, 1])
            c_rv = (
                float(np.corrcoef(r[:-h], np.asarray(rv_daily)[h:])[0, 1])
                if rv_daily is not None
                else None
            )
        rows.append({"h": h, "corr_abs": num(c_abs), "corr_rv": num(c_rv) if c_rv is not None else None})

    # 因果の向きの対照: 過去の |r| と今日の r (ほぼ 0 のはず)。
    reverse = [
        {"h": -h, "corr_abs": num(float(np.corrcoef(a[:-h], r[h:])[0, 1]))}
        for h in (1, 5, 10)
    ]
    l1 = next((row["corr_rv"] if row["corr_rv"] is not None else row["corr_abs"])
              for row in rows if row["h"] == 1)
    neg_range = [row["corr_abs"] for row in rows if 1 <= row["h"] <= 20]
    all_neg = bool(all(v is not None and v < 0 for v in neg_range))
    mean_1_20 = float(np.mean([v for v in neg_range if v is not None])) if neg_range else None
    l1_abs = next(row["corr_abs"] for row in rows if row["h"] == 1)
    return ok(
        num(l1),
        corr_r_rv_h1=num(l1),
        corr_abs_h1=num(l1_abs),
        all_negative_h1_20=all_neg,
        mean_h1_20=num(mean_1_20) if mean_1_20 is not None else None,
        # 弱いレバレッジ水準では個々の L(h) がゼロ近傍でノイズに埋まるため、
        # 形状判定は「L(1) 負 かつ 平均負」で行う (2026-08-20 裁定)。
        shape_ok=bool(
            l1_abs is not None and l1_abs < 0
            and mean_1_20 is not None and mean_1_20 < 0
        ),
        n=int(r.size),
        se_iid=num(1.0 / np.sqrt(r.size)),
        table=rows,
        reverse_check=reverse,
    )


def _as_2d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("入力は 1 次元または (セッション, 標本) の 2 次元である必要があります")
    return a


def acf(x: np.ndarray, max_lag: int = 100, demean: bool = True) -> dict:
    """自己相関関数。

    ``x`` が 2 次元なら行をセッションとみなし、**行内で完結するラグだけ**を使って
    自己共分散をプールする。分母は全標本数で固定する (biased 推定量)。ラグごとに
    有効標本数で割る unbiased 版は正定値性を失い、大きなラグで暴れるため。

    Returns
    -------
    dict
        ``value`` はラグ 1 の自己相関。``lag1_z`` は ``rho(1) * sqrt(N)`` で、
        独立系列なら標準正規に従う (ゲートはこれを使う)。
    """
    a = _as_2d(x)
    n_rows, n_cols = a.shape
    if max_lag < 1:
        raise ValueError("max_lag は 1 以上である必要があります")
    if max_lag >= n_cols:
        max_lag = n_cols - 1
    total = n_rows * n_cols
    if total < 50:
        return na(f"標本数が足りません (N={total})")

    y = a - (a.mean() if demean else 0.0)
    nfft = sp_fft.next_fast_len(2 * n_cols)
    spectrum = sp_fft.rfft(y, n=nfft, axis=1)
    acov_rows = sp_fft.irfft(spectrum * np.conj(spectrum), n=nfft, axis=1)[:, : max_lag + 1]
    acov = acov_rows.sum(axis=0) / total
    if acov[0] <= 0:
        return na("分散が 0 です")
    rho = acov / acov[0]

    conf = 1.96 / np.sqrt(total)
    return ok(
        num(rho[1]),
        n=int(total),
        n_sessions=int(n_rows),
        max_lag=int(max_lag),
        lag1=num(rho[1]),
        lag1_z=num(rho[1] * np.sqrt(total)),
        conf95=num(conf),
        n_outside_conf95=int(np.sum(np.abs(rho[1:]) > conf)),
        expected_outside_conf95=num(0.05 * max_lag),
        lags=list(range(0, max_lag + 1)),
        values=[num(v) for v in rho],
    )


def ljung_box(x: np.ndarray, lags=(1, 5, 10, 20, 50), primary_lag: int | None = None) -> dict:
    """Ljung-Box 検定。

    ``Q(m) = N(N+2) sum_{h=1..m} rho(h)^2 / (N-h)``、自由度 m のカイ二乗で評価する。
    :func:`acf` と同じくセッション構造を尊重してプールした自己相関を使う。

    Notes
    -----
    複数のラグで検定するので、最小 p 値をそのまま判定に使うと多重比較で偽陽性が
    出る (ラグ 5 本なら、独立系列でも「どれかが 1% を切る」確率は 1% より高い)。
    したがってゲートは ``pvalue_primary`` (既定でラグ 20) 単独を見る。他のラグは
    記録のみ。``min_pvalue`` も併記するが、判定には使わない。
    """
    lags = tuple(int(v) for v in lags)
    max_lag = max(lags)
    base = acf(x, max_lag=max_lag)
    if base["status"] != "ok":
        return base
    rho = np.array([v if v is not None else np.nan for v in base["values"]], dtype=np.float64)
    n = float(base["n"])

    rows = []
    for m in lags:
        h = np.arange(1, m + 1)
        q = n * (n + 2.0) * np.sum(rho[1 : m + 1] ** 2 / (n - h))
        p = float(stats.chi2.sf(q, df=m))
        rows.append({"lag": int(m), "stat": num(q), "pvalue": num(p)})

    primary = primary_lag if primary_lag is not None else max(lags)
    match = [row for row in rows if row["lag"] == primary]
    if not match:
        raise ValueError(f"primary_lag={primary} が lags に含まれていません")
    pvalues = [row["pvalue"] for row in rows if row["pvalue"] is not None]
    return ok(
        match[0]["pvalue"],
        n=int(n),
        primary_lag=int(primary),
        pvalue_primary=match[0]["pvalue"],
        stat_primary=match[0]["stat"],
        min_pvalue=num(min(pvalues)) if pvalues else None,
        table=rows,
    )


def _periodogram(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """フーリエ周波数 ``lambda_j = 2 pi j / n`` (j=1..n//2) での周辺スペクトル。"""
    y = np.asarray(x, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    n = y.size
    y = y - y.mean()
    spectrum = sp_fft.rfft(y)
    power = (np.abs(spectrum) ** 2) / (2.0 * np.pi * n)
    j = np.arange(spectrum.size)
    lam = 2.0 * np.pi * j / n
    return lam[1:], power[1:]  # j=0 (平均) は捨てる


def _bandwidth(n: int, exponent: float, m: int | None) -> int:
    if m is not None:
        return int(m)
    return max(int(n**exponent), 8)


def gph_estimator(
    x: np.ndarray, bandwidth_exponent: float = 0.65, m: int | None = None
) -> dict:
    """GPH (log-periodogram) 回帰による長期記憶パラメータ d の推定。

    ``log I(lambda_j) = c - d * log(4 sin^2(lambda_j / 2)) + e_j`` を j=1..m で
    最小二乗する。漸近分散は ``pi^2 / (24 m)``。

    バンド幅について
    ----------------
    古典的な ``m = N^0.5`` は既定設定 (N ~ 195,000) だと d の標準誤差が 0.030 に
    なり、ゲート ``|d| < 0.05`` が 1.6 標準誤差しか無い実質無意味な検定になる。
    既定を ``N^0.65`` (標準誤差 ~0.011) にしてあるのはそのため。バイアスと分散の
    トレードオフを見るために、呼び出し側は複数の指数でプロファイルを取ること。
    """
    y = np.asarray(x, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    n = y.size
    if n < 200:
        return na(f"標本数が足りません (n={n})")
    lam, power = _periodogram(y)
    m_eff = min(_bandwidth(n, bandwidth_exponent, m), lam.size)
    if m_eff < 8:
        return na(f"バンド幅が小さすぎます (m={m_eff})")

    lam_m = lam[:m_eff]
    power_m = power[:m_eff]
    positive = power_m > 0
    if positive.sum() < 8:
        return na("周辺スペクトルの正値が足りません")

    regressor = -np.log(4.0 * np.sin(lam_m[positive] / 2.0) ** 2)
    response = np.log(power_m[positive])
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(regressor, response)
    se_asym = float(np.pi / np.sqrt(24.0 * m_eff))
    return ok(
        num(slope),
        d=num(slope),
        se_asymptotic=num(se_asym),
        se_regression=num(stderr),
        t_stat=num(slope / se_asym) if se_asym > 0 else None,
        m=int(m_eff),
        n=int(n),
        bandwidth_exponent=float(bandwidth_exponent),
        intercept=num(intercept),
        r2=num(rvalue**2),
        regression_pvalue=num(pvalue),
    )


def local_whittle(
    x: np.ndarray, bandwidth_exponent: float = 0.65, m: int | None = None
) -> dict:
    """local Whittle 推定量による d の推定。

    ``R(d) = log( mean_j lambda_j^{2d} I_j ) - (2d/m) sum_j log lambda_j`` を
    最小化する。漸近標準誤差は ``1 / (2 sqrt(m))`` で GPH より効率が良い。
    """
    y = np.asarray(x, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    n = y.size
    if n < 200:
        return na(f"標本数が足りません (n={n})")
    lam, power = _periodogram(y)
    m_eff = min(_bandwidth(n, bandwidth_exponent, m), lam.size)
    if m_eff < 8:
        return na(f"バンド幅が小さすぎます (m={m_eff})")

    lam_m = lam[:m_eff]
    power_m = power[:m_eff]
    log_lam_mean = float(np.mean(np.log(lam_m)))

    def objective(d: float) -> float:
        g = np.mean(lam_m ** (2.0 * d) * power_m)
        if not np.isfinite(g) or g <= 0:
            return np.inf
        return float(np.log(g) - 2.0 * d * log_lam_mean)

    res = optimize.minimize_scalar(objective, bounds=(-0.499, 0.999), method="bounded")
    if not res.success:
        return na(f"最適化に失敗しました: {res.message}")
    d_hat = float(res.x)
    return ok(
        num(d_hat),
        d=num(d_hat),
        se_asymptotic=num(1.0 / (2.0 * np.sqrt(m_eff))),
        t_stat=num(d_hat * 2.0 * np.sqrt(m_eff)),
        m=int(m_eff),
        n=int(n),
        bandwidth_exponent=float(bandwidth_exponent),
        objective=num(res.fun),
    )


def power_law_fit(lags: np.ndarray, values: np.ndarray, lag_range: tuple[int, int]) -> dict:
    """``values(lag) ~ lag^{-gamma}`` の両対数回帰。

    正値だけを使う。S0 のように記憶が無い系列では正値がまばらになり推定が
    無意味になるため、有効点が半分未満なら ``not_applicable`` を返す。
    """
    lag_arr = np.asarray(lags, dtype=np.float64)
    val_arr = np.asarray(values, dtype=np.float64)
    lo, hi = lag_range
    window = (lag_arr >= lo) & (lag_arr <= hi) & np.isfinite(val_arr)
    n_window = int(window.sum())
    if n_window < 8:
        return na(f"回帰に使えるラグが足りません (n={n_window})", lag_range=[lo, hi])
    positive = window & (val_arr > 0)
    n_positive = int(positive.sum())
    if n_positive < max(8, n_window // 2):
        return na(
            f"正値が足りず冪則を当てはめられません (正値 {n_positive}/{n_window})。"
            f" 記憶が無い系列では正常な結果です。",
            lag_range=[lo, hi],
            n_window=n_window,
            n_positive=n_positive,
        )
    slope, intercept, rvalue, _, stderr = stats.linregress(
        np.log(lag_arr[positive]), np.log(val_arr[positive])
    )
    return ok(
        num(-slope),
        gamma=num(-slope),
        se=num(stderr),
        intercept=num(intercept),
        r2=num(rvalue**2),
        n_points=n_positive,
        lag_range=[lo, hi],
    )


def acf_power_law(acf_result: dict, lag_range: tuple[int, int]) -> dict:
    """:func:`acf` の結果に冪則を当てはめる (``rho(h) ~ h^{-gamma}``)。"""
    if acf_result.get("status") != "ok":
        return na("元の ACF が有効ではありません")
    lags = np.asarray(acf_result["lags"], dtype=np.float64)
    values = np.array(
        [np.nan if v is None else v for v in acf_result["values"]], dtype=np.float64
    )
    return power_law_fit(lags, values, lag_range)


def vol_increment_acf(log_vol: np.ndarray, max_lag: int = 60) -> dict:
    """log sigma の**増分**の ACF (S2 追加)。

    ラフ成分 (H < 1/2) は反持続的で、増分の 1 ラグ自己相関は負になる
    (fGn では 2^{2H-1} - 1、H=0.1 で ~-0.43)。定数ボラや持続的なボラでは
    負にならないので、粗さの存在の独立な検査になる。入力はラフグリッド解像度
    (60 秒) でサンプルした log sigma を想定。
    """
    x = np.asarray(log_vol, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 200:
        return na(f"標本数が足りません (n={x.size})")
    d = np.diff(x)
    if float(d.var()) <= 0:
        return na("log sigma が定数です (増分の分散が 0)")
    base = acf(d, max_lag=min(max_lag, d.size - 1))
    if base["status"] != "ok":
        return base
    return ok(
        base["lag1"],
        lag1=base["lag1"],
        lag1_z=base["lag1_z"],
        n=base["n"],
        lags=base["lags"][:21],
        values=base["values"][:21],
    )


def acf_powerlaw_fit(
    x: np.ndarray,
    lag_range: tuple[int, int],
    max_lag: int | None = None,
    n_bins: int = 12,
) -> dict:
    """系列 ``x`` の ACF に log-log 回帰で冪則を当て、減衰指数と R^2 を返す。

    指数減衰との識別は R^2 で行う: 真の冪則なら log-log で直線 (R^2 高)、
    指数減衰なら下に凸に折れて R^2 が下がる。

    **回帰は生のラグ点ではなく対数間隔ビンの平均に対して行う。** 個々の ACF 点は
    SE ~ 1/sqrt(N) のノイズを持ち、遠いラグでは真値がノイズと同程度になる。
    log を取るとこのノイズが増幅・非対称化され (正に振れた点だけが log で残り、
    負に振れた点は捨てられる)、R^2 が「曲線の直線性」ではなく「ノイズ量」を測る
    統計量に化けてしまう。べき則検定の定石どおり、対数等間隔ビンで平均してから
    (ビン内 ~m 点で SE が 1/sqrt(m) に減る) 回帰する。生ラグ点の回帰結果も
    ``raw_fit`` として併記する。
    """
    lo, hi = int(lag_range[0]), int(lag_range[1])
    effective_max = max_lag if max_lag is not None else hi
    base = acf(x, max_lag=effective_max)
    if base["status"] != "ok":
        return base
    lags = np.asarray(base["lags"], dtype=np.float64)
    values = np.array([np.nan if v is None else v for v in base["values"]], dtype=np.float64)

    window = (lags >= lo) & (lags <= hi) & np.isfinite(values)
    lag_w = lags[window]
    val_w = values[window]
    if lag_w.size < 8:
        return na(f"当てはめ範囲のラグが足りません (n={lag_w.size})", lag_range=[lo, hi])

    # 対数等間隔のビン境界。各ビンに最低 1 ラグが入るよう重複境界は潰す。
    edges = np.unique(
        np.round(np.geomspace(lo, hi + 1, n_bins + 1)).astype(np.int64)
    )
    bin_lags: list[float] = []
    bin_vals: list[float] = []
    bin_counts: list[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (lag_w >= a) & (lag_w < b)
        if not np.any(mask):
            continue
        mean_val = float(val_w[mask].mean())
        bin_lags.append(float(np.exp(np.log(lag_w[mask]).mean())))  # 幾何平均ラグ
        bin_vals.append(mean_val)
        bin_counts.append(int(mask.sum()))

    positive = [(l_, v_) for l_, v_ in zip(bin_lags, bin_vals) if v_ > 0]
    if len(positive) < max(5, len(bin_lags) // 2):
        return na(
            f"ビン平均 ACF の正値が足りません (正 {len(positive)}/{len(bin_lags)})。"
            f" 記憶が無い系列では正常な結果です。",
            bins=[{"lag": l_, "acf": v_} for l_, v_ in zip(bin_lags, bin_vals)],
        )
    lx = np.log([p[0] for p in positive])
    ly = np.log([p[1] for p in positive])
    slope, intercept, rvalue, _, stderr = stats.linregress(lx, ly)

    raw_fit = power_law_fit(lags, values, (lo, hi))
    gamma = -slope
    return ok(
        num(gamma),
        gamma=num(gamma),
        se=num(stderr),
        r2=num(rvalue**2),
        intercept=num(intercept),
        n_bins=len(positive),
        bin_counts=bin_counts,
        bins=[{"lag": num(l_), "acf": num(v_)} for l_, v_ in zip(bin_lags, bin_vals)],
        acf_lag1=base["lag1"],
        implied_d=num((1.0 - gamma) / 2.0),
        lag_range=[lo, hi],
        raw_fit={
            "r2": raw_fit.get("r2"),
            "gamma": raw_fit.get("gamma"),
            "status": raw_fit.get("status"),
        },
    )
