"""結果の永続化・プロット・段階間比較。

``results/<stage>/metrics.json`` は単なるログではなく**回帰テストの基準**である。
後段で異常が出たときに「どの段階までは正常だったか」を遡るために、段階ごとに
必ず残す。したがって書式は段階をまたいで安定していなければならない。

プロットのラベルは英語で書いてある。matplotlib の既定フォントに日本語グリフが
無く、日本語を入れると豆腐と警告の山になるため。
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import Config
from .validation.base import jsonable
from .validation.gates import GateResult
from .validation.suite import flatten

__all__ = [
    "git_info",
    "results_dir",
    "write_metrics",
    "load_metrics",
    "verify_metrics_file",
    "make_plots",
    "compare_stages",
    "REQUIRED_TOP_LEVEL_KEYS",
    "REQUIRED_METRIC_GROUPS",
]

#: metrics.json に必ず含まれていなければならない最上位キー。
REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "stage",
    "git_commit",
    "config",
    "metrics",
    "gates",
    "all_critical_passed",
    "runtime_sec",
)

#: metrics 以下に必ず含まれていなければならない群。
REQUIRED_METRIC_GROUPS: tuple[str, ...] = ("tails", "memory", "scaling", "micro", "cross")

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent


def _git(args: Sequence[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    """コード版数。リポジトリでない場合も例外にせず ``None`` を返す。"""
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": None if status is None else bool(status.strip()),
    }


def results_dir(stage: str, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else _PROJECT_ROOT / "results"
    return base / stage


def write_metrics(
    stage: str,
    config: Config,
    metrics: Mapping[str, Any],
    gate_results: Sequence[GateResult],
    gate_summary: Mapping[str, Any],
    runtime_sec: float,
    root: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """``results/<stage>/metrics.json`` を書き出してパスを返す。"""
    out_dir = results_dir(stage, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"

    info = git_info()
    payload: dict[str, Any] = {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": info["commit"],
        "git_branch": info["branch"],
        "git_dirty": info["dirty"],
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "metrics": jsonable(metrics),
        "gates": [g.to_dict() for g in gate_results],
        "gate_summary": jsonable(gate_summary),
        "all_critical_passed": bool(gate_summary.get("all_critical_passed", False)),
        "runtime_sec": float(runtime_sec),
    }
    if extra:
        payload.update(jsonable(dict(extra)))

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def load_metrics(stage: str, root: str | Path | None = None) -> dict[str, Any]:
    path = results_dir(stage, root) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} がありません。先に `python -m simchart.cli run --stage {stage}` を実行してください。"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_metrics_file(path: str | Path) -> dict[str, Any]:
    """書き出した metrics.json を読み直して必須項目の有無を確認する。

    「書いたつもり」で終わらせないために、書いた後に必ず読み直す。
    """
    path = Path(path)
    if not path.exists():
        return {"metrics_json_ok": False, "reason": f"{path} が存在しません", "path": str(path)}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {"metrics_json_ok": False, "reason": f"読み込みに失敗: {exc}", "path": str(path)}

    missing_top = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in data]
    metrics = data.get("metrics", {})
    missing_groups = [g for g in REQUIRED_METRIC_GROUPS if g not in metrics]
    ok = not missing_top and not missing_groups and bool(data.get("gates"))
    return {
        "metrics_json_ok": bool(ok),
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "missing_top_level_keys": missing_top,
        "missing_metric_groups": missing_groups,
        "n_gates": len(data.get("gates", [])),
    }


# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------
def make_plots(
    metrics: Mapping[str, Any],
    stage: str,
    result: Any | None = None,
    root: str | Path | None = None,
) -> list[Path]:
    """診断プロットを ``results/<stage>/plots/`` に書き出す。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = results_dir(stage, root) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    # 1. 価格経路
    if result is not None:
        obs = result.observation
        stride = max(obs.n_points // 5000, 1)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(obs.t[::stride] / obs.session_seconds, np.exp(obs.log_price[::stride]), lw=0.7)
        ax.set_xlabel("session index")
        ax.set_ylabel("price")
        ax.set_title(f"{stage}: observed price path (every {stride} steps)")
        save(fig, "price_path.png")

    # 2. QQ プロット
    qq = metrics.get("tails", {}).get("qq_normal", {})
    if qq.get("status") == "ok":
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        ax.plot(qq["theoretical_quantiles"], qq["empirical_quantiles"], "o", ms=2.5)
        lim = [min(qq["theoretical_quantiles"]), max(qq["theoretical_quantiles"])]
        ax.plot(lim, lim, "r-", lw=1)
        ax.set_xlabel("normal quantiles")
        ax.set_ylabel("empirical quantiles (standardized)")
        ax.set_title(f"{stage}: normal QQ  (R^2={qq['r2']:.5f})")
        save(fig, "qq_normal.png")

    # 3. ACF
    acf_r = metrics.get("memory", {}).get("acf_r", {})
    acf_abs = metrics.get("memory", {}).get("acf_abs_r", {})
    if acf_r.get("status") == "ok":
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, data, label in (
            (axes[0], acf_r, "returns"),
            (axes[1], acf_abs, "|returns|"),
        ):
            if data.get("status") != "ok":
                ax.set_visible(False)
                continue
            lags = np.array(data["lags"][1:])
            values = np.array([v if v is not None else np.nan for v in data["values"][1:]])
            show = min(len(lags), 200)
            ax.bar(lags[:show], values[:show], width=0.9)
            ax.axhline(data["conf95"], color="r", ls="--", lw=0.8)
            ax.axhline(-data["conf95"], color="r", ls="--", lw=0.8)
            ax.set_xlabel("lag")
            ax.set_ylabel("autocorrelation")
            ax.set_title(f"{stage}: ACF of {label}")
        save(fig, "acf.png")

    # 4. スケール別尖度
    kbs = metrics.get("scaling", {}).get("kurtosis_by_scale", {})
    if kbs.get("status") == "ok":
        rows = [r for r in kbs["table"] if r.get("kurtosis") is not None]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        gated = [r for r in rows if r.get("gated")]
        other = [r for r in rows if not r.get("gated")]
        for subset, marker, label in ((gated, "o", "gated"), (other, "x", "recorded only")):
            if subset:
                ax.errorbar(
                    [r["scale_sec"] for r in subset],
                    [r["kurtosis"] for r in subset],
                    yerr=[2 * (r["se_under_normal"] or 0) for r in subset],
                    fmt=marker, capsize=3, label=label,
                )
        ax.axhline(3.0, color="r", ls="--", lw=0.9, label="normal (3)")
        ax.set_xscale("log")
        ax.set_xlabel("aggregation scale (sec)")
        ax.set_ylabel("kurtosis")
        ax.set_title(f"{stage}: kurtosis vs aggregation scale (+/-2 s.e.)")
        ax.legend(fontsize=8)
        save(fig, "kurtosis_by_scale.png")

    # 5. zeta_q
    zq = metrics.get("scaling", {}).get("zeta_q", {})
    if zq.get("status") == "ok":
        fig, ax = plt.subplots(figsize=(5, 3.4))
        qs = np.array(zq["qs"])
        ax.plot(qs, zq["zetas"], "o-", label="estimated")
        ax.plot(qs, qs / 2.0, "r--", label="q/2 (monofractal)")
        ax.set_xlabel("q")
        ax.set_ylabel("zeta_q")
        ax.set_title(f"{stage}: structure function exponents (R^2={zq['r2']:.5f})")
        ax.legend(fontsize=8)
        save(fig, "zeta_q.png")

    # 6. signature plot
    sig = metrics.get("scaling", {}).get("signature_plot", {})
    if sig.get("status") == "ok":
        rows = [r for r in sig["table"] if r.get("rv_per_second")]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot([r["scale_sec"] for r in rows], [r["rv_per_second"] for r in rows], "o-")
        ax.axhline(sig["reference_rv_per_second"], color="r", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("sampling scale (sec)")
        ax.set_ylabel("realized variance per second")
        ax.set_title(f"{stage}: signature plot (max rel dev={sig['max_rel_dev']:.4f})")
        save(fig, "signature_plot.png")

    # 7. 分散比
    vr = metrics.get("scaling", {}).get("variance_ratio", {})
    if vr.get("status") == "ok":
        rows = [r for r in vr["table"] if r.get("vr") is not None]
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.errorbar(
            [r["q"] for r in rows], [r["vr"] for r in rows],
            yerr=[2 * (r["se"] or 0) for r in rows], fmt="o-", capsize=3,
        )
        ax.axhline(1.0, color="r", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("q (in primary bars)")
        ax.set_ylabel("variance ratio")
        ax.set_title(f"{stage}: variance ratio (+/-2 s.e.)")
        save(fig, "variance_ratio.png")

    return written


# ---------------------------------------------------------------------------
# 段階間比較
# ---------------------------------------------------------------------------
def compare_stages(
    stages: Sequence[str], root: str | Path | None = None, only_changed: bool = False
) -> dict[str, Any]:
    """複数段階の metrics.json を読み、指標を横並びにする。

    S2 (ラフ成分を入れた後に |r| ACF の長ラグが S1 から不変か) や S4 (phi で
    割った系列が S3 と一致するか) のゲート判定で必要になるため S0 の時点で用意する。
    """
    loaded: dict[str, dict[str, Any]] = {}
    for stage in stages:
        loaded[stage] = load_metrics(stage, root)

    flat = {stage: flatten(data.get("metrics", {})) for stage, data in loaded.items()}
    keys = sorted({k for table in flat.values() for k in table})

    rows: list[dict[str, Any]] = []
    for key in keys:
        values = {stage: flat[stage].get(key) for stage in stages}
        numeric = [v for v in values.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        delta = None
        rel = None
        if len(stages) == 2 and len(numeric) == 2:
            first, second = (values[stages[0]], values[stages[1]])
            delta = second - first
            if first not in (0, None):
                rel = delta / abs(first)
        changed = bool(delta is None or abs(delta) > 0) if len(stages) == 2 else True
        if only_changed and len(stages) == 2 and delta is not None and delta == 0:
            continue
        rows.append({"metric": key, "values": values, "delta": delta, "rel_delta": rel, "changed": changed})

    gate_rows = []
    gate_names = sorted({g["name"] for data in loaded.values() for g in data.get("gates", [])})
    for name in gate_names:
        entry: dict[str, Any] = {"gate": name}
        for stage in stages:
            match = [g for g in loaded[stage].get("gates", []) if g["name"] == name]
            entry[stage] = match[0]["passed"] if match else None
        gate_rows.append(entry)

    return {
        "stages": list(stages),
        "config_hashes": {s: loaded[s].get("config_hash") for s in stages},
        "git_commits": {s: loaded[s].get("git_commit") for s in stages},
        "all_critical_passed": {s: loaded[s].get("all_critical_passed") for s in stages},
        "metrics": rows,
        "gates": gate_rows,
    }
