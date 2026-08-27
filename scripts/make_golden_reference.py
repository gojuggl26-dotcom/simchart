"""S5 §13: L2 構造凍結のゴールデン参照データを作る。

固定シード (seed=42、本番設定) の完全パスを保存する:
  log_sigma (117M 点 float64) / log_p_star (同) / jump_times / overnight_gaps /
  chi_2 (系固有グリッド)

用途: S6 以降で L2 に触れていないことの検証。`--verify` で再生成して SHA256 を
照合する — 一致すれば L2 は (コード変更があっても) ビット単位で凍結時と同一。
digest (metrics.json) だけでも検出はできるが、フルパスがあるとどこから
ずれたかを診断できる。

保存先: results/S5/golden/ (npz 圧縮 ~1.6GB + sidecar JSON)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from simchart import Config, run
from simchart.validation.base import jsonable

ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: str) -> Config:
    return Config.load(ROOT / path if not Path(path).is_absolute() else path)


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def build(config_path: str) -> dict:
    config = _load_config(config_path)
    result = run(config)
    chaos_meta = result.meta["l2"].get("chaos") or {}
    arrays = {
        "log_sigma": result.price.log_vol,
        "log_p_star": result.price.log_p_star,
        "jump_times": result.price.jump_times,
        "overnight_gaps": result.price.overnight_gaps,
    }
    hashes = {name: _sha(arr) for name, arr in arrays.items()}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = None
    sidecar = {
        "stage": config.stage,
        "seed": config.seed,
        "n_days": config.n_days,
        "steps_per_day": config.steps_per_day,
        "result_digest": result.digest(),
        "array_sha256": hashes,
        "chi2_sha256": chaos_meta.get("sha256"),
        "git_commit": commit,
        "config": jsonable(config.to_dict()),
        "note": (
            "L2 構造凍結 (S5 §13) のゴールデン参照。--verify で再生成照合。"
            "sigma_bar の S10 での上方修正は凍結違反ではない (水準と構造の区別)。"
        ),
    }
    return {"arrays": arrays, "sidecar": sidecar, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s5.yaml")
    parser.add_argument("--out", default="results/S5/golden")
    parser.add_argument(
        "--verify", action="store_true",
        help="保存済み参照と再生成を SHA256 で照合する (保存はしない)",
    )
    args = parser.parse_args()
    out_dir = ROOT / args.out
    sidecar_path = out_dir / "golden_reference.json"
    npz_path = out_dir / "golden_paths.npz"

    if args.verify:
        stored = json.loads(sidecar_path.read_text(encoding="utf-8"))
        print("再生成中 (キャッシュ済みなら ~40 秒)...", flush=True)
        built = build(args.config)
        ok = True
        for name, sha in built["sidecar"]["array_sha256"].items():
            match = sha == stored["array_sha256"].get(name)
            ok &= match
            print(f"  {name:16s} {'一致' if match else '不一致'}")
        print(f"結果: {'L2 は凍結時とビット単位で同一' if ok else '凍結時から変化あり'}")
        return 0 if ok else 1

    print("ゴールデン参照を生成中...", flush=True)
    built = build(args.config)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("書き出し中 (npz 圧縮 ~1.9GB)...", flush=True)
    np.savez_compressed(npz_path, **built["arrays"])
    sidecar_path.write_text(
        json.dumps(built["sidecar"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    size_gb = npz_path.stat().st_size / 1024**3
    print(f"保存: {npz_path} ({size_gb:.2f} GB) + {sidecar_path.name}")
    for name, sha in built["sidecar"]["array_sha256"].items():
        print(f"  {name:16s} sha256 {sha[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
