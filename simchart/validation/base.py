"""検証結果の共通表現。

方針
----
検証関数は**該当層が無効でも例外を投げない。** 構造化された「N/A」を返す。
理由は 2 つある。

1. S0 の metrics.json を S1 以降の回帰テストの基準にしたいので、どの段階でも
   同じ形の辞書が出てほしい。「まだ測れない」ことも測定結果の一種として記録する。
2. 「例外が出た」と「測ったら該当なしだった」が同じ扱いになると、静かな破損を
   見逃す。前者は ``status="error"``、後者は ``status="not_applicable"`` に
   分けて、前者だけをゲート違反にする。
"""

from __future__ import annotations

import math
import traceback
from typing import Any, Callable, Mapping

import numpy as np

__all__ = ["ok", "na", "err", "is_ok", "is_na", "get_path", "jsonable", "safe_call", "num"]

STATUS_OK = "ok"
STATUS_NA = "not_applicable"
STATUS_ERROR = "error"


def num(x: Any) -> float | None:
    """numpy スカラーを Python の float へ。非有限値は ``None`` にする。

    JSON は NaN / Infinity を表現できない (厳密には ``json`` モジュールが
    非標準の literal を書いてしまう) ので、ここで潰しておく。値が無かったのか
    計算が壊れたのかは呼び出し側が別のキーで記録すること。
    """
    if x is None:
        return None
    value = float(x)
    return value if math.isfinite(value) else None


def ok(value: Any, **extra: Any) -> dict[str, Any]:
    """正常に測れた結果。"""
    out: dict[str, Any] = {"status": STATUS_OK, "value": value}
    out.update(extra)
    return out


def na(reason: str, **extra: Any) -> dict[str, Any]:
    """該当しない結果 (その層がまだ無い、標本が足りない、など)。"""
    out: dict[str, Any] = {"status": STATUS_NA, "reason": reason, "value": None}
    out.update(extra)
    return out


def err(exc: BaseException, **extra: Any) -> dict[str, Any]:
    """例外で測れなかった結果。ゲート ``validation_callable`` はこれを見る。"""
    out: dict[str, Any] = {
        "status": STATUS_ERROR,
        "value": None,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=6),
    }
    out.update(extra)
    return out


def is_ok(result: Mapping[str, Any] | None) -> bool:
    return bool(result) and result.get("status") == STATUS_OK


def is_na(result: Mapping[str, Any] | None) -> bool:
    return bool(result) and result.get("status") == STATUS_NA


def safe_call(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """検証関数を呼び、例外を ``status="error"`` の結果に変換する。

    1 つの推定器が壊れても他の指標を全部失わないようにするため。
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 意図的に全部拾う
        return err(exc, function=getattr(fn, "__name__", repr(fn)))


def get_path(data: Mapping[str, Any], path: str) -> Any:
    """``"memory.acf_r.lag1"`` のようなドット記法で入れ子辞書から値を取る。

    見つからない場合は :class:`KeyError` を送出する (``None`` を返すと
    「値が無い」と「値が None」の区別がつかなくなるため)。
    """
    node: Any = data
    for part in path.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            raise KeyError(f"指標のパス {path!r} が見つかりません (未解決の要素: {part!r})")
    return node


def jsonable(obj: Any) -> Any:
    """numpy 由来の型を JSON 化できる形へ再帰的に変換する。"""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return num(obj)
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return obj
