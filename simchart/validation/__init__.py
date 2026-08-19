"""検証スイート。

**全関数を S0 の時点で実装してある。** 該当する層がまだ無い関数 (micro / cross) も
スタブではなく本実装で、入力が無い場合に構造化された ``not_applicable`` を返す。
こうしておくと S0 の metrics.json がそのまま S1 以降の回帰テストの基準になり、
「どの段階までは正常だったか」を後から遡れる。
"""

from . import base, cross, ensemble, gates, memory, micro, scaling, suite, tails
from .base import is_na, is_ok, na, ok
from .gates import Gate, GateResult, evaluate, gates_for, summarize
from .suite import run_all

__all__ = [
    "base",
    "cross",
    "ensemble",
    "gates",
    "memory",
    "micro",
    "scaling",
    "suite",
    "tails",
    "ok",
    "na",
    "is_ok",
    "is_na",
    "Gate",
    "GateResult",
    "evaluate",
    "gates_for",
    "summarize",
    "run_all",
]
