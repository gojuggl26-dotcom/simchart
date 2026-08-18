"""段階構築式の金融マイクロ構造シミュレータ。

S0 (骨格層) の時点で存在するのは幾何ブラウン運動だけで、価格モデルとしては
意図的に非現実的である。S0 の価値はフラグ設計・RNG 設計・検証スイート・
結果永続化にあり、それらが S1〜S13 全部を支える。

    from simchart import Config, run
    result = run(Config())
"""

from .config import IMPLEMENTED_STAGES, STAGES, Config, ValidationConfig
from .pipeline import determinism_check, rng_stability_check, run
from .rng import RNGRegistry, derive_seed
from .types import (
    BarSeries,
    BookSnapshot,
    EventLog,
    EventType,
    Observation,
    PriceProcess,
    StageResult,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ValidationConfig",
    "STAGES",
    "IMPLEMENTED_STAGES",
    "RNGRegistry",
    "derive_seed",
    "PriceProcess",
    "EventLog",
    "EventType",
    "BookSnapshot",
    "Observation",
    "BarSeries",
    "StageResult",
    "run",
    "determinism_check",
    "rng_stability_check",
    "__version__",
]
