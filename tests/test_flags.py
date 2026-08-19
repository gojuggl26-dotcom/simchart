"""未実装フラグの扱い。

**暗黙の no-op が最悪の事故**なので、``True`` を渡したら必ず停止し、しかも
「どの段階で実装されるか」がメッセージから判ること。
"""

from __future__ import annotations

import pytest

from simchart.config import IMPLEMENTED_FLAGS, STAGES, UNIMPLEMENTED_FLAGS, Config
from simchart.layers import (
    build_activity,
    build_book_layer,
    build_calendar,
    build_price_layer,
)
from simchart.rng import RNGRegistry


@pytest.mark.parametrize("flag,stage", [(k, v[0]) for k, v in UNIMPLEMENTED_FLAGS.items()])
def test_unimplemented_flag_raises_with_stage_name(flag: str, stage: str) -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        Config(**{flag: True})
    message = str(excinfo.value)
    assert stage in message, message
    assert flag in message, message


def test_every_flag_is_registered() -> None:
    """bool フラグの追加漏れを防ぐ。

    新しい ``enable_*`` を Config に足したのに ``UNIMPLEMENTED_FLAGS`` へ登録し
    忘れると、そのフラグは黙って無視される。それを構造的に防ぐ。
    """
    import dataclasses

    # `from __future__ import annotations` により f.type は文字列になる。
    bool_flags = {f.name for f in dataclasses.fields(Config) if f.type in (bool, "bool")}
    assert bool_flags, "bool フラグが 1 つも検出できていません (型注釈の書式を確認)"
    unregistered = bool_flags - set(UNIMPLEMENTED_FLAGS) - set(IMPLEMENTED_FLAGS)
    assert not unregistered, (
        f"どの台帳にも載っていないフラグ: {sorted(unregistered)} "
        f"(UNIMPLEMENTED_FLAGS か IMPLEMENTED_FLAGS に登録すること)"
    )
    overlap = set(UNIMPLEMENTED_FLAGS) & set(IMPLEMENTED_FLAGS)
    assert not overlap, f"両方の台帳に載っているフラグ: {sorted(overlap)}"


def test_kappa_raises_for_s10() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        Config(kappa=0.3)
    assert "S10" in str(excinfo.value)


def test_feedback_gain_raises_for_s11() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        Config(feedback_gain=0.1)
    assert "S11" in str(excinfo.value)


def test_multi_asset_raises_for_s13() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        Config(n_assets=3)
    assert "S13" in str(excinfo.value)


def test_unimplemented_stage_raises() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        Config(stage="S3")
    assert "S3" in str(excinfo.value)


def test_unknown_stage_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        Config(stage="S99")


def test_all_stage_names_are_declared() -> None:
    assert STAGES[0] == "S0" and STAGES[-1] == "S13" and len(STAGES) == 14


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        Config.from_dict({"seed": 1, "enable_teleportation": True})
    assert "enable_teleportation" in str(excinfo.value)


def test_defaults_are_all_off() -> None:
    config = Config()
    for flag in UNIMPLEMENTED_FLAGS:
        assert getattr(config, flag) is False
    assert config.kappa == 0.0
    assert config.n_assets == 1
    assert config.stage == "S0"


# ---------------------------------------------------------------------------
# 層のビルダー側の二重チェック。Config を経由せずに直接構築されても止まること。
# ---------------------------------------------------------------------------
def _force(config: Config, **changes: object) -> Config:
    """frozen dataclass の検証を迂回してフラグを立てる (テスト専用)。"""
    for key, value in changes.items():
        object.__setattr__(config, key, value)
    return config


@pytest.mark.parametrize(
    "flag,builder,stage",
    [
        ("enable_seasonality", "calendar", "S4"),
        ("enable_jump", "price", "S3"),
        ("enable_hawkes", "activity", "S7"),
        ("enable_book", "book", "S6"),
        ("enable_metaorder", "book", "S8"),
        ("enable_queue_reactive", "book", "S9"),
    ],
)
def test_layer_builders_reject_unimplemented_flags(flag: str, builder: str, stage: str) -> None:
    config = _force(Config(n_days=2, steps_per_day=234), **{flag: True})
    rng = RNGRegistry(config.seed)
    calendar = build_calendar(Config(n_days=2, steps_per_day=234), rng)
    activity = build_activity(Config(n_days=2, steps_per_day=234), rng, calendar)

    with pytest.raises(NotImplementedError) as excinfo:
        if builder == "calendar":
            build_calendar(config, rng)
        elif builder == "activity":
            build_activity(config, rng, calendar)
        elif builder == "price":
            build_price_layer(config, rng, calendar, activity)
        else:
            build_book_layer(config, rng, calendar, activity)
    assert stage in str(excinfo.value)


def test_l1_event_generation_refuses_instead_of_returning_empty() -> None:
    """イベント生成は空配列を返さず停止すること。

    「イベントが 0 件だった」という測定結果と「まだ実装していない」を
    取り違えないため。
    """
    config = Config(n_days=2, steps_per_day=234)
    rng = RNGRegistry(config.seed)
    calendar = build_calendar(config, rng)
    activity = build_activity(config, rng, calendar)
    with pytest.raises(NotImplementedError) as excinfo:
        activity.event_times(0.0, 100.0)
    assert "S6" in str(excinfo.value) or "S7" in str(excinfo.value)
