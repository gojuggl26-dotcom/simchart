"""未実装フラグの扱い。

**暗黙の no-op が最悪の事故**なので、``True`` を渡したら必ず停止し、しかも
「どの段階で実装されるか」がメッセージから判ること。
"""

from __future__ import annotations

import pytest

from simchart.config import IMPLEMENTED_FLAGS, STAGES, UNIMPLEMENTED_FLAGS, Config
from simchart.layers import build_activity, build_calendar
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


def test_kappa_requires_metaorder() -> None:
    # S10 で実装済み。ただし κ はメタオーダー符号に乗るので単独では立てられない。
    with pytest.raises(ValueError) as excinfo:
        Config(kappa=0.3)
    assert "enable_metaorder" in str(excinfo.value)


def test_jump_hawkes_raises_until_s11d() -> None:
    # S11d (任意) で要否判断 — 有効化は §7.2 の手順を踏むまで止まる。
    with pytest.raises(NotImplementedError) as excinfo:
        Config(enable_jump_hawkes=True)
    assert "S11" in str(excinfo.value)


def test_multi_asset_raises_for_s13() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        Config(n_assets=3)
    assert "S13" in str(excinfo.value)


def test_unimplemented_stage_raises() -> None:
    # 実装が進んだら「次の未実装段階」へ更新する (S12 の実装で S12 -> S13)。
    with pytest.raises(NotImplementedError) as excinfo:
        Config(stage="S13")
    assert "S13" in str(excinfo.value)


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
# 層ビルダー側の二重チェックは S9 で対象が尽きた (板・活動度系の未実装フラグは
# 全て実装済み。S11 feedback は pipeline、S12 chaos は Config 検証が防壁)。
# Config 側の防壁は test_unimplemented_flag_raises_with_stage_name が引き続き検査する。
# ---------------------------------------------------------------------------


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
