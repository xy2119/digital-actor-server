"""Tests for GameDrivenStage lifecycle and switching (TTS disabled)."""
from __future__ import annotations

import pytest

from metahuman_actor.game_driven.stage import GameDrivenStage

from .conftest import write_scenario_tree


@pytest.fixture(autouse=True)
def _dummy_llm_key(monkeypatch):
    # SingleSceneStage constructs a real LLM client which validates the API key
    # is present (no network call). Supply a dummy so tests don't need real keys.
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-dummy-key")


@pytest.fixture
def stage(local_prompts):
    # Two scenes, two interactions, so set_scene / set_interaction have targets.
    # scene_description renders as "<scene> desc." and steer as "<interaction> steer."
    write_scenario_tree(
        local_prompts,
        scenes=("scene_1", "scene_2"),
        interactions=("converse", "barter"),
    )
    return GameDrivenStage(llm_model="cerebras/qwen-3-235b-a22b-instruct-2507", tts_enabled=False)


@pytest.fixture
def multi_stage(local_prompts):
    write_scenario_tree(
        local_prompts,
        scenes=("scene_1", "scene_2"),
        interactions=("converse", "barter"),
        characters=(("dorn", "Dorn"), ("barkeep", "Barkeep")),
    )
    return GameDrivenStage(llm_model="cerebras/qwen-3-235b-a22b-instruct-2507", tts_enabled=False)


@pytest.mark.asyncio
async def test_load_multi_populates_all_characters(multi_stage):
    await multi_stage.load_scenario("tavern")
    assert multi_stage.scenario is not None
    assert set(multi_stage.character_ids()) == {"dorn", "barkeep"}
    assert multi_stage.active_character == "dorn"
    assert "Dorn is wary." == multi_stage.scene_data_for("dorn").character_back_story
    assert "Barkeep is wary." == multi_stage.scene_data_for("barkeep").character_back_story


@pytest.mark.asyncio
async def test_interactions_map_has_all_characters(multi_stage):
    await multi_stage.load_scenario("tavern")
    assert multi_stage.interactions_map() == {"dorn": "converse", "barkeep": "converse"}


@pytest.mark.asyncio
async def test_unload_clears_all_characters(multi_stage):
    await multi_stage.load_scenario("tavern")
    await multi_stage.unload_scenario()
    assert multi_stage.scenario is None
    assert multi_stage.character_ids() == []
    assert multi_stage.active_character is None
    assert multi_stage.scene_data is None


@pytest.mark.asyncio
async def test_active_scene_data_follows_active_pointer(multi_stage):
    await multi_stage.load_scenario("tavern")
    multi_stage._active_character = "barkeep"
    assert multi_stage.scene_data.character_back_story == "Barkeep is wary."
    multi_stage._active_character = "dorn"
    assert multi_stage.scene_data.character_back_story == "Dorn is wary."


@pytest.mark.asyncio
async def test_set_interaction_changes_only_one_character(multi_stage):
    await multi_stage.load_scenario("tavern")
    await multi_stage.set_interaction("dorn", "barter")
    assert multi_stage.interactions_map() == {"dorn": "barter", "barkeep": "converse"}


@pytest.mark.asyncio
async def test_set_scene_resets_all_interactions(multi_stage):
    await multi_stage.load_scenario("tavern")
    await multi_stage.set_interaction("dorn", "barter")
    await multi_stage.set_scene("scene_2")
    assert multi_stage.current_scene == "scene_2"
    assert multi_stage.interactions_map() == {"dorn": "converse", "barkeep": "converse"}


@pytest.mark.asyncio
async def test_starts_empty(stage):
    assert stage.scenario is None
    assert stage.scene_data is None


@pytest.mark.asyncio
async def test_load_scenario_populates(stage):
    await stage.load_scenario("tavern")
    assert stage.scenario is not None
    assert stage.scene_data is not None
    assert stage.current_scene == "scene_1"
    assert stage.interactions_map()["zeek"] == "converse"
    assert stage.character_ids() == ["zeek"]
    assert stage.active_character == "zeek"


@pytest.mark.asyncio
async def test_unload_returns_to_empty(stage):
    await stage.load_scenario("tavern")
    await stage.unload_scenario()
    assert stage.scenario is None
    assert stage.scene_data is None
    assert stage.character_ids() == []
    assert stage.active_character is None


@pytest.mark.asyncio
async def test_unload_when_empty_is_noop(stage):
    await stage.unload_scenario()  # must not raise
    assert stage.scenario is None


@pytest.mark.asyncio
async def test_set_interaction_swaps_and_preserves_history(stage):
    await stage.load_scenario("tavern")
    stage._characters["zeek"].actor.history.add_message("Player", "earlier message")
    await stage.set_interaction("zeek", "barter")
    assert stage.interactions_map()["zeek"] == "barter"
    assert "barter steer." == stage.scene_data.steer_back_instruction
    assert any(
        m.text == "earlier message"
        for m in stage._characters["zeek"].actor.history.messages
    )


@pytest.mark.asyncio
async def test_set_interaction_unknown_raises_and_keeps_state(stage):
    await stage.load_scenario("tavern")
    with pytest.raises(Exception):
        await stage.set_interaction("zeek", "nonexistent")
    assert stage.interactions_map()["zeek"] == "converse"


@pytest.mark.asyncio
async def test_set_scene_swaps_and_resets_interaction(stage):
    await stage.load_scenario("tavern")
    await stage.set_interaction("zeek", "barter")
    stage._characters["zeek"].actor.history.add_message("Player", "carry me")
    await stage.set_scene("scene_2")
    assert stage.current_scene == "scene_2"
    assert stage.interactions_map()["zeek"] == "converse"
    assert "scene_2 desc." == stage.scene_data.scene_description
    assert any(
        m.text == "carry me"
        for m in stage._characters["zeek"].actor.history.messages
    )


@pytest.mark.asyncio
async def test_set_scene_unknown_raises_and_keeps_state(stage):
    await stage.load_scenario("tavern")
    with pytest.raises(Exception):
        await stage.set_scene("scene_99")
    assert stage.current_scene == "scene_1"


@pytest.mark.asyncio
async def test_respond_routes_to_named_character(multi_stage):
    from unittest.mock import AsyncMock
    from digital_actor.dialogue import DialogueLine
    await multi_stage.load_scenario("tavern")

    multi_stage._characters["dorn"].scene.respond_with_hint = AsyncMock(
        return_value=(DialogueLine(name="Dorn", text="Arr.", line_id="D1"), None)
    )
    multi_stage._characters["barkeep"].scene.respond_with_hint = AsyncMock(
        return_value=(DialogueLine(name="Barkeep", text="Aye.", line_id="B1"), None)
    )

    resolved, hint = await multi_stage.respond("dorn", "hi", world_state={})
    assert resolved == "dorn"
    multi_stage._characters["dorn"].scene.respond_with_hint.assert_awaited_once()
    multi_stage._characters["barkeep"].scene.respond_with_hint.assert_not_awaited()
    assert multi_stage.active_character == "dorn"


@pytest.mark.asyncio
async def test_respond_casefold_resolves(multi_stage):
    from unittest.mock import AsyncMock
    from digital_actor.dialogue import DialogueLine
    await multi_stage.load_scenario("tavern")
    multi_stage._characters["dorn"].scene.respond_with_hint = AsyncMock(
        return_value=(DialogueLine(name="Dorn", text="Arr.", line_id="D1"), None)
    )
    resolved, _ = await multi_stage.respond("Dorn", "hi", world_state={})  # capitalized FName
    assert resolved == "dorn"
    multi_stage._characters["dorn"].scene.respond_with_hint.assert_awaited_once()


@pytest.mark.asyncio
async def test_respond_unknown_npc_raises(multi_stage):
    from metahuman_actor.game_driven.stage import UnknownNpcError
    await multi_stage.load_scenario("tavern")
    with pytest.raises(UnknownNpcError):
        await multi_stage.respond("nobody", "hi", world_state={})


@pytest.mark.asyncio
async def test_respond_omitted_npc_uses_primary(multi_stage):
    from unittest.mock import AsyncMock
    from digital_actor.dialogue import DialogueLine
    await multi_stage.load_scenario("tavern")
    multi_stage._characters["dorn"].scene.respond_with_hint = AsyncMock(
        return_value=(DialogueLine(name="Dorn", text="Arr.", line_id="D1"), None)
    )
    resolved, _ = await multi_stage.respond(None, "hi", world_state={})
    assert resolved == "dorn"  # primary = default_character = first = dorn


@pytest.mark.asyncio
async def test_trigger_routes_to_named_character(multi_stage):
    from unittest.mock import AsyncMock
    from digital_actor.dialogue import DialogueLine
    await multi_stage.load_scenario("tavern")
    multi_stage._characters["barkeep"].scene.trigger_with_hint = AsyncMock(
        return_value=(DialogueLine(name="Barkeep", text="Aye.", line_id="B1"), None)
    )
    resolved, hint = await multi_stage.trigger("barkeep", "greet", info={}, world_state={})
    assert resolved == "barkeep"
    multi_stage._characters["barkeep"].scene.trigger_with_hint.assert_awaited_once()
    assert multi_stage.active_character == "barkeep"


@pytest.mark.asyncio
async def test_respond_history_isolated_per_character(multi_stage):
    from digital_actor.dialogue import DialogueLine
    await multi_stage.load_scenario("tavern")

    def make_recorder(cid):
        async def _rwh(text, world_state, emotions=None, request_followup_hint=False, deliver_after=None):
            multi_stage._characters[cid].actor.history.add_message("Player", text)
            return DialogueLine(name=cid, text="ok", line_id=cid), None
        return _rwh

    multi_stage._characters["dorn"].scene.respond_with_hint = make_recorder("dorn")
    multi_stage._characters["barkeep"].scene.respond_with_hint = make_recorder("barkeep")

    await multi_stage.respond("dorn", "to dorn", world_state={})
    await multi_stage.respond("barkeep", "to barkeep", world_state={})

    dorn_texts = [m.text for m in multi_stage._characters["dorn"].actor.history.messages]
    barkeep_texts = [m.text for m in multi_stage._characters["barkeep"].actor.history.messages]
    assert "to dorn" in dorn_texts and "to barkeep" not in dorn_texts
    assert "to barkeep" in barkeep_texts and "to dorn" not in barkeep_texts


class _RecorderTTS:
    """TTS double that records every generate_audio call."""

    def __init__(self):
        self.calls: list[str] = []
        self.chunks_yielded = 0

    @property
    def sample_rate(self) -> int:
        return 24000

    async def generate_audio(self, text: str):
        self.calls.append(text)
        self.chunks_yielded += 1
        yield b"\x00\x00"


@pytest.mark.asyncio
async def test_warmup_character_drives_generate_audio(multi_stage):
    await multi_stage.load_scenario("tavern")
    recorder = _RecorderTTS()
    # multi_stage fixture builds with tts_enabled=False, so tts_client is None;
    # inject the recorder to assert generate_audio is actually driven.
    multi_stage._characters["dorn"].tts_client = recorder

    await multi_stage.warmup_character("dorn")

    assert recorder.calls == ["Warming up."]
    assert recorder.chunks_yielded == 1  # iterator was actually consumed


@pytest.mark.asyncio
async def test_warmup_character_noop_when_tts_client_is_none(multi_stage):
    await multi_stage.load_scenario("tavern")
    # Fixture builds with tts_enabled=False -> tts_client is already None.
    assert multi_stage._characters["dorn"].tts_client is None

    # Must not raise.
    await multi_stage.warmup_character("dorn")


@pytest.mark.asyncio
async def test_warmup_character_unknown_cid_raises(multi_stage):
    from metahuman_actor.game_driven.stage import UnknownNpcError

    await multi_stage.load_scenario("tavern")

    with pytest.raises(UnknownNpcError):
        await multi_stage.warmup_character("ghost")


@pytest.mark.asyncio
async def test_warmup_character_no_scenario_raises(multi_stage):
    from metahuman_actor.game_driven.stage import UnknownNpcError

    # No load_scenario called — _characters is empty.
    with pytest.raises(UnknownNpcError):
        await multi_stage.warmup_character("dorn")
