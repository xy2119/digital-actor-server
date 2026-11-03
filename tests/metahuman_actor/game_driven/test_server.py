"""Tests for GameDrivenServer message dispatch using a fake websocket."""
from __future__ import annotations

import json

import pytest

from metahuman_actor.game_driven.scene import FollowupHint
from metahuman_actor.game_driven.server import GameDrivenServer
from metahuman_actor.game_driven.stage import GameDrivenStage

from .conftest import write_scenario_tree


@pytest.fixture(autouse=True)
def _dummy_llm_key(monkeypatch):
    # Constructing GameDrivenStage builds an LLM client which checks for the key.
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")


class FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, raw: str):
        self.sent.append(json.loads(raw))


@pytest.fixture
def server(local_prompts):
    write_scenario_tree(local_prompts)
    stage = GameDrivenStage(
        llm_model="cerebras/qwen-3-235b-a22b-instruct-2507", tts_enabled=False
    )
    return GameDrivenServer(stage)


@pytest.mark.asyncio
async def test_list_scenarios_returns_names_and_null_active(server):
    ws = FakeWS()
    await server._handle_message({"type": "list_scenarios"}, ws)
    assert ws.sent[-1]["type"] == "scenarios"
    assert ws.sent[-1]["active"] is None
    assert "tavern" in ws.sent[-1]["names"]


@pytest.mark.asyncio
async def test_respond_before_load_errors(server):
    ws = FakeWS()
    await server._handle_message(
        {"type": "respond", "npc": "zeek", "text": "hi", "world_state": {}}, ws
    )
    assert ws.sent[-1]["type"] == "error"


@pytest.mark.asyncio
async def test_load_scenario_emits_scenario_loaded(server):
    ws = FakeWS()
    await server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    frame = ws.sent[-1]
    assert frame["type"] == "scenario_loaded"
    assert frame["name"] == "tavern"
    assert frame["scene"] == "scene_1"
    assert frame["interactions"] == {"zeek": "converse"}


@pytest.mark.asyncio
async def test_load_unknown_scenario_errors(server):
    ws = FakeWS()
    await server._handle_message({"type": "load_scenario", "name": "nope"}, ws)
    assert ws.sent[-1]["type"] == "error"


@pytest.mark.asyncio
async def test_respond_with_followup_emits_hint(server, monkeypatch):
    ws = FakeWS()
    await server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)

    async def fake_respond_with_hint(text, world_state, emotions=None, request_followup_hint=False, deliver_after=None):
        from digital_actor.dialogue import DialogueLine

        line = DialogueLine(name="Zeek", text="Hi.", line_id="L1")
        hint = FollowupHint(line_id="L1", available=True, suggested_delay_seconds=6.0) if request_followup_hint else None
        return line, hint

    server._stage._characters["zeek"].scene.respond_with_hint = fake_respond_with_hint  # type: ignore
    await server._handle_message(
        {"type": "respond", "npc": "zeek", "text": "hi", "world_state": {}, "request_followup_hint": True},
        ws,
    )
    hint_frames = [f for f in ws.sent if f["type"] == "followup_hint"]
    assert hint_frames
    assert hint_frames[-1]["line_id"] == "L1"
    assert hint_frames[-1]["available"] is True
    assert hint_frames[-1]["npc"] == "zeek"


@pytest.mark.asyncio
async def test_respond_unknown_npc_errors(server):
    ws = FakeWS()
    await server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    await server._handle_message(
        {"type": "respond", "npc": "grog", "text": "hi", "world_state": {}}, ws
    )
    assert ws.sent[-1]["type"] == "error"


@pytest.mark.asyncio
async def test_respond_npc_is_case_insensitive(server, monkeypatch):
    # Unreal stores npc ids as FName, which auto-capitalizes and is
    # case-insensitive, so the client may send "Zeek" for canonical "zeek".
    ws = FakeWS()
    await server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)

    async def fake_respond_with_hint(text, world_state, emotions=None, request_followup_hint=False, deliver_after=None):
        from digital_actor.dialogue import DialogueLine

        return DialogueLine(name="Zeek", text="Hi.", line_id="L1"), None

    server._stage._characters["zeek"].scene.respond_with_hint = fake_respond_with_hint  # type: ignore
    await server._handle_message(
        {"type": "respond", "npc": "Zeek", "text": "hi", "world_state": {}}, ws
    )
    assert ws.sent[-1]["type"] != "error"


@pytest.mark.asyncio
async def test_set_scene_emits_scene_changed(server):
    ws = FakeWS()
    await server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    # scene_2 does not exist in this minimal tree -> error
    await server._handle_message({"type": "set_scene", "scene": "scene_2"}, ws)
    assert ws.sent[-1]["type"] == "error"


@pytest.fixture
def multi_server(local_prompts):
    write_scenario_tree(
        local_prompts,
        characters=(("dorn", "Dorn"), ("barkeep", "Barkeep")),
    )
    stage = GameDrivenStage(
        llm_model="cerebras/qwen-3-235b-a22b-instruct-2507", tts_enabled=False
    )
    return GameDrivenServer(stage)


@pytest.mark.asyncio
async def test_scenario_loaded_lists_all_characters(multi_server):
    ws = FakeWS()
    await multi_server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    frame = ws.sent[-1]
    assert frame["type"] == "scenario_loaded"
    assert frame["interactions"] == {"dorn": "converse", "barkeep": "converse"}


@pytest.mark.asyncio
async def test_trigger_for_second_character_works(multi_server):
    from unittest.mock import AsyncMock
    from digital_actor.dialogue import DialogueLine

    ws = FakeWS()
    await multi_server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)

    multi_server._stage._characters["dorn"].scene.trigger_with_hint = AsyncMock(
        return_value=(DialogueLine(name="Dorn", text="Arr.", line_id="D1"),
                      FollowupHint(line_id="D1", available=True, suggested_delay_seconds=6.0))
    )
    # Unreal sends capitalized "Dorn"; canonical id is "dorn".
    await multi_server._handle_message(
        {"type": "trigger", "npc": "Dorn", "name": "greet", "info": {},
         "world_state": {}, "request_followup_hint": True},
        ws,
    )
    assert not any(f.get("type") == "error" for f in ws.sent)
    hints = [f for f in ws.sent if f["type"] == "followup_hint"]
    assert hints and hints[-1]["npc"] == "dorn" and hints[-1]["line_id"] == "D1"


@pytest.mark.asyncio
async def test_respond_for_second_character_no_error(multi_server):
    from unittest.mock import AsyncMock
    from digital_actor.dialogue import DialogueLine
    ws = FakeWS()
    await multi_server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    multi_server._stage._characters["barkeep"].scene.respond_with_hint = AsyncMock(
        return_value=(DialogueLine(name="Barkeep", text="Aye.", line_id="B1"), None)
    )
    await multi_server._handle_message(
        {"type": "respond", "npc": "barkeep", "text": "hello", "world_state": {}}, ws
    )
    assert not any(f.get("type") == "error" for f in ws.sent)


@pytest.mark.asyncio
async def test_multi_respond_unknown_npc_errors(multi_server):
    ws = FakeWS()
    await multi_server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    await multi_server._handle_message(
        {"type": "respond", "npc": "ghost", "text": "hi", "world_state": {}}, ws
    )
    assert ws.sent[-1]["type"] == "error"


@pytest.mark.asyncio
async def test_warmup_tts_before_load_errors(server):
    ws = FakeWS()
    await server._handle_message({"type": "warmup_tts"}, ws)

    assert ws.sent[-1]["type"] == "error"
    assert not any(f["type"] == "tts_warmed_up" for f in ws.sent)
    assert not any(f["type"] == "tts_warmup_complete" for f in ws.sent)


@pytest.mark.asyncio
async def test_warmup_tts_emits_per_character_then_complete_single(server):
    ws = FakeWS()
    await server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    pre_count = len(ws.sent)

    await server._handle_message({"type": "warmup_tts"}, ws)

    new_frames = ws.sent[pre_count:]
    warmed = [f for f in new_frames if f["type"] == "tts_warmed_up"]
    completes = [f for f in new_frames if f["type"] == "tts_warmup_complete"]

    # Single-character `tavern` fixture has one character.
    assert [f["npc"] for f in warmed] == ["zeek"]
    assert len(completes) == 1
    # tts_warmup_complete is the last warmup-related frame.
    assert new_frames[-1]["type"] == "tts_warmup_complete"


@pytest.mark.asyncio
async def test_warmup_tts_emits_per_character_then_complete_multi(multi_server):
    ws = FakeWS()
    await multi_server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    pre_count = len(ws.sent)

    await multi_server._handle_message({"type": "warmup_tts"}, ws)

    new_frames = ws.sent[pre_count:]
    warmed = [f for f in new_frames if f["type"] == "tts_warmed_up"]
    completes = [f for f in new_frames if f["type"] == "tts_warmup_complete"]

    # `multi_server` fixture loads dorn + barkeep in that insertion order.
    assert [f["npc"] for f in warmed] == ["dorn", "barkeep"]
    assert len(completes) == 1
    assert new_frames[-1]["type"] == "tts_warmup_complete"


@pytest.mark.asyncio
async def test_warmup_tts_drives_generate_audio_on_each_client(multi_server):
    """When TTS clients are present, warmup actually calls generate_audio."""

    class _RecorderTTS:
        def __init__(self):
            self.calls: list[str] = []

        @property
        def sample_rate(self) -> int:
            return 24000

        async def generate_audio(self, text: str):
            self.calls.append(text)
            yield b"\x00\x00"

    ws = FakeWS()
    await multi_server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)

    dorn_rec = _RecorderTTS()
    barkeep_rec = _RecorderTTS()
    multi_server._stage._characters["dorn"].tts_client = dorn_rec
    multi_server._stage._characters["barkeep"].tts_client = barkeep_rec

    await multi_server._handle_message({"type": "warmup_tts"}, ws)

    assert dorn_rec.calls == ["Warming up."]
    assert barkeep_rec.calls == ["Warming up."]


@pytest.mark.asyncio
async def test_warmup_tts_generate_audio_raises_sends_error_no_complete(multi_server):
    """If generate_audio raises mid-loop, server emits error and skips tts_warmup_complete."""

    class _RaisingTTS:
        @property
        def sample_rate(self) -> int:
            return 24000

        async def generate_audio(self, text: str):
            raise RuntimeError("GPU OOM")
            yield  # makes this an async-generator function (parser-level)

    ws = FakeWS()
    await multi_server._handle_message({"type": "load_scenario", "name": "tavern"}, ws)
    multi_server._stage._characters["dorn"].tts_client = _RaisingTTS()

    await multi_server._handle_message({"type": "warmup_tts"}, ws)

    types = [f["type"] for f in ws.sent]
    assert "error" in types
    assert "tts_warmup_complete" not in types
