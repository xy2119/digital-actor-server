"""GameDrivenStage — owns scenario lifecycle for the request-driven server.

Holds one LoadedCharacter per scenario character (each with its own actor,
history, scene, tts client, and current interaction) plus an active-character
pointer. stage_context.scene_data / .tts_client return the ACTIVE character's
values; each request sets the active pointer to the addressed character before
generating. Safe because the WS loop is sequential and each scene serializes its
own generation — only one character generates at a time.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable
from dataclasses import dataclass

from app_logging import get_logger
from digital_actor.game_events import GameEventBase
from digital_actor.messenger import Messenger, MessengerType
from digital_actor.stage import SingleSceneStage
from tts_lib import TTSClient, get_tts_client

from metahuman_actor.actor import MetaHumanDigitalActor
from metahuman_actor.game_driven.scenario import GameDrivenScenario
from metahuman_actor.game_driven.scene import FollowupHint, GameDrivenScene
from metahuman_actor.game_driven.scene_data import GameDrivenSceneData

logger = get_logger(__name__)


class UnknownSceneError(ValueError):
    pass


class UnknownInteractionError(ValueError):
    pass


class UnknownNpcError(ValueError):
    pass


@dataclass
class LoadedCharacter:
    """Per-character runtime state held by the stage."""

    actor: MetaHumanDigitalActor
    scene: GameDrivenScene
    tts_client: TTSClient | None
    current_interaction: str


class GameDrivenStage(SingleSceneStage):
    def __init__(
        self,
        llm_model: str,
        messenger: Messenger | MessengerType | None = None,
        tts_enabled: bool = True,
    ) -> None:
        super().__init__(
            llm_model,
            tts_provider=None,
            tts_voice_id=None,
            tts_model_id=None,
            messenger=messenger,
        )
        self._scenario: GameDrivenScenario | None = None
        self._characters: dict[str, LoadedCharacter] = {}
        self._active_character: str | None = None
        self._primary: str | None = None
        self.current_scene: str | None = None
        self._tts_enabled = tts_enabled
        logger.info("GameDrivenStage ready (no scenario loaded)")

    @property
    def scenario(self) -> GameDrivenScenario | None:
        return self._scenario

    @property
    def active_character(self) -> str | None:
        return self._active_character

    def character_ids(self) -> list[str]:
        return list(self._characters)

    def interactions_map(self) -> dict[str, str]:
        return {cid: lc.current_interaction for cid, lc in self._characters.items()}

    def scene_data_for(self, cid: str) -> GameDrivenSceneData | None:
        lc = self._characters.get(cid)
        return lc.scene.scene_data if lc is not None else None

    @property
    def scene_data(self) -> GameDrivenSceneData | None:
        if self._active_character is None:
            return None
        lc = self._characters.get(self._active_character)
        return lc.scene.scene_data if lc is not None else None

    @property
    def tts_client(self) -> TTSClient | None:
        if self._active_character is None:
            return None
        lc = self._characters.get(self._active_character)
        return lc.tts_client if lc is not None else None

    async def on_game_event(self, event: GameEventBase) -> None:
        return

    async def on_user_input(self, message: str) -> None:
        cid = self._active_character or self._primary
        if cid is not None:
            await self._characters[cid].scene.on_user_input(message)

    async def await_idle(self) -> None:
        for lc in self._characters.values():
            await lc.scene.await_idle()

    def reset(self) -> None:
        for lc in self._characters.values():
            lc.scene.reset()

    def _resolve_npc(self, npc: str | None) -> str:
        if not npc:
            if self._primary is None:
                raise UnknownNpcError("no scenario loaded")
            return self._primary
        for cid in self._characters:
            if cid.casefold() == npc.casefold():
                return cid
        raise UnknownNpcError(npc)

    def _build_character(
        self, scenario: GameDrivenScenario, cid: str, scene: str, interaction: str
    ) -> LoadedCharacter:
        persona_path = scenario.persona_path(cid)
        with open(persona_path, encoding="utf-8") as f:
            persona = json.load(f)
        voice = (persona.get("voice") or {}) if self._tts_enabled else {}
        actor = MetaHumanDigitalActor(persona)
        scene_data = GameDrivenSceneData.load(
            scenario, scene=scene, character=cid, interaction=interaction
        )
        if voice.get("provider"):
            logger.info(
                "loading tts for %s (provider=%s model=%s)",
                cid,
                voice.get("provider"),
                voice.get("model_id"),
            )
            tts = get_tts_client(
                voice.get("provider"),
                voice_id=voice.get("voice_id"),
                model_id=voice.get("model_id"),
            )
            logger.info("tts loaded for %s", cid)
        else:
            tts = None
        return LoadedCharacter(
            actor=actor,
            scene=GameDrivenScene(actor=actor, scene_data=scene_data),
            tts_client=tts,
            current_interaction=interaction,
        )

    async def load_scenario(self, name: str) -> None:
        new_scenario = GameDrivenScenario.load(name)
        scene = new_scenario.default_scene
        interaction = new_scenario.default_interaction
        new_characters: dict[str, LoadedCharacter] = {}
        for cid in new_scenario.characters:
            new_characters[cid] = self._build_character(
                new_scenario, cid, scene, interaction
            )

        if self._scenario is not None:
            await self.await_idle()
        self.reset()
        self._scenario = new_scenario
        self._characters = new_characters
        self._primary = new_scenario.default_character
        self._active_character = new_scenario.default_character
        self.current_scene = scene
        logger.info(
            "Loaded game-driven scenario=%s characters=%s",
            new_scenario.name,
            list(new_characters),
        )

    async def unload_scenario(self) -> None:
        if self._scenario is None:
            return
        await self.await_idle()
        self.reset()
        self._characters = {}
        self._active_character = None
        self._primary = None
        self._scenario = None
        self.current_scene = None
        logger.info("Unloaded game-driven scenario")

    async def set_scene(self, scene: str) -> None:
        if self._scenario is None:
            raise UnknownSceneError("no scenario loaded")
        if not self._scenario.has_scene(scene):
            raise UnknownSceneError(scene)
        interaction = self._scenario.default_interaction
        new_data: dict[str, GameDrivenSceneData] = {}
        for cid in self._characters:
            if not self._scenario.has_interaction(scene, cid, interaction):
                raise UnknownInteractionError(f"{scene}/{cid}/{interaction}")
            new_data[cid] = GameDrivenSceneData.load(
                self._scenario, scene=scene, character=cid, interaction=interaction
            )
        await self.await_idle()
        for cid, lc in self._characters.items():
            lc.scene.scene_data = new_data[cid]
            lc.current_interaction = interaction
        self.current_scene = scene
        logger.info("Scene -> %s (all interactions reset to %s)", scene, interaction)

    async def set_interaction(self, npc: str, interaction: str) -> str:
        """Switch one character's interaction. Returns the resolved character id
        (uniform with respond/trigger, so callers don't re-resolve for echoes)."""
        if self._scenario is None:
            raise UnknownNpcError("no scenario loaded")
        cid = self._resolve_npc(npc)
        if not self._scenario.has_interaction(self.current_scene, cid, interaction):
            raise UnknownInteractionError(interaction)
        new_scene_data = GameDrivenSceneData.load(
            self._scenario,
            scene=self.current_scene,
            character=cid,
            interaction=interaction,
        )
        await self._characters[cid].scene.await_idle()
        self._characters[cid].scene.scene_data = new_scene_data
        self._characters[cid].current_interaction = interaction
        logger.info("Interaction[%s] -> %s", cid, interaction)
        return cid

    async def warmup_character(self, cid: str) -> None:
        """Drive a full TTS generate-pass for one character to prime the model.

        Discards the PCM bytes. No-op if the character has no TTS client.

        Raises:
            UnknownNpcError: if cid isn't in the loaded character set.
        """
        if cid not in self._characters:
            raise UnknownNpcError(cid)
        lc = self._characters[cid]
        if lc.tts_client is None:
            return
        async for _ in lc.tts_client.generate_audio("Warming up."):
            pass

    # --- routing ---

    def resolve_display_name(self, npc: str | None) -> str:
        """Resolve an inbound npc id to the actor's display name (wire actor_name)."""
        cid = self._resolve_npc(npc)
        return self._characters[cid].actor.name

    async def respond(
        self,
        npc: str | None,
        text: str,
        world_state: dict | None,
        emotions: list[str] | None = None,
        request_followup_hint: bool = False,
        deliver_after: Awaitable | None = None,
    ) -> tuple[str, FollowupHint | None]:
        cid = self._resolve_npc(npc)
        self._active_character = cid
        _, hint = await self._characters[cid].scene.respond_with_hint(
            text,
            world_state,
            emotions=emotions,
            request_followup_hint=request_followup_hint,
            deliver_after=deliver_after,
        )
        return cid, hint

    async def trigger(
        self,
        npc: str | None,
        name: str,
        info: dict[str, str],
        world_state: dict | None,
        request_followup_hint: bool = False,
    ) -> tuple[str, FollowupHint | None]:
        cid = self._resolve_npc(npc)
        self._active_character = cid
        _, hint = await self._characters[cid].scene.trigger_with_hint(
            name,
            info,
            world_state,
            request_followup_hint=request_followup_hint,
        )
        return cid, hint
