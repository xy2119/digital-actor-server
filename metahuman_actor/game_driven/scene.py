"""GameDrivenScene — request-driven dialogue, no server-side clock.

Implements the library BaseScene but ignores tick/idle/followup-timer
behaviour. Lines are produced only when the game calls respond()/trigger().
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass

from app_logging import get_logger
from digital_actor.data_models import PromptInfo
from digital_actor.dialogue import NARRATOR_ROLE_NAME, PLAYER_ROLE_NAME, DialogueLine
from digital_actor.scene import BaseScene
from langfuse_utils import get_prompt
from metahuman_actor.actor import MetaHumanDigitalActor
from metahuman_actor.game_driven.scene_data import GameDrivenSceneData
from metahuman_actor.game_driven.world_state import render_world_state

logger = get_logger(__name__)


@dataclass(frozen=True)
class FollowupHint:
    line_id: str
    available: bool
    suggested_delay_seconds: float


class GameDrivenScene(BaseScene):
    def __init__(
        self,
        actor: MetaHumanDigitalActor,
        scene_data: GameDrivenSceneData,
        suggested_delay_seconds: float = 6.0,
    ) -> None:
        self.actor = actor
        self.scene_data = scene_data
        self.suggested_delay_seconds = suggested_delay_seconds
        self._response_lock = asyncio.Lock()

    # --- BaseScene abstract methods (mostly inert in this path) ---

    async def tick(self) -> None:
        return

    async def on_interrupt(self, line_id: str, elapsed_seconds: float) -> None:
        return

    async def on_game_event(self, name: str, event_info: dict[str, str]) -> None:
        # Event handling arrives via trigger(); BaseScene's hook is unused here.
        return

    async def on_user_input(self, message: str, emotions: list[str] | None = None) -> None:
        await self.respond(message, world_state={}, emotions=emotions)

    def reset(self) -> None:
        self.actor.reset()

    async def await_idle(self) -> None:
        """Block until any in-flight response pipeline has finished.

        Acquiring the response lock (held for the duration of respond/trigger)
        guarantees no response is mid-flight, so lifecycle code can safely swap
        scene_data or tear down.
        """
        async with self._response_lock:
            pass

    # --- request entry points ---

    async def respond(
        self,
        text: str,
        world_state: dict | None,
        emotions: list[str] | None = None,
    ) -> DialogueLine:
        """No-hint convenience wrapper. Use respond_with_hint to request a followup hint."""
        line, _ = await self.respond_with_hint(
            text, world_state, emotions=emotions, request_followup_hint=False
        )
        return line

    async def respond_with_hint(
        self,
        text: str,
        world_state: dict | None,
        emotions: list[str] | None = None,
        request_followup_hint: bool = False,
        deliver_after: Awaitable | None = None,
    ) -> tuple[DialogueLine, "FollowupHint | None"]:
        """deliver_after: if given, awaited after the LLM reply is generated and
        before it is delivered, so the caller can run other work (e.g. the
        listener's reaction clip) concurrently with generation but still send
        it first."""
        if not text or not text.strip():
            raise ValueError("respond: empty text")
        async with self._response_lock:
            self.actor.history.add_message(PLAYER_ROLE_NAME, text.strip())
            prompt_info = self._build_line_prompt(
                template="dialogue/get_respond_line",
                world_state=world_state,
                trigger_prompt=None,
            )
            line = await self._generate_and_deliver(
                prompt_info, emotions, deliver_after=deliver_after
            )
            hint = await self._maybe_followup_hint(line, request_followup_hint)
            await self.actor.history.summarize_if_needed()
            return line, hint

    async def trigger(
        self,
        name: str,
        info: dict[str, str],
        world_state: dict | None,
    ) -> DialogueLine:
        """No-hint convenience wrapper. Use trigger_with_hint to request a followup hint."""
        line, _ = await self.trigger_with_hint(name, info, world_state)
        return line

    async def trigger_with_hint(
        self,
        name: str,
        info: dict[str, str],
        world_state: dict | None,
        request_followup_hint: bool = False,
    ) -> tuple[DialogueLine, "FollowupHint | None"]:
        config = self.scene_data.triggers[name]  # KeyError -> caller emits error frame
        async with self._response_lock:
            narrator = config.render_narrator(info)
            if narrator is not None:
                self.actor.history.add_message(NARRATOR_ROLE_NAME, narrator)
            self._evaluate_event_checkpoints(name)
            prompt_info = self._build_line_prompt(
                template="dialogue/get_trigger_line",
                world_state=world_state,
                trigger_prompt=config.render_prompt(info),
            )
            line = await self._generate_and_deliver(prompt_info, emotions=None)
            hint = await self._maybe_followup_hint(line, request_followup_hint)
            await self.actor.history.summarize_if_needed()
            return line, hint

    # --- followup hint ---

    async def _maybe_followup_hint(
        self, line: DialogueLine, request_followup_hint: bool
    ) -> "FollowupHint | None":
        if not request_followup_hint:
            return None
        try:
            available = await self._query_followup()
        except Exception:
            logger.exception("followup query failed; emitting no hint")
            return None
        return FollowupHint(
            line_id=line.line_id,
            available=available,
            suggested_delay_seconds=self.suggested_delay_seconds,
        )

    async def _query_followup(self) -> bool:
        from digital_actor.stage_context import stage_context

        prompt_info = self._build_followup_prompt()
        response = await stage_context.llm_acomplete(prompt_info, obs_name="query_followup")
        return response.strip().lower() in ("yes", "true")

    def _build_followup_prompt(self) -> PromptInfo:
        history = self.actor.history
        last = history.last_actor_line()
        if last is None:
            raise RuntimeError("query_followup invoked before the actor has spoken")
        last_line, last_idx = last
        dialogue = "\n\n".join(
            f"{m.name}: {m.text}"
            for m in history.messages[history.summary_idx : last_idx + 1]
            if m.name != NARRATOR_ROLE_NAME
        )
        dialogue_summary_wrapper = ""
        if history.summary:
            dialogue_summary_wrapper = get_prompt("common/dialogue_summary_wrapper").compile(
                dialogue_summary=history.summary
            )
        prompt = get_prompt("query/query_followup")
        prompt_input = {
            "scene_description": self.scene_data.scene_description,
            "actors": ", ".join([self.actor.name, PLAYER_ROLE_NAME]),
            "dialogue": dialogue,
            "dialogue_summary_wrapper": dialogue_summary_wrapper,
            "last_line": f"{last_line.name}: {last_line.text}",
        }
        compiled = prompt.compile(**prompt_input)
        return PromptInfo(prompt=compiled, input_args=prompt_input, langfuse_prompt=prompt)

    def _evaluate_event_checkpoints(self, name: str) -> None:
        from digital_actor.checkpoints import EventCheckpoint
        from digital_actor.game_events import GameEvent
        from digital_actor.stage_context import stage_context

        checkpoints = self.scene_data.checkpoints
        if not checkpoints or not checkpoints.nodes:
            return
        for node in checkpoints.active_nodes():
            if isinstance(node, EventCheckpoint) and node.event_id == name:
                if node.narrator_message and "true" in node.narrator_message:
                    self.actor.history.add_message(
                        NARRATOR_ROLE_NAME, node.narrator_message["true"]
                    )
                checkpoints.complete(node.id)
                for callback in node.callbacks or []:
                    stage_context.deliver_event(GameEvent(name=callback, info={}))
                break

    # --- internals ---

    def _build_line_prompt(
        self,
        *,
        template: str,
        world_state: dict | None,
        trigger_prompt: str | None,
    ) -> PromptInfo:
        situation = render_world_state(world_state)
        current_situation_wrapper = (
            f"# Current situation\n{situation}\n\n" if situation else ""
        )
        dialogue_summary_wrapper = ""
        if self.actor.history.summary:
            dialogue_summary_wrapper = get_prompt("common/dialogue_summary_wrapper").compile(
                dialogue_summary=self.actor.history.summary
            )
        prompt_input: dict = {
            "scene_back_story": self.scene_data.scene_back_story,
            "character_back_story": self.scene_data.character_back_story,
            "scene_description": self.scene_data.scene_description,
            "steer_back_instructions": self.scene_data.steer_back_instruction,
            "current_situation_wrapper": current_situation_wrapper,
            "dialogue_summary_wrapper": dialogue_summary_wrapper,
            "actors": ", ".join([self.actor.name, PLAYER_ROLE_NAME]),
            "actor_name": self.actor.name,
            "dialogue": self.actor.history.to_string(include_summary=False),
        }
        if trigger_prompt is not None:
            prompt_input["trigger_prompt"] = trigger_prompt
        prompt = get_prompt(template)
        compiled = prompt.compile(**prompt_input)
        return PromptInfo(prompt=compiled, input_args=prompt_input, langfuse_prompt=prompt)

    async def _generate_and_deliver(
        self,
        prompt_info: PromptInfo,
        emotions: list[str] | None,
        deliver_after: Awaitable | None = None,
    ) -> DialogueLine:
        from digital_actor.stage_context import stage_context

        response = await stage_context.llm_acomplete(prompt_info, obs_name="generate_next_line")
        if deliver_after is not None:
            await deliver_after
        line = self.actor.history.add_message(self.actor.name, response)
        emotion = line.tags[0] if line.tags else None
        intensity = line.tags[1] if len(line.tags) > 1 else None
        stage_context.deliver_text(line, interruptible=True, emotion=emotion, intensity=intensity)
        await self.actor.run_tts(line)
        return line
