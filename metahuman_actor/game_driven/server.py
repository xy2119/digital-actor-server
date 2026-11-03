"""WebSocket server for the game-driven dialogue path.

Implements the request-driven wire protocol (respond/trigger/set_scene/
set_interaction). Dialogue text/audio frames flow through the messenger's
outbound drain (inherited from WebSocketServer); the new control/hint frames
are sent directly over the socket here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from app_logging import get_logger, setup_logging
from digital_actor.messenger import MessengerType, WebSocketServer
from dotenv import load_dotenv
from langfuse_utils import fetch_all_prompts_from_project, langfuse_session

from metahuman_actor.game_driven.scenario import list_game_driven_scenarios
from metahuman_actor.game_driven.stage import GameDrivenStage
from metahuman_actor.settings import settings

logger = get_logger(__name__)
get_logger("digital_actor")


class GameDrivenServer(WebSocketServer):
    def __init__(
        self,
        stage: GameDrivenStage,
        *,
        port: int = 8788,
        tick_rate: int = 20,
        listener_mode: bool = False,
        listener_speech: bool = False,
        reaction_match_mode: str = "context",
    ) -> None:
        super().__init__(stage, port=port, tick_rate=tick_rate)
        self._listener_mode = listener_mode
        # Speak *and* react: the reaction clip carries expression and head while
        # TTS + Audio2Face drive the mouth. Off = the original silent listener.
        self._listener_speech = listener_speech
        self._reaction_match_mode = reaction_match_mode
        self._reactions = None
        if listener_mode:
            self._ensure_reactions()

    def _ensure_reactions(self):
        """Create + warm the reaction service (blocking model/index load)."""
        if self._reactions is None:
            from metahuman_actor.reaction.service import ReactionService

            self._reactions = ReactionService(match_mode=self._reaction_match_mode)
            self._reactions.warmup()
        else:
            self._reactions.set_match_mode(self._reaction_match_mode)
        return self._reactions

    async def _handle_connection(self, ws) -> None:
        self._runtime.resume()
        # The game connects well before the player's first line (it loads the
        # scenario and warms TTS at BeginPlay), so this overlaps that idle time.
        self._prewarm_task = asyncio.create_task(self._prewarm_llm())
        try:
            await super()._handle_connection(ws)
        finally:
            self._runtime.pause()
            await self._stage.unload_scenario()
            logger.info("client disconnected; scenario unloaded")

    async def _handle_inbound(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "message": "invalid JSON"}))
                continue
            await self._handle_message(msg, ws)

    async def _handle_message(self, msg: dict, ws) -> None:
        msg_type = msg.get("type")
        stage: GameDrivenStage = self._stage
        logger.info("<<< %s", msg_type)
        try:
            if msg_type == "list_scenarios":
                active = stage.scenario.name if stage.scenario else None
                await ws.send(
                    json.dumps(
                        {
                            "type": "scenarios",
                            "names": list_game_driven_scenarios(),
                            "active": active,
                        }
                    )
                )
                return

            if msg_type == "load_scenario":
                name = (msg.get("name") or "").strip()
                if not name:
                    await ws.send(
                        json.dumps(
                            {"type": "error", "message": "load_scenario: empty name"}
                        )
                    )
                    return
                await stage.load_scenario(name)
                await ws.send(
                    json.dumps(
                        {
                            "type": "scenario_loaded",
                            "name": name,
                            "scene": stage.current_scene,
                            "interactions": stage.interactions_map(),
                        }
                    )
                )
                return

            if msg_type == "unload_scenario":
                await stage.unload_scenario()
                await ws.send(json.dumps({"type": "scenario_unloaded"}))
                return

            # Everything below requires a loaded scenario.
            if stage.scenario is None:
                await ws.send(
                    json.dumps({"type": "error", "message": "no scenario loaded"})
                )
                return

            if msg_type == "set_scene":
                scene = (msg.get("scene") or "").strip()
                await stage.set_scene(scene)
                await ws.send(
                    json.dumps(
                        {
                            "type": "scene_changed",
                            "scene": stage.current_scene,
                            "interactions": stage.interactions_map(),
                        }
                    )
                )
                return

            if msg_type == "set_interaction":
                npc = (msg.get("npc") or "").strip()
                interaction = (msg.get("interaction") or "").strip()
                resolved = await stage.set_interaction(npc, interaction)
                await ws.send(
                    json.dumps(
                        {
                            "type": "interaction_changed",
                            "npc": resolved,
                            "interaction": stage.interactions_map()[resolved],
                        }
                    )
                )
                return

            if msg_type == "set_listener_mode":
                enabled = bool(msg.get("enabled", True))
                if enabled:
                    await asyncio.to_thread(self._ensure_reactions)
                self._listener_mode = enabled
                logger.info("listener mode -> %s", enabled)
                await ws.send(
                    json.dumps({"type": "listener_mode", "enabled": enabled})
                )
                return

            if msg_type == "set_listener_speech":
                # Speech belongs to the listener *setting*, not to the process:
                # the ARKit comparison is animation-only while the CTRL one
                # speaks, and switching between them must not mean a restart.
                enabled = bool(msg.get("enabled", True))
                self._listener_speech = enabled
                logger.info("listener speech -> %s", enabled)
                await ws.send(
                    json.dumps({"type": "listener_speech", "enabled": enabled})
                )
                return

            if msg_type == "set_reaction_match_mode":
                from metahuman_actor.reaction.service import (
                    MATCH_MODES,
                    normalize_match_mode,
                )

                raw = msg.get("mode")
                try:
                    mode = normalize_match_mode(raw)
                except ValueError:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": (
                                    f"set_reaction_match_mode: bad mode {raw!r}; "
                                    f"expected one of {list(MATCH_MODES)}"
                                ),
                            }
                        )
                    )
                    return
                self._reaction_match_mode = mode
                # Warm reactions if needed so mode sticks even before --listener.
                reactions = await asyncio.to_thread(self._ensure_reactions)
                applied = reactions.set_match_mode(mode)
                await ws.send(
                    json.dumps({"type": "reaction_match_mode", "mode": applied})
                )
                return

            if msg_type == "warmup_tts":
                cids = stage.character_ids()
                for cid in cids:
                    logger.info("warming up tts for %s", cid)
                    await stage.warmup_character(cid)
                    logger.info("tts warmed: %s", cid)
                    await ws.send(json.dumps({"type": "tts_warmed_up", "npc": cid}))
                logger.info("tts warmup complete (%d characters)", len(cids))
                await ws.send(json.dumps({"type": "tts_warmup_complete"}))
                return

            if msg_type == "respond":
                npc = msg.get("npc")
                text = (msg.get("text") or "").strip()
                if not text:
                    await ws.send(
                        json.dumps({"type": "error", "message": "respond: empty text"})
                    )
                    return
                if self._listener_mode:
                    actor_name = stage.resolve_display_name(npc)
                    if text.startswith("/clip "):
                        # Dev shortcut: play an exact clip, bypass bridge+retrieval.
                        from metahuman_actor.reaction.clips import (
                            build_reaction_frame,
                            iter_reaction_frames,
                        )

                        stem = text.split(None, 1)[1].strip()
                        frame = await asyncio.to_thread(
                            build_reaction_frame, stem, actor_name, stem, "debug"
                        )
                        logger.info(">>> debug clip %s for %s", stem, actor_name)
                        for part in iter_reaction_frames(frame):
                            await ws.send(json.dumps(part))
                        return
                    t0 = time.perf_counter()
                    react_task = asyncio.create_task(
                        self._send_reaction(ws, stage, text, actor_name, t0)
                    )
                    if not self._listener_speech:
                        await react_task
                        return
                    # Speaking listener: the clip is no longer the whole answer,
                    # it is the performance the line is delivered over. The
                    # clip pick (an LLM call) and the dialogue LLM call run
                    # concurrently, but the line is only delivered once the
                    # reaction has been sent — the face should be reacting
                    # before the voice starts, which is also the order a person
                    # does it in. Client-side the clip keeps expression, brows,
                    # eyes and head for the whole line; only the mouth goes to
                    # Audio2Face.
                    reaction_gate = asyncio.ensure_future(
                        self._await_reaction(react_task)
                    )
                    logger.info("listener-speech: generating a spoken reply")
                    # falls through to the dialogue path below
                else:
                    t0 = time.perf_counter()
                    reaction_gate = None
                world_state = msg.get("world_state") or {}
                request_followup = bool(msg.get("request_followup_hint", False))
                emotions = msg.get("emotions")
                try:
                    resolved, hint = await stage.respond(
                        npc,
                        text,
                        world_state,
                        emotions=emotions,
                        request_followup_hint=request_followup,
                        deliver_after=reaction_gate,
                    )
                finally:
                    if reaction_gate is not None:
                        await reaction_gate
                logger.info(
                    "respond done in %.0f ms (line + tts delivered)",
                    (time.perf_counter() - t0) * 1000,
                )
                await self._maybe_send_hint(ws, resolved, hint)
                return

            if msg_type == "trigger":
                npc = msg.get("npc")
                name = (msg.get("name") or "").strip()
                info = {str(k): str(v) for k, v in (msg.get("info") or {}).items()}
                world_state = msg.get("world_state") or {}
                request_followup = bool(msg.get("request_followup_hint", False))
                resolved, hint = await stage.trigger(
                    npc,
                    name,
                    info,
                    world_state,
                    request_followup_hint=request_followup,
                )
                await self._maybe_send_hint(ws, resolved, hint)
                return

            await ws.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"unknown message type {msg_type!r}",
                    }
                )
            )
        except Exception as exc:
            logger.exception("error handling %s", msg_type)
            await ws.send(json.dumps({"type": "error", "message": str(exc)}))

    async def _send_reaction(self, ws, stage, text: str, actor_name: str, t0: float):
        frame = await self._ensure_reactions().react(stage.llm_client, text, actor_name)
        logger.info(
            ">>> reaction %s (%s, mode=%s, %d frames) for %s, picked in %.0f ms",
            frame["clip_id"], frame["function"],
            frame.get("match_mode"), frame["num_frames"], actor_name,
            (time.perf_counter() - t0) * 1000,
        )
        from metahuman_actor.reaction.clips import iter_reaction_frames

        # Long takes go out as a header plus reaction_chunk frames so no single
        # frame trips the client's receive cap.
        for part in iter_reaction_frames(frame):
            await ws.send(json.dumps(part))
        return frame

    @staticmethod
    async def _await_reaction(task: asyncio.Task) -> None:
        """Gate for the spoken line: wait for the reaction, but never let a
        failed reaction swallow the line."""
        try:
            await task
        except Exception:
            logger.exception("reaction failed; delivering the line without it")

    async def _prewarm_llm(self) -> None:
        """Open the LLM HTTPS connection before the first request needs it.

        The client is built at startup but its TCP/TLS connection is only
        opened by the first call, which otherwise lands on the player's first
        line. A model-list GET is enough to put a live connection in the pool.
        """
        client = getattr(self._stage.llm_client, "async_client", None)
        if client is None:
            return
        t0 = time.perf_counter()
        try:
            await client.models.list()
            logger.info("llm connection prewarmed in %.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.warning("llm prewarm failed (%s); first request will connect", exc)

    async def _maybe_send_hint(self, ws, npc: str, hint) -> None:
        if hint is None:
            return
        await ws.send(
            json.dumps(
                {
                    "type": "followup_hint",
                    "npc": npc,
                    "line_id": hint.line_id,
                    "available": hint.available,
                    "suggested_delay_seconds": hint.suggested_delay_seconds,
                }
            )
        )


def _apply_reasoning_effort(stage: GameDrivenStage, llm_model: str, effort: str) -> None:
    """gpt-oss reasons before its first visible token, and every reply here
    (clip pick, line, follow-up) waits on that. Short in-character lines don't
    need it, so cap it. Other models reject the parameter, so only set it for
    reasoning models."""
    if not effort or effort == "default" or "gpt-oss" not in llm_model:
        return
    kwargs = getattr(stage.llm_client, "extra_create_kwargs", None)
    if kwargs is not None:
        kwargs["reasoning_effort"] = effort
        logger.info("llm reasoning_effort -> %s", effort)


def main(
    port: int,
    llm_model: str,
    langfuse_local: bool = False,
    listener: bool = False,
    listener_speech: bool = False,
    reaction_match_mode: str = "context",
    reasoning_effort: str = "low",
) -> None:
    session = langfuse_session(
        prompt_label=settings.digital_actor_server.prompt_label,
        local=langfuse_local,
    )
    with session:
        fetch_all_prompts_from_project()
        stage = GameDrivenStage(llm_model, messenger=MessengerType.WEBSOCKET)
        _apply_reasoning_effort(stage, llm_model, reasoning_effort)
        GameDrivenServer(
            stage,
            port=port,
            listener_mode=listener,
            listener_speech=listener_speech,
            reaction_match_mode=reaction_match_mode,
        ).run()


if __name__ == "__main__":
    load_dotenv()
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="cerebras/qwen-3-235b-a22b-instruct-2507")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--langfuse-local", action="store_true")
    parser.add_argument(
        "--listener",
        action="store_true",
        help="Empathetic-listener mode: respond returns a facial reaction clip "
        "instead of dialogue (no TTS).",
    )
    parser.add_argument(
        "--reaction-match",
        choices=["context", "caption_llm", "caption_ctrl"],
        default="context",
        help="Clip match strategy in --listener mode: "
        "'context' = situation↔usage-contexts MaxSim (default); "
        "'caption_llm' = LLM picks by facial caption after function filter; "
        "'caption_ctrl' = LLM picks by video caption over the CTRL rig-control "
        "bank (high fidelity, no ARKit round-trip).",
    )
    parser.add_argument(
        "--listener-speech",
        action="store_true",
        help="In --listener mode, also generate an LLM reply and speak it "
        "(TTS -> Audio2Face) over the reaction clip. The clip keeps expression "
        "and head; only the mouth is driven by the audio.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "default"],
        default="low",
        help="reasoning_effort for gpt-oss models (ignored for others); "
        "'default' leaves the provider default.",
    )
    args = parser.parse_args()
    main(
        port=args.port,
        llm_model=args.llm,
        langfuse_local=args.langfuse_local,
        listener=args.listener,
        listener_speech=args.listener_speech,
        reaction_match_mode=args.reaction_match,
        reasoning_effort=args.reasoning_effort,
    )
