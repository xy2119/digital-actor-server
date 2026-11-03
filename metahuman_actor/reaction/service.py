"""Runtime facade: player text in, `reaction` wire frame out.

Three match modes (switchable for A/B eval):

  context      — LLM bridge → situation + function; MaxSim situation↔contexts
                 (default; needs reaction_index embeddings)
  caption_llm  — LLM bridge → function only; LLM picks best clip by caption
                 (no situation / usage-context similarity)
  caption_ctrl — same caption pick, but over the CTRL-curve bank: MetaHuman rig
                 controls straight from video (no ARKit projection), captioned
                 by video summary. No function routing — those clips carry no
                 taxonomy, so the whole bank is the pool. Frames go out tagged
                 `schema:"metahuman_ctrl_v1"`, which is what routes them past
                 the client's ARKit mapping pose into RigLogic.
"""

from __future__ import annotations

import asyncio
import time

from app_logging import get_logger

from metahuman_actor.reaction.bridge import bridge_player_input
from metahuman_actor.reaction.caption_select import CaptionSelector, select_by_caption
from metahuman_actor.reaction.clips import build_reaction_frame
from metahuman_actor.reaction.ctrl_bank import (
    build_ctrl_reaction_frame,
    load_ctrl_clips,
    preload_ctrl_wires,
)
from metahuman_actor.reaction.retriever import ReactionRetriever

logger = get_logger(__name__)

MATCH_MODES = ("context", "caption_llm", "caption_ctrl")
DEFAULT_MATCH_MODE = "context"

#: Modes served by the CTRL bank rather than the Express4D dataset.
CTRL_MATCH_MODES = ("caption_ctrl",)


def normalize_match_mode(mode: str | None) -> str:
    value = (mode or DEFAULT_MATCH_MODE).strip().lower()
    if value not in MATCH_MODES:
        raise ValueError(
            f"unknown reaction match mode {mode!r}; expected one of {MATCH_MODES}"
        )
    return value


class ReactionService:
    def __init__(self, match_mode: str = DEFAULT_MATCH_MODE) -> None:
        self._retriever = ReactionRetriever()
        self._caption_selector = CaptionSelector()
        self._retriever_warm = False
        self._match_mode = normalize_match_mode(match_mode)

    @property
    def match_mode(self) -> str:
        return self._match_mode

    def set_match_mode(self, mode: str) -> str:
        """Switch mode. Blocking when it first enters an Express4D mode, since
        that loads the index + embedding model — call it off the event loop."""
        self._match_mode = normalize_match_mode(mode)
        self._warm_for_mode()
        logger.info("reaction match mode -> %s", self._match_mode)
        return self._match_mode

    def warmup(self) -> None:
        """Blocking: load what the current mode picks from. Call once at boot.

        The CTRL bank is always scanned so a missing wire directory is a
        boot-time error instead of a first-utterance one. The Express4D index and
        embedder are loaded only for the modes that retrieve from them.
        """
        started = time.perf_counter()
        load_ctrl_clips()
        if self._match_mode in CTRL_MATCH_MODES:
            preload_ctrl_wires()
        self._warm_for_mode()
        logger.info("reaction service warm in %.1fs (mode=%s)",
                    time.perf_counter() - started, self._match_mode)

    def _warm_for_mode(self) -> None:
        if self._match_mode not in CTRL_MATCH_MODES and not self._retriever_warm:
            self._retriever.warmup()
            self._retriever_warm = True

    async def react(self, llm_client, player_text: str, actor_name: str) -> dict:
        """Match (context | caption_llm | caption_ctrl) -> clip frame."""
        if self._match_mode in CTRL_MATCH_MODES:
            # No function routing: the CTRL clips carry no taxonomy, so the
            # bridge's function guess has nothing to filter on and the whole
            # bank goes to the LLM.
            candidates = list(load_ctrl_clips())
            if not candidates:
                raise RuntimeError(
                    "caption_ctrl: CTRL bank is empty — check ICONIC_CTRL_WIRE_DIR"
                )
            result = await select_by_caption(
                llm_client,
                player_text,
                candidates,
                exclude=self._retriever.recent_stems(),
                selector=self._caption_selector,
            )
            chosen = result.clip
            self._retriever.note_played(chosen.stem)
            frame = await asyncio.to_thread(
                build_ctrl_reaction_frame,
                chosen.stem,
                actor_name,
                chosen.caption,
                chosen.fn,
            )
            frame["match_mode"] = self._match_mode
            return frame

        bridge = await bridge_player_input(llm_client, player_text)
        if self._match_mode == "caption_llm":
            candidates = await asyncio.to_thread(
                self._retriever.list_candidates, bridge.functions
            )
            result = await select_by_caption(
                llm_client,
                player_text,
                candidates,
                exclude=self._retriever.recent_stems(),
                selector=self._caption_selector,
            )
            chosen = result.clip
            self._retriever.note_played(chosen.stem)
        else:
            chosen = await asyncio.to_thread(
                self._retriever.retrieve,
                bridge.situation,
                bridge.functions,
                bridge.intensity,
                bridge.valence,
            )

        frame = await asyncio.to_thread(
            build_reaction_frame,
            chosen.stem,
            actor_name,
            chosen.caption,
            chosen.fn,
        )
        frame["match_mode"] = self._match_mode
        return frame
