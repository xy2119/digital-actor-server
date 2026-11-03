"""LLM caption matching: player utterance -> best clip by facial caption.

Used by ReactionService when match_mode == "caption_llm". Function tags still
hard-route the candidate pool (same as context mode); the LLM then picks one
clip by comparing the player's words to each clip's facial caption. No
situation sentence and no usage-context embeddings are involved.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from app_logging import get_logger

from metahuman_actor.reaction.bridge import _extract_json
from metahuman_actor.reaction.retriever import RetrievedClip

logger = get_logger(__name__)

# Soft cap so the prompt stays within typical context budgets when a function
# bucket is large (e.g. empathy / continuer). Prefer high-appropriateness clips.
MAX_CANDIDATES = 48

CAPTION_SELECT_PROMPT = """\
You pick a nonverbal facial reaction for a silent, empathetic listener NPC.
The player just spoke. Choose the single best clip whose facial caption best
matches how the listener should look in response.

Player said: "{player_text}"

Candidate clips (stem | facial caption):
{candidates}

Rules:
- Pick only from the listed stems.
- Match the facial action to the player's words (e.g. surprise → widened eyes;
  empathy → concern; think → furrowed brow), not a spoken reply.
- Prefer a natural silent reaction over a dramatic mismatch.

Reply with ONLY a JSON object, no markdown:
{{"stem": "<exact stem from the list>", "reason": "<short why>"}}"""


@dataclass
class CaptionSelectResult:
    clip: RetrievedClip
    reason: str = ""


class CaptionSelector:
    """Trims large function buckets; anti-repeat comes from the caller."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def filter_excluded(self, stems: list[str], exclude: list[str]) -> list[str]:
        banned = set(exclude)
        filtered = [s for s in stems if s not in banned]
        return filtered or stems  # never empty the pool entirely

    def trim_candidates(
        self, metas: list[dict], limit: int = MAX_CANDIDATES
    ) -> list[dict]:
        """Prefer high appropriateness / head motion, then shuffle ties."""
        if len(metas) <= limit:
            return list(metas)
        ranked = sorted(
            metas,
            key=lambda m: (
                float(m.get("appropriateness") or 0.0),
                float(m.get("head_motion") or 0.0),
                self._rng.random(),
            ),
            reverse=True,
        )
        return ranked[:limit]


def parse_caption_select_reply(
    reply: str, candidates: list[dict]
) -> tuple[str, str]:
    """Return (stem, reason). Raises ValueError if stem is not in the pool."""
    data = _extract_json(reply)
    stem = str(data.get("stem") or "").strip()
    reason = str(data.get("reason") or "").strip()
    allowed = {c["stem"] for c in candidates}
    if stem not in allowed:
        raise ValueError(f"stem {stem!r} not in candidate pool ({len(allowed)})")
    return stem, reason


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for c in candidates:
        caption = (c.get("caption") or "").replace("\n", " ").strip()
        lines.append(f"- {c['stem']} | {caption}")
    return "\n".join(lines)


async def select_by_caption(
    llm_client,
    player_text: str,
    candidates: list[dict],
    *,
    exclude: list[str] | None = None,
    selector: CaptionSelector | None = None,
) -> CaptionSelectResult:
    """Ask the LLM to pick one stem from the caption list."""
    if not candidates:
        raise ValueError("caption select: empty candidate list")

    sel = selector or CaptionSelector()
    by_stem = {c["stem"]: c for c in candidates}
    stems = sel.filter_excluded(list(by_stem), exclude or [])
    pool = sel.trim_candidates([by_stem[s] for s in stems])

    prompt = CAPTION_SELECT_PROMPT.format(
        player_text=player_text.replace('"', "'"),
        candidates=_format_candidates(pool),
    )
    reply = await llm_client.acomplete(
        [{"role": "user", "content": prompt}], obs_name="reaction_caption_select"
    )

    try:
        stem, reason = parse_caption_select_reply(reply, pool)
    except (ValueError, json.JSONDecodeError) as exc:
        # Fallback: first pool entry (already appropriateness-ranked).
        stem = pool[0]["stem"]
        reason = f"parse fallback ({exc})"
        logger.warning("caption select parse failed (%s); using %s", exc, stem)

    meta = by_stem[stem]
    logger.info(
        "caption pick: %s (fn=%s) reason=%r from pool=%d",
        stem, meta.get("fn"), reason, len(pool),
    )
    return CaptionSelectResult(
        clip=RetrievedClip(
            stem=stem,
            score=1.0,
            caption=meta.get("caption") or "",
            dur=float(meta.get("dur") or 0.0),
            fn=meta.get("fn") or "",
            intensity=meta.get("intensity") or "medium",
        ),
        reason=reason,
    )
