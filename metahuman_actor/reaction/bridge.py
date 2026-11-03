"""Runtime LLM bridge: player utterance -> English situation + listener function.

The bridge moves the semantic gap to a controlled place: the query that enters
the vector space is always a short English "usage context" style sentence
produced by the LLM, matching the document space built offline by enrich.py.
Function tags are picked by the LLM from the fixed 8-function taxonomy and are
used as a hard routing filter, never embedded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app_logging import get_logger

from metahuman_actor.reaction.dataset import LISTENER_FUNCTIONS

logger = get_logger(__name__)

_INTENSITIES = ("slight", "medium", "strong")
_VALENCES = ("negative", "neutral", "positive")

BRIDGE_PROMPT = """\
You pick a nonverbal facial reaction for a silent, empathetic listener NPC.
The player just spoke to the NPC. Decide how the listener should react.

Player said: "{player_text}"

Listener reaction functions (pick the single best `function`, optionally one
`function2` if a second one genuinely fits):
- affirm: agree, approve, nod along, "yes exactly"
- continuer: neutral acknowledgement that keeps the speaker talking ("mm-hm")
- doubt: skepticism, disbelief, not convinced
- empathy: concern, sympathy, sharing the player's feeling (worry, sadness, warmth)
- surprise: startled, amazed, eyes widening at unexpected news
- think: pondering, working something out, concentrating on a hard question
- understand: sudden comprehension, "ah, I see"

Also write `situation`: ONE short English sentence of the form
"the player <what the player is doing/expressing>", written so it could match
a description of when this reaction is appropriate. No names, no quotes.

And `intensity`: how strong the reaction should be: slight | medium | strong.
And `valence`: the emotional tone the listener's face should carry:
negative (concern, worry, sadness) | neutral | positive (warm, amused, delighted).

Reply with ONLY a JSON object, no markdown:
{{"situation": "...", "function": "...", "function2": null, "intensity": "...", "valence": "..."}}"""


@dataclass
class BridgeResult:
    situation: str
    functions: list[str] = field(default_factory=list)
    intensity: str = "medium"
    valence: str = "neutral"


def _extract_json(text: str) -> dict:
    """Parse the first JSON object out of an LLM reply (tolerates fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in bridge reply: {text[:200]!r}")
    return json.loads(match.group(0))


def parse_bridge_reply(reply: str, player_text: str) -> BridgeResult:
    data = _extract_json(reply)
    functions = []
    for key in ("function", "function2"):
        fn = data.get(key)
        if isinstance(fn, str) and fn.strip().lower() in LISTENER_FUNCTIONS:
            functions.append(fn.strip().lower())
    situation = str(data.get("situation") or "").strip()
    if not situation:
        situation = f"the player says: {player_text}"
    intensity = str(data.get("intensity") or "medium").strip().lower()
    if intensity not in _INTENSITIES:
        intensity = "medium"
    valence = str(data.get("valence") or "neutral").strip().lower()
    if valence not in _VALENCES:
        valence = "neutral"
    return BridgeResult(
        situation=situation, functions=functions, intensity=intensity, valence=valence
    )


async def bridge_player_input(llm_client, player_text: str) -> BridgeResult:
    """Run the bridge LLM call. Falls back to a bare situation on parse errors."""
    prompt = BRIDGE_PROMPT.format(player_text=player_text.replace('"', "'"))
    reply = await llm_client.acomplete(
        [{"role": "user", "content": prompt}], obs_name="reaction_bridge"
    )
    try:
        result = parse_bridge_reply(reply, player_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("bridge parse failed (%s); falling back to raw text", exc)
        result = BridgeResult(situation=f"the player says: {player_text}")
    logger.info(
        "bridge: functions=%s intensity=%s valence=%s situation=%r",
        result.functions, result.intensity, result.valence, result.situation,
    )
    return result
