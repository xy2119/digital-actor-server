"""Offline enrichment: generate K usage-context sentences per captioned clip.

This moves the semantic bridging cost offline: at runtime the query (an LLM
"the player ..." situation sentence) and these documents live in the same
space. One LLM call per clip, resumable, ~851 clips.

Run:
    uv run python -m metahuman_actor.reaction.enrich [--limit N] [--concurrency 8]

Writes <index_dir>/usage_contexts.json: {stem: [context, ...]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re

from app_logging import get_logger, setup_logging
from dotenv import load_dotenv
from llm_lib import get_llm_client

from metahuman_actor.reaction.dataset import ClipRecord, index_dir, load_records

logger = get_logger(__name__)

K_CONTEXTS = 5
DEFAULT_MODEL = "cerebras/gpt-oss-120b"

ENRICH_PROMPT = """\
An NPC is a silent, empathetic listener in a game. The player talks; the NPC
reacts only with facial expressions. One reaction clip is described as:

  expression: "{caption}"
  listener function: {fn}{fn2}
  valence: {valence}, intensity: {intensity}

Write {k} DIFFERENT short English sentences describing player utterances or
moments this reaction would suit. Each sentence MUST start with "the player"
and describe what the player says/expresses (not the NPC). Cover distinct
situations (topics, emotions, phrasings), stay concrete and everyday.

Example for "pursing lips and lowering gaze as if thinking deeply":
  "the player asks something that requires working out an answer"
  "the player explains complicated details"

Reply with ONLY a JSON array of {k} strings, no markdown."""


def _parse_contexts(reply: str, k: int) -> list[str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip(), flags=re.MULTILINE)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array in reply: {reply[:200]!r}")
    items = json.loads(match.group(0))
    contexts = [str(s).strip() for s in items if str(s).strip()]
    if not contexts:
        raise ValueError("empty contexts array")
    return contexts[:k]


async def _enrich_one(llm_client, rec: ClipRecord, semaphore: asyncio.Semaphore) -> list[str]:
    prompt = ENRICH_PROMPT.format(
        caption=rec.caption,
        fn=rec.fn,
        fn2=f" (secondary: {rec.fn2})" if rec.fn2 else "",
        valence=rec.valence,
        intensity=rec.intensity,
        k=K_CONTEXTS,
    )
    async with semaphore:
        for attempt in range(3):
            reply = await llm_client.acomplete(
                [{"role": "user", "content": prompt}], obs_name="reaction_enrich"
            )
            try:
                return _parse_contexts(reply, K_CONTEXTS)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("%s attempt %d: %s", rec.stem, attempt + 1, exc)
    raise RuntimeError(f"enrichment failed for {rec.stem}")


async def main(model: str, limit: int | None, concurrency: int) -> None:
    out_path = index_dir() / "usage_contexts.json"
    done: dict[str, list[str]] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = json.load(f)

    records = load_records()
    todo = [r for r in records if r.stem not in done]
    if limit:
        todo = todo[:limit]
    logger.info("enriching %d clips (%d already done)", len(todo), len(done))
    if not todo:
        return

    llm_client = get_llm_client(model)
    semaphore = asyncio.Semaphore(concurrency)
    failed: list[str] = []

    async def worker(rec: ClipRecord) -> None:
        try:
            done[rec.stem] = await _enrich_one(llm_client, rec, semaphore)
        except Exception:
            logger.exception("giving up on %s", rec.stem)
            failed.append(rec.stem)

    checkpoint_every = 50
    for start in range(0, len(todo), checkpoint_every):
        batch = todo[start : start + checkpoint_every]
        await asyncio.gather(*(worker(r) for r in batch))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(done, f, ensure_ascii=False, indent=1)
        logger.info("checkpoint: %d/%d done, %d failed", len(done), len(records), len(failed))

    logger.info("enrichment complete: %d clips, %d failed %s", len(done), len(failed), failed[:10])


if __name__ == "__main__":
    load_dotenv()
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.limit, args.concurrency))
