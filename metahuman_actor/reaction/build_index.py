"""Offline index build: embed captions + usage contexts with bge-small.

Only text views are embedded; function tags and numeric metadata stay in
clips.json for hard routing / reranking (embedding discrete tags is noise).

Run (after enrich.py):
    uv run python -m metahuman_actor.reaction.build_index

Writes to <index_dir>/:
    vectors.npy  (N, 384) float32, L2-normalized
    rows.json    N x {"stem", "kind": "caption"|"context", "text"}
    clips.json   per-clip metadata for routing + rerank
"""

from __future__ import annotations

import json

from app_logging import get_logger, setup_logging

from metahuman_actor.reaction.dataset import (
    head_motion_magnitude,
    index_dir,
    load_records,
)
from metahuman_actor.reaction.retriever import EMBED_MODEL, load_embed_model

logger = get_logger(__name__)


def main() -> None:
    import numpy as np

    out = index_dir()
    contexts_path = out / "usage_contexts.json"
    contexts: dict[str, list[str]] = {}
    if contexts_path.exists():
        with open(contexts_path, encoding="utf-8") as f:
            contexts = json.load(f)
    else:
        logger.warning("no usage_contexts.json yet — indexing captions only")

    records = load_records()
    rows: list[dict] = []
    for rec in records:
        rows.append({"stem": rec.stem, "kind": "caption", "text": rec.caption})
        for ctx in contexts.get(rec.stem, []):
            rows.append({"stem": rec.stem, "kind": "context", "text": ctx})

    logger.info("embedding %d texts over %d clips with %s", len(rows), len(records), EMBED_MODEL)
    encoder = load_embed_model()
    vectors = encoder.encode(
        [row["text"] for row in rows],
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(out / "vectors.npy", vectors)
    with open(out / "rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    clips = [
        {
            "stem": rec.stem,
            "caption": rec.caption,
            "dur": rec.dur,
            "fn": rec.fn,
            "fn2": rec.fn2,
            "intensity": rec.intensity,
            "valence": rec.valence,
            "appropriateness": rec.appropriateness,
            "head_motion": head_motion_magnitude(rec.stem),
        }
        for rec in records
    ]
    with open(out / "clips.json", "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=1)
    logger.info("index written to %s: vectors %s", out, vectors.shape)


if __name__ == "__main__":
    setup_logging()
    main()
