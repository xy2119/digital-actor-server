"""Three-layer clip retrieval: tag route -> MaxSim cosine -> metadata rerank.

Index artifacts (built offline by build_index.py, loaded lazily here):
    vectors.npy   -- (N, 384) float32, L2-normalized bge-small embeddings
    rows.json     -- N entries [{"stem": ..., "kind": "caption"|"context"}]
    clips.json    -- per-clip metadata (subset of ClipRecord fields)

Scores are exact inner products over the full matrix (~5k x 384: <1 ms in
numpy); FAISS is deliberately not used at this scale.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from app_logging import get_logger

from metahuman_actor.reaction.dataset import index_dir

logger = get_logger(__name__)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_embed_model():
    """Load bge-small, falling back to the local HF cache when the hub
    lookup fails (offline / flaky network)."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(EMBED_MODEL)
    except OSError as exc:
        logger.warning("hub lookup failed (%s); loading from local cache", exc)
        return SentenceTransformer(EMBED_MODEL, local_files_only=True)

# Rerank weights (cosine scores live in roughly [0.4, 0.9] here).
W_INTENSITY_MATCH = 0.03
W_APPROPRIATENESS = 0.03
W_DURATION = 0.03
# Opposite-valence picks (e.g. a broad smile for grief) read as mockery, so
# this penalty is deliberately much larger than the other rerank nudges.
W_VALENCE_OPPOSITE = 0.15
W_VALENCE_OFF = 0.03
# Prefer clips whose head visibly moves so the listener reads as engaged
# rather than a frozen face. head_motion is ~0 for facial-only clips and ~1
# for a strong turn/nod; this is a tie-breaker, not a semantic override.
W_HEAD_MOTION = 0.06
DURATION_SWEET_SPOT = 4.0  # seconds; soft preference, not a cutoff
REPEAT_PENALTY = 0.12
REPEAT_MEMORY = 8
TOP_K = 8
SAMPLING_TEMPERATURE = 0.03


@dataclass
class RetrievedClip:
    stem: str
    score: float
    caption: str
    dur: float
    fn: str
    intensity: str


class ReactionRetriever:
    """Loads the offline index and serves tag-routed MaxSim retrieval."""

    def __init__(self, index_path: Path | None = None, seed: int | None = None):
        self._index_path = index_path or index_dir()
        self._rng = random.Random(seed)
        self._recent: deque[str] = deque(maxlen=REPEAT_MEMORY)
        self._encoder = None
        self._vectors = None  # (N, 384) float32, normalized
        self._rows: list[dict] = []
        self._clips: dict[str, dict] = {}
        self._clip_row_ids: dict[str, list[int]] = {}

    # -- lazy loading ------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._vectors is not None:
            return
        import numpy as np

        vectors_path = self._index_path / "vectors.npy"
        rows_path = self._index_path / "rows.json"
        clips_path = self._index_path / "clips.json"
        self._vectors = np.load(vectors_path).astype(np.float32)
        with open(rows_path, encoding="utf-8") as f:
            self._rows = json.load(f)
        with open(clips_path, encoding="utf-8") as f:
            self._clips = {c["stem"]: c for c in json.load(f)}
        self._clip_row_ids = {}
        for i, row in enumerate(self._rows):
            self._clip_row_ids.setdefault(row["stem"], []).append(i)
        logger.info(
            "reaction index: %d vectors over %d clips (%s)",
            len(self._rows), len(self._clips), vectors_path,
        )

    def _ensure_encoder(self):
        if self._encoder is None:
            logger.info("loading embedding model %s", EMBED_MODEL)
            self._encoder = load_embed_model()
        return self._encoder

    def warmup(self) -> None:
        """Load index + model and run one dummy encode (call off the hot path)."""
        self._ensure_index()
        self._ensure_encoder().encode(["warmup"], normalize_embeddings=True)

    # -- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        situation: str,
        functions: list[str] | None = None,
        intensity: str = "medium",
        valence: str = "neutral",
        top_k: int = TOP_K,
        temperature: float = SAMPLING_TEMPERATURE,
    ) -> RetrievedClip:
        """Pick one clip for the given situation description."""
        import numpy as np

        self._ensure_index()
        query = self._ensure_encoder().encode(
            [situation], normalize_embeddings=True
        )[0].astype(np.float32)

        # Layer 1: hard tag routing on primary/secondary function.
        stems = self._route(functions or [])

        # Layer 2: MaxSim cosine within the bucket.
        sims = self._vectors @ query  # (N,)
        scored: list[tuple[str, float]] = []
        for stem in stems:
            rows = self._clip_row_ids.get(stem)
            if rows:
                scored.append((stem, float(sims[rows].max())))

        # Layer 3: metadata rerank + anti-repeat, then temperature sampling.
        adjusted = [
            (stem, self._rerank(stem, sim, intensity, valence)) for stem, sim in scored
        ]
        adjusted.sort(key=lambda item: item[1], reverse=True)
        pool = adjusted[: max(1, top_k)]
        chosen_stem = self._sample(pool, temperature)

        self._recent.append(chosen_stem)
        meta = self._clips[chosen_stem]
        base = dict(scored)[chosen_stem]
        logger.info(
            "reaction pick: %s (cos=%.3f fn=%s dur=%.1fs) from bucket=%d pool=%s",
            chosen_stem, base, meta["fn"], meta["dur"],
            len(scored), [(s, round(a, 3)) for s, a in pool],
        )
        return RetrievedClip(
            stem=chosen_stem,
            score=base,
            caption=meta["caption"],
            dur=meta["dur"],
            fn=meta["fn"],
            intensity=meta["intensity"],
        )

    def list_candidates(self, functions: list[str] | None = None) -> list[dict]:
        """Clip metadata dicts after function routing (for caption_llm mode)."""
        self._ensure_index()
        return [dict(self._clips[stem]) for stem in self._route(functions or [])]

    def note_played(self, stem: str) -> None:
        """Record a played stem so both match modes share anti-repeat memory."""
        self._recent.append(stem)

    def recent_stems(self) -> list[str]:
        return list(self._recent)

    def _route(self, functions: list[str]) -> list[str]:
        if functions:
            bucket = [
                stem
                for stem, meta in self._clips.items()
                if meta["fn"] in functions or meta.get("fn2") in functions
            ]
            if bucket:
                return bucket
            logger.warning("empty bucket for functions=%s; using all clips", functions)
        return list(self._clips)

    def _rerank(self, stem: str, sim: float, intensity: str, valence: str) -> float:
        meta = self._clips[stem]
        score = sim
        if meta["intensity"] == intensity:
            score += W_INTENSITY_MATCH
        clip_valence = meta.get("valence") or "neutral"
        if clip_valence != valence:
            opposite = {clip_valence, valence} == {"negative", "positive"}
            score -= W_VALENCE_OPPOSITE if opposite else W_VALENCE_OFF
        # Reward visible head motion (saturates ~0.5 so a gentle nod already
        # scores; avoids always picking the single most extreme head-swing).
        score += W_HEAD_MOTION * min(float(meta.get("head_motion") or 0.0) / 0.5, 1.0)
        score += W_APPROPRIATENESS * (float(meta["appropriateness"]) - 0.6) / 0.4
        dur_off = abs(float(meta["dur"]) - DURATION_SWEET_SPOT) / DURATION_SWEET_SPOT
        score -= W_DURATION * min(dur_off, 1.5)
        if stem in self._recent:
            score -= REPEAT_PENALTY
        return score

    def _sample(self, pool: list[tuple[str, float]], temperature: float) -> str:
        if len(pool) == 1 or temperature <= 0:
            return pool[0][0]
        top = pool[0][1]
        weights = [math.exp((score - top) / temperature) for _, score in pool]
        return self._rng.choices([stem for stem, _ in pool], weights=weights, k=1)[0]
