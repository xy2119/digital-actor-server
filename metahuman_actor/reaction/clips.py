"""Build the outbound `reaction` wire frame carrying one clip's curve data.

Frame layout (mirrors the audio_chunk convention of base64 payloads):
    {
      "type": "reaction",
      "actor_name": "Owen",
      "clip_id": "MySlate_100_iPhone_cal",
      "caption": "looking concerned while slightly furrowing their brows",
      "function": "empathy",
      "fps": 60.0,
      "num_frames": 395,
      "channels": [... 61 channel names, npy column order ...],
      "duration": 6.58,
      "data": "<base64 of float32 little-endian, row-major (num_frames x 61)>"
    }

Long clips are split across frames — see `iter_reaction_frames`.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

from app_logging import get_logger

from metahuman_actor.reaction.dataset import CHANNELS, FPS, load_clip_frames

logger = get_logger(__name__)

#: Raw curve bytes per outbound frame once a clip is split.
#:
#: Unreal's WebSocket receive cap defaults to 1 MiB per text frame and the engine
#: closes the socket with code 1009 when a frame passes it — which drops the clip
#: *and* the server-side scenario with it. 512 KiB of float32 becomes ~700 KB of
#: base64 plus JSON, comfortably inside that default, so a client needs no config
#: change to receive a 104 s take.
CHUNK_BYTES = 512 * 1024


def iter_reaction_frames(
    frame: dict, chunk_bytes: int = CHUNK_BYTES
) -> Iterator[dict]:
    """Split one `reaction` frame into what should actually go on the wire.

    Small clips yield the frame unchanged — one self-contained `reaction`, the
    original protocol, so nothing about the Express4D path changes. Above the
    threshold the curve payload is carried by separate `reaction_chunk` frames,
    mirroring how TTS audio already streams as `audio_chunk`:

        {"type": "reaction", ..., "chunked": true, "chunk_count": 13}   # no data
        {"type": "reaction_chunk", "clip_id": ..., "actor_name": ...,
         "seq": 0, "data": "<base64 slice>", "is_final": false}
        ...
        {..., "seq": 12, "is_final": true}

    The header carries every field except `data`, so the client can size its
    buffer (`num_frames * len(channels) * 4`) before the first chunk lands.

    Slicing happens on the *raw* float bytes and each slice is base64'd on its
    own — slicing the base64 string instead would break on padding boundaries.
    """
    payload = frame.get("data") or ""
    raw = base64.b64decode(payload) if payload else b""

    if len(raw) <= chunk_bytes:
        yield frame
        return

    slices = [raw[i : i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)]

    header = {k: v for k, v in frame.items() if k != "data"}
    header["chunked"] = True
    header["chunk_count"] = len(slices)
    logger.info(
        "reaction %s split into %d chunk(s) of <=%d raw bytes (%d total)",
        frame.get("clip_id"), len(slices), chunk_bytes, len(raw),
    )
    yield header

    last = len(slices) - 1
    for seq, part in enumerate(slices):
        yield {
            "type": "reaction_chunk",
            "clip_id": frame.get("clip_id"),
            "actor_name": frame.get("actor_name"),
            "seq": seq,
            "data": base64.b64encode(part).decode("ascii"),
            "is_final": seq == last,
        }


def build_reaction_frame(
    stem: str,
    actor_name: str,
    caption: str = "",
    function: str = "",
    root: Path | None = None,
) -> dict:
    frames = load_clip_frames(stem, root)
    payload = base64.b64encode(frames.astype("<f4").tobytes()).decode("ascii")
    return {
        "type": "reaction",
        "actor_name": actor_name,
        "clip_id": stem,
        "caption": caption,
        "function": function,
        "fps": FPS,
        "num_frames": int(frames.shape[0]),
        "channels": list(CHANNELS),
        "duration": round(frames.shape[0] / FPS, 3),
        "data": payload,
    }
