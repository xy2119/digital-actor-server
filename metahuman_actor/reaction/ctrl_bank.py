"""CTRL-curve clip bank: MetaHuman rig-control reactions picked by video caption.

The Express4D path (`dataset.py` + `retriever.py`) serves 52 ARKit blendshapes
retrieved by situation↔usage-context similarity. This bank is the high-fidelity
alternative: the raw MetaHuman Animator solve — ~251 `CTRL_expressions_*` rig
controls plus head yaw/pitch/roll — straight from video, with no ARKit
projection in between.

Two consequences shape the module:

* **Captions are all the metadata there is.** These clips come with a one-line
  video summary each (`video_summaries.txt`), not the Express4D inventory's
  function / valence / intensity / appropriateness tags. So there is nothing to
  route or rerank on: the whole bank is the candidate pool and the LLM picks by
  caption. That is why the match mode is `caption_ctrl` and not a variant of
  `context`.
* **The frames are already on the wire.** `wire_from_ctrl.py` writes complete
  `type:"reaction"` JSON frames (base64 float32 LE, row-major, tagged
  `schema:"metahuman_ctrl_v1"`), which is exactly what the client expects. We
  forward the file and patch the per-request fields rather than re-encoding.
  Metadata (duration) comes from the sibling `.npy` header so listing the bank
  does not mean parsing megabytes of base64.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from app_logging import get_logger

from metahuman_actor.reaction.dataset import dataset_root

logger = get_logger(__name__)

#: The Unreal plugin's waiting-clip folder, where capture sessions accumulate.
#: Override with ICONIC_CTRL_WIRE_DIR.
DEFAULT_CTRL_WIRE_DIR = (
    r"C:\Users\patty\Embodi\UGS\DigitalActor\Plugins\IconicDigitalActor\Extras\WaitingClipsCTRL"
)

#: Captions for the source videos, one `<file>.mp4, <caption>` line each.
CAPTIONS_FILENAME = "video_summaries.txt"

#: Comma-separated substrings; when set, only clips whose stem contains one of
#: them enter the bank. The point is focused evaluation: pointing the wire dir at
#: a folder that has accumulated several capture sessions and then narrowing to
#: one ("MySlate_7") is how you look at a single session's takes without moving
#: files around. Empty = the whole folder.
CLIP_FILTER_ENV = "ICONIC_CTRL_CLIP_FILTER"

#: Suffix tokens the export pipeline appends that the caption file does not have:
#: `_ctrl` marks the curve export, `_depth` the tracking source.
_STRIP_TOKENS = {"ctrl", "depth"}

#: `seg01` -> `01`. The exporter segments a long capture and names the pieces
#: `<slate>_seg<NN>`, while the captions are keyed by the split video
#: `<slate>_<NN>.mp4`.
_SEGMENT_RE = re.compile(r"^seg(\d+)$")

#: Solve rate of the CTRL export. The wire frames carry their own `fps`; this is
#: only the fallback for duration when a clip's .npy is read without it.
FALLBACK_FPS = 60.0

#: No function taxonomy exists for these clips. They are all silent listener
#: takes, so they answer to the same bucket the Express4D empathy clips do —
#: recorded on the frame for the client's logs, never used for routing.
DEFAULT_FUNCTION = "empathy"


def ctrl_wire_dir() -> Path:
    return Path(os.environ.get("ICONIC_CTRL_WIRE_DIR", DEFAULT_CTRL_WIRE_DIR))


def clip_filters() -> list[str]:
    raw = os.environ.get(CLIP_FILTER_ENV, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def caption_key(clip_stem: str) -> str:
    """Clip file stem -> the video stem the caption file is keyed by.

    `MySlate_7_seg01_depth_ctrl` -> `MySlate_7_01`
    `F_EyeContact3_ctrl`         -> `F_EyeContact3`

    Export names carry provenance the captions know nothing about (which solve,
    which tracking source, which segment of a longer capture). Dropping those
    tokens here keeps the two naming schemes independent — the alternative is
    renaming files every time the capture pipeline gains a suffix.
    """
    parts = []
    for token in clip_stem.split("_"):
        if token in _STRIP_TOKENS:
            continue
        segment = _SEGMENT_RE.match(token)
        parts.append(segment.group(1) if segment else token)
    return "_".join(parts)


def load_captions(root: Path | None = None) -> dict[str, str]:
    """`{video stem: caption}` from video_summaries.txt.

    Keyed by the video stem with the extension dropped, which is how the CTRL
    exports are named (`F_EyeContact.mp4` -> `F_EyeContact_ctrl.json`).
    """
    path = (root or dataset_root()) / CAPTIONS_FILENAME
    captions: dict[str, str] = {}
    if not path.exists():
        logger.warning("CTRL captions file missing: %s", path)
        return captions

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, sep, caption = line.partition(",")
            if not sep:
                logger.warning("CTRL caption line has no comma: %r", line[:80])
                continue
            stem = Path(name.strip()).stem
            captions[stem] = caption.strip()
    return captions


def _frame_count(wire_path: Path) -> int:
    """Frames in a clip, read from the sibling .npy header when possible.

    Falls back to parsing the wire JSON, which is correct but costs a full read
    of an 8 MB base64 payload per clip.
    """
    npy = wire_path.parent.parent / f"{wire_path.stem}.npy"
    if npy.exists():
        try:
            import numpy as np

            return int(np.load(npy, mmap_mode="r").shape[0])
        except Exception as exc:  # corrupt / unreadable header
            logger.warning("CTRL npy header unreadable (%s): %s", npy.name, exc)

    with open(wire_path, encoding="utf-8") as f:
        return int(json.load(f).get("num_frames") or 0)


@lru_cache(maxsize=1)
def load_ctrl_clips() -> tuple[dict, ...]:
    """The whole bank as caption-selector candidate dicts.

    Shape matches what `select_by_caption` consumes (stem / caption / dur / fn,
    plus the ranking keys it sorts on). Cached: the bank is a fixed set of files
    written offline by the extraction pipeline.
    """
    wire_dir = ctrl_wire_dir()
    if not wire_dir.is_dir():
        logger.error("CTRL wire dir missing: %s", wire_dir)
        return ()

    captions = load_captions()
    filters = clip_filters()
    clips: list[dict] = []
    missing_captions: list[str] = []
    skipped = 0

    for path in sorted(wire_dir.glob("*.json")):
        stem = path.stem
        if filters and not any(f in stem for f in filters):
            skipped += 1
            continue
        video_stem = caption_key(stem)
        caption = captions.get(video_stem, "")
        if not caption:
            missing_captions.append(f"{stem} (looked for '{video_stem}')")

        frames = _frame_count(path)
        clips.append(
            {
                "stem": stem,
                "caption": caption,
                "dur": round(frames / FALLBACK_FPS, 3) if frames else 0.0,
                "fn": DEFAULT_FUNCTION,
                # The selector sorts candidates on these when a pool needs
                # trimming. Uniform here: nothing distinguishes the clips but
                # their captions, so trimming falls through to a shuffle.
                "appropriateness": 1.0,
                "head_motion": 0.0,
                "num_frames": frames,
                "path": str(path),
            }
        )

    if missing_captions:
        # Worth shouting about: a clip with no caption still enters the pool, and
        # the LLM then picks it (or avoids it) on an empty line — the match is
        # blind rather than broken, which is harder to notice.
        logger.warning(
            "CTRL clips with no caption in %s: %s",
            CAPTIONS_FILENAME, ", ".join(missing_captions),
        )
    logger.info(
        "CTRL bank: %d clip(s) from %s (%.0fs total)%s",
        len(clips), wire_dir, sum(c["dur"] for c in clips),
        f", {skipped} skipped by {CLIP_FILTER_ENV}={','.join(filters)}" if filters else "",
    )
    return tuple(clips)


def find_ctrl_clip(stem: str) -> dict | None:
    for clip in load_ctrl_clips():
        if clip["stem"] == stem:
            return clip
    return None


@lru_cache(maxsize=None)
def _load_wire(path: str) -> dict:
    """Parsed wire JSON for one clip, cached: each is an ~8 MB base64 payload
    and re-reading it on every request put a full parse on the reply path."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def preload_ctrl_wires() -> int:
    """Parse every bank clip's wire JSON now, so no request pays for it."""
    clips = load_ctrl_clips()
    for clip in clips:
        _load_wire(clip["path"])
    return len(clips)


def build_ctrl_reaction_frame(
    stem: str,
    actor_name: str,
    caption: str = "",
    function: str = "",
) -> dict:
    """Load the pre-built wire frame and stamp the per-request fields.

    The file already holds `type`, `fps`, `num_frames`, `channels`, `data` and
    the `schema` tag the client uses to route these curves past the ARKit
    mapping pose — none of which we touch.
    """
    clip = find_ctrl_clip(stem)
    if clip is None:
        raise ValueError(f"CTRL clip {stem!r} not in bank ({len(load_ctrl_clips())})")

    # Shallow copy: only top-level per-request fields are stamped below, the
    # cached payload itself is never mutated.
    frame = dict(_load_wire(clip["path"]))

    frame["actor_name"] = actor_name
    frame["clip_id"] = stem
    frame["caption"] = caption or clip["caption"]
    frame["function"] = function or clip["fn"]
    fps = float(frame.get("fps") or FALLBACK_FPS)
    num_frames = int(frame.get("num_frames") or 0)
    frame["duration"] = round(num_frames / fps, 3) if fps > 0 else 0.0
    return frame
