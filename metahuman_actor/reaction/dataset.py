"""Express4D dataset access: inventory records, captions, and clip frames.

Layout (see C:/Users/patty/Embodi/Express4D/dataset):
    data/MySlate_<N>_iPhone_cal.npy   -- (T, 61) float32 @ 60 fps
    texts/MySlate_<N>_iPhone_cal.txt  -- "<caption>#<POS tokens>" (IDs 1-880 only)
    listener_inventory.json           -- per-clip function/valence/intensity metadata

The 61 channels are, in order: the 52 standard ARKit blendshapes
(PascalCase, identical to the ACE curve names in UE), then HeadYaw/HeadPitch/
HeadRoll, then Left/Right eye Yaw/Pitch/Roll. Head + eye rotations are signed
normalized fractions (multiply by ~72 deg to render).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from app_logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET_ROOT = r"C:\Users\patty\Embodi\Express4D\dataset"
FPS = 60.0
APPROPRIATENESS_MIN = 0.6

#: The 8-function active-listener taxonomy used by listener_inventory.json.
LISTENER_FUNCTIONS = (
    "affirm",
    "continuer",
    "doubt",
    "empathy",
    "surprise",
    "think",
    "understand",
    "turn_yield",
)

#: Channel names in npy column order (CSV header minus Timecode/BlendshapeCount).
CHANNELS: tuple[str, ...] = (
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight",
    "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight",
    "EyeSquintRight", "EyeWideRight", "JawForward", "JawRight", "JawLeft",
    "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight",
    "MouthLeft", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft",
    "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight",
    "MouthStretchLeft", "MouthStretchRight", "MouthRollLower", "MouthRollUpper",
    "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft", "MouthPressRight",
    "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft",
    "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft",
    "CheekSquintRight", "NoseSneerLeft", "NoseSneerRight", "TongueOut",
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
)


def dataset_root() -> Path:
    return Path(os.environ.get("EXPRESS4D_DATASET", DEFAULT_DATASET_ROOT))


def index_dir() -> Path:
    """Directory for offline enrichment/index artifacts (created on demand)."""
    default = Path(__file__).resolve().parents[2] / "reaction_index"
    path = Path(os.environ.get("REACTION_INDEX_DIR", str(default)))
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class ClipRecord:
    """One captioned clip with the inventory metadata used for retrieval."""

    stem: str  # e.g. "MySlate_100_iPhone_cal"
    caption: str  # cleaned caption, no POS tail / boilerplate prefix
    dur: float  # seconds
    fn: str  # primary listener function (text_fn)
    fn2: str | None  # secondary listener function
    intensity: str  # slight | medium | strong
    valence: str  # negative | neutral | positive
    appropriateness: float
    motion_actions: tuple[str, ...]


def clean_caption(raw: str) -> str:
    """Strip the POS tail after '#' and the 'A person is' boilerplate prefix."""
    text = raw.split("#", 1)[0].strip()
    for prefix in ("A person is ", "a person is "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip()


def load_records(root: Path | None = None) -> list[ClipRecord]:
    """Load captioned clips passing the appropriateness gate (851 of 880)."""
    root = root or dataset_root()
    inventory_path = root / "listener_inventory.json"
    with open(inventory_path, encoding="utf-8") as f:
        inventory = json.load(f)

    records: list[ClipRecord] = []
    for rec in inventory["records"]:
        appropriateness = float(rec.get("appropriateness") or 0.0)
        if appropriateness < APPROPRIATENESS_MIN:
            continue
        records.append(
            ClipRecord(
                stem=rec["id"],
                caption=clean_caption(rec.get("caption") or ""),
                dur=float(rec["dur"]),
                fn=rec.get("text_fn") or "continuer",
                fn2=rec.get("text_fn2"),
                intensity=rec.get("intensity") or "medium",
                valence=rec.get("valence") or "neutral",
                appropriateness=appropriateness,
                motion_actions=tuple(rec.get("motion_actions") or ()),
            )
        )
    logger.info(
        "Express4D inventory: %d/%d captioned clips pass appropriateness>=%.1f",
        len(records), len(inventory["records"]), APPROPRIATENESS_MIN,
    )
    return records


def load_clip_frames(stem: str, root: Path | None = None):
    """Load one clip as a (T, 61) float32 numpy array."""
    import numpy as np

    root = root or dataset_root()
    frames = np.load(root / "data" / f"{stem}.npy").astype(np.float32)
    if frames.ndim != 2 or frames.shape[1] != len(CHANNELS):
        raise ValueError(f"{stem}: unexpected clip shape {frames.shape}")
    return frames


#: Channel indices for HeadYaw/HeadPitch/HeadRoll (see CHANNELS order).
HEAD_ROT_INDICES = (52, 53, 54)


def head_motion_magnitude(stem: str, root: Path | None = None) -> float:
    """Peak head-rotation magnitude of a clip (max over frames of the L2 norm
    of the yaw/pitch/roll channels). ~0 for facial-only clips, up to ~1 for
    clips with a strong head turn/nod. Used to bias retrieval toward clips
    whose head visibly moves."""
    import numpy as np

    frames = load_clip_frames(stem, root)
    head = frames[:, list(HEAD_ROT_INDICES)]
    return float(np.sqrt((head * head).sum(axis=1)).max())
