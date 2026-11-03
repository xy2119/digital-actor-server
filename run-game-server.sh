#!/usr/bin/env bash
# Linux counterpart of run-game-server.bat (used on hades). Same defaults:
# --listener --listener-speech --reaction-match caption_ctrl, cloud LLM + TTS.
# Extra args are appended and win over the defaults.
#
# The server binds ws://localhost:$PORT only; reach it from another machine
# with an SSH tunnel:  ssh -N -L 8788:localhost:8788 hades
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8788}"
MATCH="${MATCH:-caption_ctrl}"
SPEECH="${SPEECH---listener-speech}"

# Pin to the RTX 5090 (PCI order index 0 on hades; the 4090 is shared).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export ICONIC_CTRL_WIRE_DIR="${ICONIC_CTRL_WIRE_DIR:-$HOME/embodi/WaitingClipsCTRL}"
export ICONIC_CTRL_CLIP_FILTER="${ICONIC_CTRL_CLIP_FILTER-MySlate_7}"
export EXPRESS4D_DATASET="${EXPRESS4D_DATASET:-$HOME/embodi/Express4D/dataset}"

echo "Starting game-driven listener server on ws://localhost:$PORT (match=$MATCH, speech=$SPEECH, gpu=$CUDA_VISIBLE_DEVICES)"
[ -n "$ICONIC_CTRL_CLIP_FILTER" ] && echo "CTRL bank filtered to: $ICONIC_CTRL_CLIP_FILTER"

# shellcheck disable=SC2086
exec uv run python -m metahuman_actor.game_driven.server \
    --port "$PORT" --llm cerebras/gpt-oss-120b --langfuse-local \
    --listener $SPEECH --reaction-match "$MATCH" "$@"
