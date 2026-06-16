"""End-to-end smoketest: launch the server, fire `start_game`, save the TTS audio.

Spawns ``metahuman_actor.server`` as a subprocess, waits for the websocket to
accept connections, sends ``{"type": "start_game"}`` to trigger Ava's opening
line, collects the text frame and PCM audio chunks, and writes them to
``out_opening.wav``.

Prereqs (set in your environment or .env):
    CEREBRAS_API_KEY     — for the LLM call inside the opening flow
    ELEVENLABS_API_KEY   — for the TTS provider configured in persona.json

Usage::

    uv run python tests/smoketests/tts_smoketest.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import websockets
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]  # repo root (tests/smoketests/<this>)
DEFAULT_PORT = 8788
DEFAULT_OUTPUT = ROOT / "out_opening.wav"


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """Write 16-bit mono PCM bytes to a RIFF/WAVE file."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm)
    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm)


async def _wait_for_server(port: int, deadline: float) -> None:
    """Poll the websocket port until it accepts a connection or we time out."""
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(f"ws://localhost:{port}", open_timeout=2):
                return
        except Exception as exc:  # noqa: BLE001 — any connection failure is "not ready"
            last_err = exc
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Server did not accept connections on :{port} within deadline") from last_err


async def _drive_opening(port: int, output: Path) -> None:
    """Connect, fire start_game, collect text + audio, write WAV."""
    uri = f"ws://localhost:{port}"
    print(f"[client] connecting to {uri}")
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "start_game"}))
        print("[client] sent start_game")

        pcm_chunks: list[bytes] = []
        sample_rate = 0
        text_seen = False

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                print("[client] timed out waiting for audio_done")
                break
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "text":
                text_seen = True
                print(f"[{msg.get('actor_name')}] {msg.get('text')}")
            elif mtype == "audio_chunk":
                pcm_chunks.append(base64.b64decode(msg["data"]))
                if not sample_rate:
                    sample_rate = int(msg.get("sample_rate") or 0)
                    print(f"[client] receiving audio @ {sample_rate} Hz")
                if msg.get("is_final"):
                    print(f"[client] final audio chunk for line {msg.get('line_id')}")
            elif mtype == "audio_done":
                print(f"[client] audio_done for line {msg.get('line_id')}")
                break
            elif mtype == "game_event":
                print(f"[event] {msg.get('name')} {msg.get('info')}")
            elif mtype == "error":
                print(f"[error] {msg.get('message')}")
                break
            else:
                print(f"[unhandled] {msg}")

        if pcm_chunks and sample_rate:
            pcm = b"".join(pcm_chunks)
            _write_wav(output, pcm, sample_rate)
            print(f"[client] wrote {len(pcm)} bytes of PCM → {output}")
        else:
            print("[client] no audio received — did the server have ELEVENLABS_API_KEY set?")

        if not text_seen:
            print("[client] WARNING: no text frame seen; the LLM step may have failed")

        # Tell the stage we finished playback so its followup/idle timers reset
        # for any future requests on this socket. Not strictly needed for a
        # one-shot smoketest, but keeps the contract honest.
        try:
            last_line_id = pcm_chunks and msg.get("line_id") or ""
            if last_line_id:
                await ws.send(json.dumps({"type": "audio_finished", "line_id": last_line_id}))
        except Exception:
            pass


async def _main(port: int, output: Path, server_startup_secs: float) -> int:
    cmd = [
        "uv", "run", "python", "-m", "metahuman_actor.server",
        "--port", str(port), "--langfuse-local",
    ]
    print(f"[runner] spawning: {' '.join(cmd)}")
    server = subprocess.Popen(cmd, cwd=str(ROOT))

    try:
        deadline = time.monotonic() + server_startup_secs
        try:
            await _wait_for_server(port, deadline)
        except TimeoutError as exc:
            print(f"[runner] {exc}")
            return 1
        print("[runner] server is up")

        await _drive_opening(port, output)
        return 0
    finally:
        print("[runner] terminating server")
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the server to accept its first connection.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args.port, args.output, args.startup_timeout)))


if __name__ == "__main__":
    main()
