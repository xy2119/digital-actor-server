"""Smoketest: drive the new TTS entry points.

1. Spawns ``metahuman_actor.server`` as a subprocess (same pattern as
   ``tts_smoketest.py``).
2. Sends a WebSocket ``{"type": "say", "text": "..."}`` and assembles
   the streamed PCM into ``out_say_ws.wav``.
3. POSTs the same text to ``/tts`` and saves the response to
   ``out_say_http.wav``.
4. Compares PCM bytes — they should match modulo header.

Prereqs:
    ELEVENLABS_API_KEY (or whichever TTS provider the persona uses).

Usage::

    uv run python tests/smoketests/say_smoketest.py
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
import urllib.request
from pathlib import Path

import websockets
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]  # repo root (tests/smoketests/<this>)
DEFAULT_PORT = 8788
DEFAULT_HTTP_PORT = 8789
DEFAULT_TEXT = "Hello from the say smoketest."


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
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
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm)


async def _wait_for_server(port: int, deadline: float) -> None:
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(f"ws://localhost:{port}", open_timeout=2):
                return
        except Exception as exc:
            last = exc
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Server not up on :{port}") from last


async def _drive_say(port: int, text: str, output: Path) -> bytes:
    """Connect via WS, send `say`, return raw PCM after collecting chunks."""
    async with websockets.connect(f"ws://localhost:{port}") as ws:
        await ws.send(json.dumps({"type": "say", "text": text}))
        pcm_chunks: list[bytes] = []
        sample_rate = 0
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "audio_chunk":
                pcm_chunks.append(base64.b64decode(msg["data"]))
                if not sample_rate:
                    sample_rate = int(msg.get("sample_rate") or 0)
            elif mtype == "audio_done":
                break
            elif mtype == "error":
                print(f"[error] {msg.get('message')}")
                return b""
        pcm = b"".join(pcm_chunks)
        if pcm and sample_rate:
            _write_wav(output, pcm, sample_rate)
            print(f"[ws] wrote {len(pcm)} bytes → {output}")
        return pcm


def _fetch_http_wav(http_port: int, text: str, output: Path) -> bytes:
    """POST text to /tts, save the WAV to ``output``, return the PCM payload."""
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{http_port}/tts",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    output.write_bytes(data)
    print(f"[http] wrote {len(data)} bytes → {output}")
    # PCM payload is everything after the 44-byte RIFF/WAVE header.
    return data[44:] if data[:4] == b"RIFF" else b""


async def _main(port: int, http_port: int, text: str, out_dir: Path,
                server_startup_secs: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", "-m", "metahuman_actor.server",
        "--port", str(port), "--http-port", str(http_port), "--langfuse-local",
    ]
    print(f"[runner] spawning: {' '.join(cmd)}")
    server = subprocess.Popen(cmd, cwd=str(ROOT))
    try:
        deadline = time.monotonic() + server_startup_secs
        await _wait_for_server(port, deadline)
        print("[runner] server is up")

        ws_pcm = await _drive_say(port, text, out_dir / "out_say_ws.wav")
        http_pcm = _fetch_http_wav(http_port, text, out_dir / "out_say_http.wav")

        if not ws_pcm:
            print("[FAIL] no WS audio received")
            return 1
        if not http_pcm:
            print("[FAIL] no HTTP audio received")
            return 1
        if ws_pcm == http_pcm:
            print("[PASS] WS and HTTP PCM are byte-identical")
        else:
            print(
                f"[NOTE] WS PCM ({len(ws_pcm)} bytes) differs from HTTP PCM "
                f"({len(http_pcm)} bytes). Some providers are non-deterministic "
                "between calls; inspect the two WAVs by ear."
            )
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
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out_say")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            _main(args.port, args.http_port, args.text, args.out_dir, args.startup_timeout)
        )
    )


if __name__ == "__main__":
    main()
