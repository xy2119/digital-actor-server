"""E2E test: a user_input arriving mid-stream must not interleave with the
in-flight line's audio.

Reproduces the original symptom: a second ``user_input`` sent while the
opening line is still streaming used to interleave its ``audio_chunk`` frames
with the first line on the same outbound queue, which made the audio Unreal
was playing hang and then jump to the new line. After the fix, the scene's
``_response_lock`` queues the second pipeline behind the first, so all
``audio_chunk`` frames for line 1 arrive contiguously before the first frame
for line 2.

Spawns ``metahuman_actor.server`` as a subprocess (same pattern as
``tts_smoketest.py``), drives the scenario, and writes both lines'
audio to separate WAVs so a human can also listen for smoothness.

Prereqs:
    CEREBRAS_API_KEY     — for the LLM call inside the opening/response flow
    ELEVENLABS_API_KEY   — for the TTS provider configured in persona.json

Usage::

    uv run python tests/smoketests/streaming_serialization_smoketest.py
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
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(f"ws://localhost:{port}", open_timeout=2):
                return
        except Exception as exc:
            last_err = exc
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Server did not accept connections on :{port}") from last_err


async def _run_scenario(port: int, out_dir: Path, trigger_after_chunks: int) -> int:
    uri = f"ws://localhost:{port}"
    print(f"[client] connecting to {uri}")
    async with websockets.connect(uri) as ws:
        t0 = time.monotonic()
        def ts() -> str:
            return f"+{time.monotonic() - t0:6.2f}s"

        await ws.send(json.dumps({"type": "start_game"}))
        print(f"[{ts()}] sent start_game")

        # Frame log: (line_id, mtype) in arrival order.
        order: list[tuple[str, str]] = []
        # Timestamped event log for gap measurements: (relative_ts, mtype, line_id).
        event_log: list[tuple[float, str, str]] = []
        # Per-line PCM accumulators and counters.
        pcm: dict[str, list[bytes]] = {}
        sample_rate: dict[str, int] = {}
        chunk_counts: dict[str, int] = {}
        done: set[str] = set()
        first_line_id: str | None = None
        second_sent = False
        last_message_at = time.monotonic()

        # Stop once we've seen audio_done for two distinct lines (or hit timeout).
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
            except asyncio.TimeoutError:
                gap = time.monotonic() - last_message_at
                print(f"[{ts()}] TIMEOUT — no frames for {gap:.1f}s; giving up")
                break
            now = time.monotonic()
            gap = now - last_message_at
            last_message_at = now
            relative_ts = now - t0
            msg = json.loads(raw)
            mtype = msg.get("type")
            line_id = msg.get("line_id", "")
            if mtype in ("audio_chunk", "audio_done", "text"):
                event_log.append((relative_ts, mtype, line_id))

            if mtype == "text":
                print(f"[{ts()}] text line={line_id[:8]} (+{gap:.2f}s gap): {msg.get('text')[:80]}")
                order.append((line_id, mtype))
                if first_line_id is None:
                    first_line_id = line_id
            elif mtype == "audio_chunk":
                pcm.setdefault(line_id, []).append(base64.b64decode(msg["data"]))
                chunk_counts[line_id] = chunk_counts.get(line_id, 0) + 1
                count = chunk_counts[line_id]
                if line_id not in sample_rate and msg.get("sample_rate"):
                    sample_rate[line_id] = int(msg["sample_rate"])
                # Print every chunk so we can see the streaming cadence and
                # detect any mid-stream hang.
                print(f"[{ts()}] audio_chunk line={line_id[:8]} #{count} bytes={len(msg.get('data', '')) * 3 // 4} (+{gap:.2f}s gap)")
                order.append((line_id, mtype))

                # Fire the second user_input only after we've received enough
                # chunks to be confident the stream has momentum. Triggering on
                # the very first chunk leaves no buffer to detect a hang
                # versus a slow first-chunk warm-up.
                if (
                    not second_sent
                    and first_line_id
                    and line_id == first_line_id
                    and count >= trigger_after_chunks
                ):
                    second_sent = True
                    print(f"[{ts()}] >>> sending second user_input mid-stream (after {count} chunks)")
                    await ws.send(json.dumps({
                        "type": "user_input",
                        "text": "Hey wait, real quick — what's your name?",
                    }))
            elif mtype == "audio_done":
                print(f"[{ts()}] audio_done line={line_id[:8]} (total chunks={chunk_counts.get(line_id, 0)})")
                done.add(line_id)
                order.append((line_id, mtype))
                try:
                    await ws.send(json.dumps({"type": "audio_finished", "line_id": line_id}))
                except Exception:
                    pass
                if len(done) >= 2:
                    break
            elif mtype == "game_event":
                print(f"[{ts()}] event {msg.get('name')} {msg.get('info')}")
            elif mtype == "error":
                print(f"[{ts()}] error {msg.get('message')}")
                break
            else:
                print(f"[{ts()}] unhandled {msg}")

        # Write audio for inspection.
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, lid in enumerate(pcm.keys()):
            data = b"".join(pcm[lid])
            sr = sample_rate.get(lid, 0)
            if data and sr:
                path = out_dir / f"serialization_line{i+1}_{lid[:8]}.wav"
                _write_wav(path, data, sr)
                print(f"[client] wrote {len(data)} bytes → {path}")

        # --- Assertions -------------------------------------------------------
        line_ids = list(pcm.keys())
        if len(line_ids) < 2:
            print(f"[FAIL] expected 2 lines of audio, got {len(line_ids)}: {line_ids}")
            return 1

        # Trim the order log to just audio_chunk + audio_done events.
        audio_events = [(lid, t) for lid, t in order if t in ("audio_chunk", "audio_done")]

        first = line_ids[0]
        boundary = next(
            (i for i, (lid, _) in enumerate(audio_events) if lid != first),
            len(audio_events),
        )
        before = audio_events[:boundary]
        after = audio_events[boundary:]

        if any(lid != first for lid, _ in before):
            print("[FAIL] frame from a non-first line appeared before first line's audio_done")
            return 1
        if not before or before[-1] != (first, "audio_done"):
            print("[FAIL] first line did not terminate with audio_done before any other line's frames")
            return 1
        second_lines = {lid for lid, _ in after}
        if len(second_lines) != 1:
            print(f"[FAIL] expected exactly one second line after the boundary, got {second_lines}")
            return 1
        print("[PASS] no interleaving: first line streamed to audio_done before any second-line frame")

        # Playback-anchor assertion: the gap between line 1's audio_done and
        # line 2's first audio_chunk should be at least most of line 1's
        # estimated playback duration, since the server is supposed to hold
        # the response lock until the client is estimated to have finished
        # playing line 1. Without this, the server would send line 2's
        # first chunk (user_input_ack=True) while Unreal is still playing
        # line 1, causing the "pause then resume" hiccup.
        line1_pcm_bytes = sum(len(c) for c in pcm[first])
        line1_sr = sample_rate.get(first, 0)
        if line1_sr <= 0:
            print("[FAIL] no sample rate captured for line 1")
            return 1
        line1_duration = line1_pcm_bytes / (line1_sr * 2)  # 16-bit mono
        # Tolerate that the server can't begin line 2 instantly after release:
        # require the gap to cover at least 70% of estimated line 1 playback.
        required_gap = line1_duration * 0.7

        line1_audio_done_ts = next(
            (ts for ts, mtype, lid in event_log if mtype == "audio_done" and lid == first),
            None,
        )
        line2_first_chunk_ts = next(
            (ts for ts, mtype, lid in event_log if mtype == "audio_chunk" and lid != first),
            None,
        )
        if line1_audio_done_ts is None or line2_first_chunk_ts is None:
            print("[FAIL] could not locate timestamps for the gap measurement")
            return 1
        actual_gap = line2_first_chunk_ts - line1_audio_done_ts
        print(
            f"[gap] line1 duration={line1_duration:.2f}s, "
            f"gap(line1.audio_done → line2.first_chunk)={actual_gap:.2f}s, "
            f"required ≥{required_gap:.2f}s"
        )
        if actual_gap < required_gap:
            print(
                f"[FAIL] gap too short — server didn't pace line 2 behind line 1's playback "
                f"(got {actual_gap:.2f}s, required ≥{required_gap:.2f}s)"
            )
            return 1
        print("[PASS] playback-anchor: line 2 chunks delayed until line 1's estimated playback completed")
        return 0


async def _main(port: int, out_dir: Path, server_startup_secs: float, trigger_after_chunks: int) -> int:
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
        return await _run_scenario(port, out_dir, trigger_after_chunks)
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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out_serialization")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument(
        "--trigger-after-chunks",
        type=int,
        default=3,
        help="Send the second user_input only after this many audio chunks have arrived "
             "for the first line, so the stream has visible momentum.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args.port, args.out_dir, args.startup_timeout, args.trigger_after_chunks)))


if __name__ == "__main__":
    main()
