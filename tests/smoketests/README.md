# Smoketests

Manual, end-to-end smoke tests that boot a **real** `metahuman_actor.server` subprocess, drive it over the live WebSocket/HTTP protocol with real LLM and TTS providers, and write the resulting audio to WAV files for inspection.

These are **not** collected by `pytest` (their filenames don't match the `test_*` pattern, and they need network + provider keys). The `pytest` suite under `tests/` uses `NullMessenger` and never hits a real provider — these fill that gap. They target the **legacy** authoritative-runtime server (`metahuman_actor.server`), whose `start_game` / `say` / `user_input` / `/tts` protocol they exercise.

Run from the repo root:

```bash
uv run python tests/smoketests/tts_smoketest.py                    # start_game → opening line → out_opening.wav
uv run python tests/smoketests/say_smoketest.py                    # WS `say` vs HTTP /tts produce identical PCM
uv run python tests/smoketests/streaming_serialization_smoketest.py # mid-stream user_input must not interleave audio (asserts PASS/FAIL)
```

Each spawns the server itself and tears it down on exit; pass `--help` for flags. Output WAVs land in the repo root / an `out_*` dir.

**Prereqs:** a `.env` with `CEREBRAS_API_KEY` (LLM) and the TTS provider key the persona uses (e.g. `ELEVENLABS_API_KEY`). The scripts run the server with `--langfuse-local`, so prompts come from `.langfuse_prompts/`.

> There is no equivalent smoke test for the current game-driven server (`metahuman_actor.game_driven.server`, the `respond`/`trigger` protocol) yet — worth adding.
