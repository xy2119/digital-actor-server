# digital-actor

R&D for **digital actors** — game NPCs whose dialogue is generated on the fly by a small, on-device LLM and spoken by on-device TTS, instead of being chosen from a bank of pre-recorded lines. A player can say anything; the story still needs to go where a narrative designer authored it. This repo holds the reusable framework, the application that wires it to LLM/TTS providers and game clients, and the scenario content.

## Layout

| Path | What it is |
|---|---|
| `packages/digital_actor/` | **`digital-actor`** — the reusable, transport-agnostic NPC-dialogue framework (Actor / Scene / Stage / Runtime). See its [README](packages/digital_actor/README.md) and [`GETTING_STARTED.md`](packages/digital_actor/digital_actor/GETTING_STARTED.md). |
| `metahuman_actor/` | The **`metahuman`** application — concrete actors, scenario loading, and the WebSocket servers. |
| `metahuman_actor/game_driven/` | The **game-driven** (request-driven) server — the current primary path. |
| `.langfuse_prompts/` | Prompt templates and game-driven scenario content (also the local Langfuse prompt root). |
| `docs/` | Design specs, plans, handovers, and research. |
| `tests/` | Pytest suite for both the package and the app. |

## Prerequisites

- Python 3.12–3.13 and [`uv`](https://docs.astral.sh/uv/).
- SSH access to Iconic's `iconic-python-lib` on GitHub — the framework depends on `tts_lib`, `llm_lib`, `langfuse_utils`, `app_logging`, and `timer_stack` from it.
- A `.env` with the relevant LLM/TTS provider keys (and `LANGFUSE_*` if not running with local prompts). Secrets are managed via Infisical (`.infisical.json`).
- For GPU-backed local TTS/LLM, CUDA — `torch` is pinned to the `cu128` wheel index (see `pyproject.toml`).

## Setup

```bash
uv sync                  # create the venv from uv.lock
uv run pre-commit install
```

## Running

On Windows, use the batch launchers (read the comments at the top of each before changing the `--llm` model — there are notes about which provider/model the current key can reach):

```bash
run-game-server.bat      # game-driven server (current primary path), port 8788
run-server.bat           # legacy authoritative-runtime server
```

Or directly:

```bash
uv run python -m metahuman_actor.game_driven.server --port 8788 --llm cerebras/gpt-oss-120b --langfuse-local
```

`--langfuse-local` reads prompts and scenarios from `.langfuse_prompts/` on disk rather than remote Langfuse.

## Testing

```bash
uv run pytest                        # full suite
uv run pytest path/to/test.py::name  # a single test
```

## How it fits together

The framework stacks four layers — **Actor** (the character brain), **Scene** (idle/followup/interrupts/story checkpoints), **Stage** (LLM + TTS + messenger), and **Runtime** (tick loop) — and you use only as many as your NPC needs. Story beats are authored as a JSON checkpoint graph rather than hand-written state machines. The full walkthrough with runnable examples lives in [`GETTING_STARTED.md`](packages/digital_actor/digital_actor/GETTING_STARTED.md); architecture-level guidance for working in this repo is in [`CLAUDE.md`](CLAUDE.md).

Authoring new scenarios is done through the **scenario-architect** and **scenario-writer** Claude skills, which scaffold and then fill the on-disk scenario tree.
