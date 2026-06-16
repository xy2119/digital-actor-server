# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An R&D repo for **digital actors** — game NPCs whose dialogue is generated on the fly by a small LLM and spoken by on-device TTS, rather than picked from a bank of pre-recorded lines. The player can say anything; the story still needs to go where a narrative designer authored it. There are two pieces:

- **`packages/digital_actor/`** — `digital-actor`, a reusable, transport-agnostic framework (Actor / Scene / Stage / Runtime layers). Has its own README and `GETTING_STARTED.md`. Imported as an editable workspace dependency.
- **`metahuman_actor/`** — the `metahuman` application: concrete actors, scenario loading, and two WebSocket servers built on top of `digital_actor`. This is where most active work happens.

The framework depends on Iconic's internal `iconic-python-lib` (`tts_lib`, `llm_lib`, `langfuse_utils`, `app_logging`, `timer_stack`), pulled over SSH from GitHub — you need access to that repo to install.

## Commands

This is a `uv` project (Python 3.12/3.13). Do not invoke `python` directly — go through `uv run`.

```bash
uv sync                              # install/refresh the venv from uv.lock
uv run pytest                        # run the whole suite (asyncio_mode=auto; no decorators needed)
uv run pytest tests/metahuman_actor/game_driven/test_stage.py            # one file
uv run pytest tests/metahuman_actor/game_driven/test_stage.py::test_name # one test
uv run pre-commit run --all-files    # ruff check --fix, ruff-format, isort, hygiene hooks
```

`pre-commit` hooks (ruff, isort) only apply to `metahuman_actor/`. The `digital_actor` package is not auto-linted by them.

### Running the servers (Windows)

```bash
run-game-server.bat        # game-driven server (the current/primary path)
run-server.bat             # legacy authoritative-runtime server
```

Both wrap `uv run python -m metahuman_actor.<...>.server` with `--port`, `--llm`, and `--langfuse-local`. Read the comments at the top of `run-game-server.bat` before changing the LLM — there are live constraints about which Cerebras/Nebius models the current `.env` key can actually reach. `%*` forwards extra args.

`--langfuse-local` loads prompts from `.langfuse_prompts/` on disk instead of remote Langfuse. Drop it once `LANGFUSE_*` env vars are set and prompts are uploaded. Secrets come from `.env` (and Infisical, per `.infisical.json`).

## Architecture

### The four-layer framework (`digital_actor`)

Layers stack; you stop at whichever one your NPC needs (see `packages/digital_actor/digital_actor/GETTING_STARTED.md` and `examples/` for runnable versions of each):

| Layer | Class | Responsibility |
|---|---|---|
| **Actor** | `DigitalActor` / `SceneDigitalActor` | Character brain — prompt construction, dialogue history, TTS |
| **Scene** | `SingleActorScene` / `BaseScene` | Moment-to-moment interaction — idle, followup, interrupts, checkpoints |
| **Stage** | `SingleActorStage` / `SingleSceneStage` / `MultiActorStage` / `MultiSceneStage` | Infrastructure — LLM client, TTS client, messenger, output routing |
| **Runtime** | `Runtime` | Tick loop — calls `stage.step(t)` at a fixed rate |

Two structural rules that are easy to miss:

- **Actors and scenes never reference the stage directly.** They reach infrastructure through a global proxy, `stage_context`, which is wired automatically when any stage is constructed. The app has its own `metahuman_actor/stage_context.py`. This keeps actors/scenes free of infrastructure coupling — respect it; don't pass stage references in.
- **The messenger is the only path out to the client.** Every `deliver_text` / `deliver_speech` flows through it. `WebSocketMessenger` and `GrpcMessenger` share one session-routing interface, so stage code is transport-agnostic. `NullMessenger` (the default) makes `on_user_input` return the response text directly — that's how most tests drive a stage.

### Checkpoints = story beats

`SceneCheckpoints` is a small dependency graph of story beats, authored as JSON, that lets designers sequence narrative without writing state-machine code:

- **`EventCheckpoint`** fires when a named game event arrives from the engine.
- **`QueryCheckpoint`** fires when an LLM yes/no query against the dialogue answers YES.

A node activates only once its `dependency` is satisfied. Firing injects a narrator message into history and emits callbacks as game events back to the client. Swapping a scene's `scene_data` (backstory + script + checkpoint graph) mid-run is how an actor transitions between "acts"; history carries over unless you reset it.

### Two servers — know which one you're touching

- **`metahuman_actor.game_driven.server` (primary, current work).** *Request-driven*: it owns **no clock**. NPCs speak only when the game asks — `respond` / `trigger` / `set_scene` / `set_interaction` / `load_scenario` over a JSON WebSocket protocol (see `game_driven/server.py` for the full message set, including `list_scenarios`, `warmup_tts`, and `followup_hint` frames). `GameDrivenStage` holds one `LoadedCharacter` per scenario character and an active-character pointer; `stage_context.scene_data`/`.tts_client` return the *active* character's values. This is safe only because the WS loop is sequential and each scene serializes its own generation — preserve that invariant.
- **`metahuman_actor.server` (legacy).** The authoritative-runtime server: drives its own tick loop for idle/followup behaviour.

### Scenarios — two layouts coexist

Both servers load scenarios, but from **different layouts**, and this is the single biggest source of confusion in the repo:

- **Legacy layout** (`metahuman_actor.scenario.Scenario`): lives under `metahuman_actor/scenarios/<name>/` as `persona.json` + `scene1/checkpoints.json`.
- **Game-driven layout** (`metahuman_actor.game_driven.scenario.GameDrivenScenario`): lives under **`.langfuse_prompts/scenarios/<name>/`** (the local Langfuse prompt root, or `LOCAL_LANGFUSE_PATH`). Structure:

  ```
  scenarios/<name>/
    scenario.json            # default_character/scene/interaction (+ optional "characters" list)
    back_story.txt
    personas/<character>.json
    <scene>/
      scene_description.txt
      characters/<character>/
        character_back_story.txt
        <interaction>/
          steer_back_instructions.txt
          opening_speech.txt        (optional)
          checkpoints.json          (optional)
          triggers/<name>/prompt.txt (+ optional narrator.txt)
  ```

  Key design point: scenario trees live **inside** the Langfuse prompt root so the same files are both discoverable on disk by the loader *and* addressable by `get_prompt("scenarios/<name>/...")`. **All prose content is read through `langfuse_utils.get_prompt`** (so it flips local↔remote Langfuse with no code change and stays observable in traces) — only `scenario.json`, `persona.json`, and `checkpoints.json` are read directly off disk as structured data. Trigger *structure* is discovered by listing the `triggers/` folder; each trigger's *body* is then loaded by `get_prompt`, and `{{var}}` placeholders are substituted from event info via `.compile(**info)`.

When adding scenario content, prefer the **scenario-architect** and **scenario-writer** skills — they encode the authoring workflow (architect scaffolds the empty tree + blueprint; writer fills the prose and runs a coherence pass).

### Settings & prompts

`metahuman_actor/settings.py` loads `settings.yaml` into a pydantic `Settings` (timeouts, `prompt_label`, playback buffer). Shared prompt templates (`dialogue/`, `common/`, `query/`, `summary/`) live alongside scenarios under `.langfuse_prompts/` and are resolved by name through `get_prompt`.

## Testing notes

- `asyncio_mode=auto` — async tests need no decorator.
- Game-driven tests run against a **temp scenario tree** built by `write_scenario_tree` (in `tests/metahuman_actor/game_driven/conftest.py`), with `LOCAL_LANGFUSE_PATH` pointed at `tmp_path` and the repo's real shared prompt templates copied in. When you change the scenario layout or the prompt-name scheme, update `write_scenario_tree` and `_seed_shared_prompts` to match, or every game-driven test silently drifts from production paths.
- `tests/smoketests/` holds **manual** end-to-end smoke tests that boot a real server subprocess and hit live LLM/TTS providers (see its README). They are intentionally *not* pytest-collected (filenames don't match `test_*`) and need provider keys. They target the legacy server; there's no game-driven equivalent yet.
