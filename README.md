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

## Listener mode: which clip bank, matched how

With `--listener`, `respond` returns a **facial reaction clip** instead of dialogue (no TTS). `--reaction-match` picks the bank and the matching strategy:

| Mode | Bank | Channels | Matched by |
|---|---|---|---|
| `context` | Express4D dataset (851 captioned clips) | 52 ARKit blendshapes + head/eye YPR (61) | LLM bridge → situation + function, then MaxSim situation↔usage-contexts + rerank |
| `caption_llm` | same | same | LLM bridge → function filter, then LLM picks by facial caption |
| `caption_ctrl` | **CTRL rig-control bank** (9 clips) | ~251 `CTRL_expressions_*` + head YPR (254) | LLM picks by **video caption** over the whole bank (no function routing) |

`caption_ctrl` is the high-fidelity path: the frames are the raw MetaHuman Animator solve of a source video, so the ARKit projection is skipped entirely. Frames go out tagged `schema: "metahuman_ctrl_v1"`, which is what makes the Unreal client write the curves straight into `AnimNode_RigLogic` instead of through its ARKit mapping pose.

```bash
run-game-server.bat                                  # caption_ctrl (default)
run-game-server.bat --reaction-match caption_llm     # back to the ARKit bank
```

Sources it reads (`metahuman_actor/reaction/ctrl_bank.py`):

| What | Where | Override |
|---|---|---|
| Clip data | `IconicDigitalActor/Extras/WaitingClipsCTRL/*_ctrl.json` in the Unreal plugin — complete wire frames written by `wire_from_ctrl.py`; forwarded as-is with only `actor_name` / `clip_id` / `caption` / `function` / `duration` patched | `ICONIC_CTRL_WIRE_DIR` |
| Captions | `<Express4D dataset>/video_summaries.txt`, one `<video>.mp4, <caption>` line per clip | `EXPRESS4D_DATASET` |
| Which clips | all `*.json` in the wire folder | `ICONIC_CTRL_CLIP_FILTER` |

`ICONIC_CTRL_CLIP_FILTER` is a comma-separated substring allowlist over clip stems — for evaluating one capture session out of a folder that has accumulated several, without moving files. `run-game-server.bat` ships pointed at the Unreal plugin's `Extras/WaitingClipsCTRL` and filtered to `MySlate_7`; clear the line to use the whole folder.

Clip stems and caption keys are allowed to diverge: `caption_key()` drops the provenance tokens the export pipeline appends (`_ctrl`, `_depth`) and rewrites a `_segNN` piece to `_NN`, so `MySlate_7_seg01_depth_ctrl.json` finds the `MySlate_7_01.mp4` caption. A clip whose caption cannot be found still enters the pool — the LLM then picks it on an empty line, which is why the loader warns and names the key it looked for.

Two properties of this bank to keep in mind:

- **No taxonomy.** These clips carry no function / valence / intensity / appropriateness tags, so there is nothing to route or rerank on — the whole bank is the candidate pool. Anti-repeat is shared with the other modes.
- **Long takes.** 6 s to 104 s, mostly over 30 s, versus a few seconds for Express4D. On the client, pair the mode with `PostReactionMode = Reverse Ping-Pong` (replays the finished take backwards, then forwards) instead of a waiting-clip bank — see `IconicDigitalActor/Extras/CTRL-Reaction-Mode.md` in the Unreal project.

The mode is also switchable at runtime over the existing `set_reaction_match_mode` frame, so no client rebuild is needed to A/B it.

### Speaking listener (`--listener-speech`)

By default a listener reaction *replaces* dialogue — the NPC answers with a face and no words. With `--listener-speech` a `respond` produces both: the reaction frame goes out first, then the normal LLM reply is generated, spoken by TTS and turned into mouth animation by Audio2Face.

The two do not fight over the face, because they own different parts of it:

| | Driven by |
|---|---|
| Jaw, mouth, tongue | Audio2Face, for the duration of the line |
| Brows, eyes, cheeks, nose | the CTRL reaction clip, throughout |
| Head yaw/pitch/roll | the CTRL reaction clip, throughout |

Client side that split is enforced in `FAnimNode_IconicFaceReaction`: it sits downstream of the ARKit mapping pose that converts ACE's output into rig controls, so at full weight it would otherwise overwrite the lipsync with the clip's own closed mouth. `bMaskMouthWhileSpeaking` fades its authority over the mouth region as the component's `SpeechMouthWeight` rises (ramped over `MouthHandoverSeconds`, so the jaw is handed over rather than snapped). The reaction clip also keeps playing — and keeps ping-ponging — through the line instead of standing down.

Reaction first, speech second, on purpose: the face should already be reacting when the voice starts.

## Testing

```bash
uv run pytest                        # full suite
uv run pytest path/to/test.py::name  # a single test
```

## How it fits together

The framework stacks four layers — **Actor** (the character brain), **Scene** (idle/followup/interrupts/story checkpoints), **Stage** (LLM + TTS + messenger), and **Runtime** (tick loop) — and you use only as many as your NPC needs. Story beats are authored as a JSON checkpoint graph rather than hand-written state machines. The full walkthrough with runnable examples lives in [`GETTING_STARTED.md`](packages/digital_actor/digital_actor/GETTING_STARTED.md); architecture-level guidance for working in this repo is in [`CLAUDE.md`](CLAUDE.md).

Authoring new scenarios is done through the **scenario-architect** and **scenario-writer** Claude skills, which scaffold and then fill the on-disk scenario tree.
