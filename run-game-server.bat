@echo off
REM Boot the game-driven listener server in caption_ctrl match mode.
REM
REM Default: --listener --reaction-match caption_ctrl
REM   caption_ctrl = CTRL rig-control bank (~251 CTRL_expressions_* + head YPR),
REM                  the raw MetaHuman Animator solve of a source video, picked
REM                  by that video's caption. Frames carry
REM                  schema:"metahuman_ctrl_v1" so the Unreal client writes them
REM                  straight into AnimNode_RigLogic (no ARKit round-trip).
REM                  Pair with the client's PostReactionMode = Reverse Ping-Pong.
REM
REM Extra args after the bat name are appended and win over the defaults, e.g.
REM   run-game-server.bat --reaction-match caption_llm    (Express4D ARKit bank)
REM   run-game-server.bat --reaction-match context        (ARKit bank, MaxSim)
REM
REM CTRL bank sources — set below, edit there or override in the environment:
REM   ICONIC_CTRL_WIRE_DIR     folder of clip wire JSONs
REM   ICONIC_CTRL_CLIP_FILTER  comma-separated substrings; only matching clips
REM                            enter the bank. Clear it to use the whole folder.
REM   EXPRESS4D_DATASET        holds video_summaries.txt (the captions) and the
REM                            ARKit dataset used by the other two modes
REM
REM Unlike run-server.bat (the authoritative runtime), this server produces
REM lines only when the game requests them (respond / trigger / set_scene /
REM set_interaction). It owns no clock. New-layout scenarios live under
REM .langfuse_prompts\scenarios\<name>\ (e.g. zeek_gd).
REM
REM Pass --langfuse-local to load prompts from .langfuse_prompts\ instead of
REM remote Langfuse. Drop the flag once you have LANGFUSE_BASE_URL /
REM LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY set in .env and prompts uploaded.
REM LLM: the office GPU box (nyx) is unreachable over VPN, so use a cloud model.
REM This .env's CEREBRAS_API_KEY only has access to gpt-oss-120b and zai-glm-4.7
REM (NOT qwen-3-235b, which is why the default 404s). gpt-oss-120b returns usable
REM text; zai-glm-4.7 streams only hidden reasoning and comes back empty under
REM llm_lib. For Qwen-235B parity over VPN, add NEBIUS_API_KEY to .env and use:
REM   --llm nebius/Qwen/Qwen3-235B-A22B-Instruct-2507

setlocal
cd /d "%~dp0"
set PORT=8788
set MATCH=caption_ctrl

REM Speak as well as react: LLM reply -> TTS -> Audio2Face drives the mouth,
REM while the CTRL clip keeps expression, brows, eyes and head for the line.
REM Set to empty for the original silent listener.
set SPEECH=--listener-speech 

@REM set MATCH=caption_llm
@REM set SPEECH=

REM --- CTRL bank: which clips the listener can pick from ---------------------
REM The Unreal plugin's Extras folder is where capture sessions accumulate, so
REM point at it and narrow with the filter rather than shuffling files.
set ICONIC_CTRL_WIRE_DIR=C:\Users\patty\Embodi\UGS\DigitalActor\Plugins\IconicDigitalActor\Extras\WaitingClipsCTRL

REM FOCUS MODE: only the MySlate_7 depth-tracked takes (7 clips, 206 s).
REM Clear this line to fall back to the whole folder (16 clips: + F_EyeContact
REM and google_empathetic_listener).
set ICONIC_CTRL_CLIP_FILTER=MySlate_7

REM Free 8788 if a previous server instance is still bound (WinError 10048).
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    echo Port %PORT% in use by PID %%P - stopping it...
    taskkill /F /PID %%P >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM uv resolves the venv from pyproject/uv.lock in this directory, so the script
REM works from any shell as long as the cd above ran.
where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: 'uv' not found on PATH. Install it ^(https://docs.astral.sh/uv/^)
    echo        or run: "%%USERPROFILE%%\.local\bin\uv.exe" sync
    echo.
    pause
    exit /b 1
)

echo Starting game-driven listener server on ws://localhost:%PORT% ^(match=%MATCH%, speech=%SPEECH%^)...
if defined ICONIC_CTRL_CLIP_FILTER echo CTRL bank filtered to: %ICONIC_CTRL_CLIP_FILTER%
uv run python -m metahuman_actor.game_driven.server --port %PORT% --llm cerebras/gpt-oss-120b --langfuse-local --listener %SPEECH% --reaction-match %MATCH% %*
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
    echo.
    echo Server exited with code %EXITCODE%.
)
pause
exit /b %EXITCODE%
