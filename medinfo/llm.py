"""LLM backends shared by drafting (generate.py) and the safety classifier (safety_llm.py).

Backends
  * "api"        - Anthropic Messages API (ANTHROPIC_API_KEY). The production path.
  * "claude_cli" - Claude Code headless (`claude -p`) on the developer's Claude subscription.
                   DEVELOPMENT EVALUATION ONLY: no API key needed, but latency is not
                   representative and it is not a deployment path (see NOTES.md).
  * "extractive" - no LLM at all (offline, deterministic). The hosted demo default.

Selection (select_backend): explicit choice > ANTHROPIC_API_KEY -> api >
MEDINFO_BACKEND=claude_cli -> claude_cli > extractive.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

API_MODEL = "claude-sonnet-5-5"
CLI_MODEL_ALIAS = "sonnet"     # this CLI version rejects the full ID; the alias is resolved and recorded
CLI_TIMEOUT_S = 90
BACKENDS = ("extractive", "api", "claude_cli")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMReply:
    text: str
    backend: str
    model: str
    latency_s: float


def select_backend(explicit: str | None = None, api_key: str | None = None) -> str:
    if explicit:
        if explicit not in BACKENDS:
            raise ValueError(f"unknown backend {explicit!r}; choose from {BACKENDS}")
        return explicit
    if api_key or os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    if os.environ.get("MEDINFO_BACKEND", "").strip().lower() == "claude_cli":
        return "claude_cli"
    return "extractive"


def call_llm(system: str, user: str, backend: str, api_key: str | None = None,
             max_tokens: int = 800) -> LLMReply:
    if backend == "api":
        return _call_api(system, user, api_key or os.environ.get("ANTHROPIC_API_KEY"), max_tokens)
    if backend == "claude_cli":
        return _call_cli(system, user)
    raise LLMError(f"backend {backend!r} has no LLM")


def _call_api(system: str, user: str, api_key: str | None, max_tokens: int) -> LLMReply:
    if not api_key:
        raise LLMError("api backend selected but no ANTHROPIC_API_KEY")
    import anthropic

    t0 = time.monotonic()
    msg = anthropic.Anthropic(api_key=api_key).messages.create(
        model=API_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return LLMReply(text, "api", getattr(msg, "model", API_MODEL), time.monotonic() - t0)


def _call_cli(system: str, user: str) -> LLMReply:
    """One isolated `claude -p` call.

    Isolation: the default Claude Code system prompt is REPLACED (--system-prompt-file),
    all built-in tools are disabled (--tools ""), MCP servers are dropped (--strict-mcp-config),
    user/project/local settings are not loaded (--setting-sources ""), skills/slash commands are
    off, nothing is persisted, and the working directory is a fresh empty temp dir so no project
    files or CLAUDE.md are discoverable. The init event is checked on every call: if any tool or
    MCP server is present the reply is refused.
    Known residual: Claude Code still injects a small account-context block (the user's email);
    only --bare removes it, and --bare requires an API key.
    """
    exe = shutil.which("claude")
    if not exe:
        raise LLMError("claude CLI not found on PATH")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="medinfo_cli_") as tmp:
        sp = Path(tmp) / "system_prompt.txt"
        sp.write_text(system, encoding="utf-8")
        cmd = [exe, "-p", "--system-prompt-file", str(sp), "--tools", "", "--strict-mcp-config",
               "--setting-sources", "", "--disable-slash-commands", "--model", CLI_MODEL_ALIAS,
               "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
        try:
            proc = subprocess.run(cmd, input=user, cwd=tmp, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=CLI_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise LLMError(f"claude CLI timed out after {CLI_TIMEOUT_S}s")
    model, result, init_seen = CLI_MODEL_ALIAS, None, False
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            init_seen = True
            model = ev.get("model") or model
            if ev.get("tools") or ev.get("mcp_servers"):
                raise LLMError(f"isolation check failed: tools={ev.get('tools')} mcp={ev.get('mcp_servers')}")
        elif ev.get("type") == "result":
            result = ev
    if not init_seen or result is None:
        raise LLMError(f"claude CLI gave no result (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
    if result.get("is_error"):
        raise LLMError(f"claude CLI error: {str(result.get('result'))[:200]}")
    return LLMReply(str(result.get("result") or ""), "claude_cli", model, time.monotonic() - t0)
