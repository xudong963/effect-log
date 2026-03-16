"""Anthropic Claude API middleware for effect-log.

Wraps the tool execution loop for Claude's tool_use pattern so every tool
invocation goes through the effect-log WAL, gaining crash recovery and
duplicate prevention.

Usage:
    import anthropic
    from effect_log import EffectLog, EffectKind, ToolDef
    from effect_log.middleware.anthropic import effect_logged_tool_executor, make_tooldefs

    tool_specs = [
        {"func": search_db, "effect": EffectKind.ReadOnly},
        {"func": send_email, "effect": EffectKind.IrreversibleWrite},
    ]
    tools = make_tooldefs(tool_specs)
    log = EffectLog(execution_id="task-001", tools=tools, storage="sqlite:///effects.db")

    executor = effect_logged_tool_executor(log, {
        "search_db": search_db,
        "send_email": send_email,
    })

    # In the Claude API loop:
    for block in response.content:
        if block.type == "tool_use":
            result = executor(block.name, block.input, block.id)
"""

from __future__ import annotations

import json
from typing import Any, Callable


def _ensure_anthropic():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise ImportError(
            "Anthropic middleware requires the anthropic SDK. "
            "Install it with: pip install anthropic"
        )


def make_tooldefs(tool_specs):
    """Create ToolDef entries from raw functions for Anthropic tool_use.

    Anthropic's tool_use pattern uses raw functions, so this helper takes
    the same functions and produces EffectLog ToolDefs.

    Args:
        tool_specs: List of dicts with keys:
            - "func": A raw callable
            - "effect": The EffectKind for this tool

    Returns:
        List of ToolDef instances ready for EffectLog construction.
    """
    from effect_log import ToolDef

    defs = []
    for spec in tool_specs:
        fn, effect = spec["func"], spec["effect"]

        def adapted(args, _fn=fn):
            return _fn(**args)

        defs.append(ToolDef(fn.__name__, effect, adapted))
    return defs


def effect_logged_tool_executor(
    log: Any,
    tool_map: dict[str, Callable],
    tool_effects: dict[str, Any] | None = None,
) -> Callable:
    """Create a tool executor that routes Claude tool_use calls through effect-log.

    Returns a callable that takes (tool_name, tool_input, tool_use_id) and
    returns a tool_result content block dict suitable for passing back to
    the Claude API.

    Args:
        log: An initialized EffectLog instance (tools must already be registered).
        tool_map: Dict mapping tool name -> callable. Used for documentation;
                  actual execution goes through the EffectLog.
        tool_effects: Optional dict mapping tool name -> EffectKind.
                      For documentation only; tools must already be registered.

    Returns:
        A callable: (tool_name, tool_input, tool_use_id) -> tool_result dict.
    """

    def execute(tool_name: str, tool_input: dict, tool_use_id: str) -> dict:
        """Execute a tool through effect-log and return a tool_result block."""
        if tool_name not in tool_map:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                "is_error": True,
            }

        args = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
        result = log.execute(tool_name, args)
        content = json.dumps(result) if not isinstance(result, str) else result

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }

    return execute


def process_tool_calls(
    log: Any,
    tool_map: dict[str, Callable],
    response: Any,
) -> list[dict]:
    """Process all tool_use blocks from a Claude API response.

    Iterates over the response content, executes each tool_use block through
    effect-log, and returns the list of tool_result blocks to send back.

    Args:
        log: An initialized EffectLog instance.
        tool_map: Dict mapping tool name -> callable.
        response: A Claude API response (Message object with .content).

    Returns:
        List of tool_result dicts, one per tool_use block in the response.
    """
    executor = effect_logged_tool_executor(log, tool_map)
    results = []

    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            result = executor(block.name, block.input, block.id)
            results.append(result)

    return results
