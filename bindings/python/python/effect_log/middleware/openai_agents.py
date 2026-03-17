"""OpenAI Agents SDK middleware for effect-log.

Wraps FunctionTool instances so every invocation goes through the effect-log
WAL, gaining crash recovery and duplicate prevention.

Usage:
    from agents import Agent, Runner, function_tool
    from effect_log import EffectLog, EffectKind, ToolDef
    from effect_log.middleware.openai_agents import effect_logged_agent

    log = EffectLog(execution_id=run_id, tools=[...], storage="sqlite:///effects.db")

    # Wrap an existing agent's tools
    agent = effect_logged_agent(log, agent, tool_effects={
        "search_web": EffectKind.ReadOnly,
        "send_email": EffectKind.IrreversibleWrite,
    })
"""

from __future__ import annotations

import json
from typing import Any


def _ensure_openai_agents():
    try:
        from agents import FunctionTool  # noqa: F401
    except ImportError:
        raise ImportError(
            "OpenAI Agents middleware requires the openai-agents SDK. "
            "Install it with: pip install openai-agents"
        )


def make_tools(specs):
    """Create both FunctionTool and ToolDef entries from raw functions.

    Accepts raw callables (auto-classified) or dicts with explicit effects.

    Args:
        specs: List of:
            - callables (auto-classified), or
            - dicts with keys "func" and optional "effect" (EffectKind)

    Returns:
        Tuple of (list[FunctionTool], list[ToolDef]).
    """
    _ensure_openai_agents()
    from agents import function_tool as ft
    from effect_log import ToolDef
    from effect_log.classify import classify_effect_kind

    sdk_tools, tooldefs = [], []
    for spec in specs:
        if callable(spec):
            fn, effect = spec, None
        elif isinstance(spec, dict):
            fn = spec["func"]
            effect = spec.get("effect")
        else:
            raise TypeError(f"Expected callable or dict, got {type(spec).__name__}")

        if effect is None:
            effect = classify_effect_kind(fn).effect_kind

        sdk_tools.append(ft(fn))

        def adapted(args, _fn=fn):
            return _fn(**args)

        tooldefs.append(ToolDef(fn.__name__, effect, adapted))
    return sdk_tools, tooldefs


def wrap_function_tool(log: Any, tool: Any, effect_kind: Any = None) -> Any:
    """Wrap an OpenAI Agents SDK FunctionTool to route through effect-log.

    The wrapped tool intercepts on_invoke_tool, routes the call through
    log.execute(), and returns the sealed/fresh result.

    Args:
        log: An initialized EffectLog instance.
        tool: A FunctionTool from the OpenAI Agents SDK.
        effect_kind: Optional EffectKind (for documentation; the tool must
                     already be registered in the EffectLog).

    Returns:
        A new FunctionTool with the same schema but effect-logged execution.
    """
    _ensure_openai_agents()
    from agents import FunctionTool

    async def effect_logged_invoke(ctx, args_json: str) -> Any:
        tool_name = tool.name
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except (json.JSONDecodeError, TypeError):
            args = {"raw_input": str(args_json)}

        result = log.execute(
            tool_name, args if isinstance(args, dict) else {"input": args}
        )
        return json.dumps(result) if not isinstance(result, str) else result

    return FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=tool.params_json_schema,
        on_invoke_tool=effect_logged_invoke,
    )


def effect_logged_agent(
    log: Any,
    agent: Any,
    tool_effects: dict[str, Any] | None = None,
) -> Any:
    """Wrap all FunctionTools on an Agent to route through effect-log.

    Args:
        log: An initialized EffectLog instance.
        agent: An Agent from the OpenAI Agents SDK.
        tool_effects: Optional dict mapping tool name -> EffectKind.
                      Used for documentation only; tools must already be
                      registered in the EffectLog.

    Returns:
        The agent with its tools replaced by effect-logged wrappers.
    """
    _ensure_openai_agents()
    from agents import FunctionTool

    tool_effects = tool_effects or {}
    wrapped_tools = []

    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            effect = tool_effects.get(tool.name)
            wrapped_tools.append(wrap_function_tool(log, tool, effect))
        else:
            # Non-function tools (FileSearch, WebSearch, etc.) pass through
            wrapped_tools.append(tool)

    agent.tools = wrapped_tools
    return agent
