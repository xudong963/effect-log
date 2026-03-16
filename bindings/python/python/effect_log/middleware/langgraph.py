"""LangGraph middleware for effect-log.

Wraps LangGraph tools so every invocation goes through the effect-log WAL,
gaining crash recovery and duplicate prevention automatically.

Usage:
    from effect_log import EffectLog, EffectKind
    from effect_log.middleware.langgraph import effect_logged_tools, EffectLogToolNode

    log = EffectLog(execution_id="task-001", tools=tools, storage="sqlite:///effects.db")

    # Option 1: Wrap existing LangChain tools (auto-classified or explicit)
    wrapped = effect_logged_tools(log, [search_tool, send_email_tool])
    wrapped = effect_logged_tools(log, [
        {"tool": search_tool, "effect": EffectKind.ReadOnly},
        {"tool": send_email_tool, "effect": EffectKind.IrreversibleWrite},
    ])
    tool_node = ToolNode(wrapped)

    # Option 2: Drop-in ToolNode replacement
    tool_node = EffectLogToolNode(log, [
        {"tool": search_tool, "effect": EffectKind.ReadOnly},
        {"tool": send_email_tool, "effect": EffectKind.IrreversibleWrite},
    ])
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from effect_log import EffectLog


def _ensure_langgraph():
    try:
        from langchain_core.tools import BaseTool  # noqa: F401
    except ImportError:
        raise ImportError(
            "LangGraph middleware requires langchain-core. "
            "Install it with: pip install langchain-core langgraph"
        )


class EffectLoggedTool:
    """A LangChain-compatible tool wrapper that routes calls through effect-log.

    This wraps any BaseTool (or @tool-decorated function) so that its
    invocations are logged with intent/completion records in the WAL.
    """

    def __init__(self, log: EffectLog, tool: Any, effect_kind_name: str):
        _ensure_langgraph()
        self._log = log
        self._inner = tool
        self._effect_kind_name = effect_kind_name
        # Expose the same interface LangGraph expects
        self.name = tool.name
        self.description = getattr(tool, "description", "")
        self.args_schema = getattr(tool, "args_schema", None)

    def invoke(self, input: dict | str, config: Any = None, **kwargs) -> str:
        """Synchronous invocation through effect-log."""
        if isinstance(input, str):
            try:
                input = json.loads(input)
            except json.JSONDecodeError:
                input = {"input": input}

        result = self._log.execute(self.name, input)
        return json.dumps(result) if not isinstance(result, str) else result

    async def ainvoke(self, input: dict | str, config: Any = None, **kwargs) -> str:
        """Async invocation (delegates to sync for now)."""
        return self.invoke(input, config, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


def effect_logged_tools(
    log: EffectLog,
    tool_specs: Sequence[Any],
) -> list[EffectLoggedTool]:
    """Wrap a list of LangChain tools with effect-log tracking.

    Accepts raw LangChain tools (auto-classified) or dicts with explicit effects.

    Args:
        log: An initialized EffectLog instance (tools must already be registered).
        tool_specs: List of:
            - LangChain BaseTool instances (auto-classified by name), or
            - dicts with keys "tool" and optional "effect" (EffectKind)

    Returns:
        List of EffectLoggedTool wrappers compatible with LangGraph's ToolNode.
    """
    _ensure_langgraph()
    wrapped = []
    for spec in tool_specs:
        if isinstance(spec, dict):
            tool = spec["tool"]
            effect = spec.get("effect")
        else:
            tool = spec
            effect = None

        if effect is None:
            from effect_log.classify import classify_from_name

            effect = classify_from_name(tool.name).effect_kind

        wrapped.append(EffectLoggedTool(log, tool, effect_kind_name=str(effect)))
    return wrapped


def make_tooldefs(tool_specs):
    """Create ToolDef entries from LangChain SDK tool objects.

    Accepts raw LangChain tools (auto-classified) or dicts with explicit effects.

    Args:
        tool_specs: List of:
            - LangChain BaseTool instances (auto-classified by name), or
            - dicts with keys "tool" and optional "effect" (EffectKind)

    Returns:
        List of ToolDef instances ready for EffectLog construction.
    """
    from effect_log import ToolDef
    from effect_log.classify import classify_from_name

    defs = []
    for spec in tool_specs:
        if isinstance(spec, dict):
            tool = spec["tool"]
            effect = spec.get("effect")
        else:
            tool = spec
            effect = None

        if effect is None:
            effect = classify_from_name(tool.name).effect_kind

        fn = getattr(tool, "func", None)
        if fn is not None:

            def adapted(args, _fn=fn):
                return _fn(**args)
        else:

            def adapted(args, _tool=tool):
                return _tool.invoke(args)

        defs.append(ToolDef(tool.name, effect, adapted))
    return defs


class EffectLogToolNode:
    """Drop-in replacement for LangGraph's ToolNode that routes through effect-log.

    Processes AIMessage tool_calls and returns ToolMessages with results.

    Usage in a StateGraph:
        graph.add_node("tools", EffectLogToolNode(log, tool_specs))
    """

    def __init__(
        self,
        log: EffectLog,
        tool_specs: Sequence[dict[str, Any]],
    ):
        _ensure_langgraph()
        self._log = log
        self._tools = {}
        for spec in tool_specs:
            tool = spec["tool"]
            self._tools[tool.name] = tool

    def __call__(self, state: dict) -> dict:
        """Process tool calls from the last AI message."""
        from langchain_core.messages import ToolMessage

        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        results = []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = (
                tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
            )

            result = self._log.execute(tool_name, tool_args)
            content = json.dumps(result) if not isinstance(result, str) else result

            results.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tc["id"],
                    name=tool_name,
                )
            )

        return {"messages": results}
