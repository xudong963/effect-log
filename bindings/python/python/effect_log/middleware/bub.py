"""Bub middleware for effect-log.

Wraps Bub's ToolExecutor so every tool invocation goes through the effect-log
WAL, gaining crash recovery and duplicate prevention.

Bub tools are Pydantic BaseModel subclasses with an execute(context) method.
The ToolExecutor instantiates tool classes with kwargs and calls execute().
This middleware intercepts that flow to route through EffectLog.

Usage:
    from bub.agent.core import Agent
    from effect_log import EffectLog, EffectKind, ToolDef
    from effect_log.middleware.bub import effect_logged_agent, make_tooldefs

    log = EffectLog(execution_id="task-001", tools=[...], storage="sqlite:///effects.db")

    # Wrap an agent's tool executor
    agent = effect_logged_agent(log, agent, tool_effects={
        "run_command": EffectKind.IrreversibleWrite,
        "file_read":   EffectKind.ReadOnly,
        "file_write":  EffectKind.IdempotentWrite,
        "file_edit":   EffectKind.IdempotentWrite,
    })
"""

from __future__ import annotations

import json
from typing import Any


def _ensure_bub():
    try:
        from bub.agent.tools import Tool, ToolResult  # noqa: F401
    except ImportError:
        raise ImportError(
            "Bub middleware requires bub. Install it with: pip install bub"
        )


class EffectLoggedToolExecutor:
    """A drop-in replacement for bub's ToolExecutor that routes calls through effect-log.

    Tools listed in ``tool_effects`` go through the WAL; all other tools
    pass through to the original executor unchanged.
    """

    def __init__(
        self,
        log: Any,
        original_executor: Any,
        tool_effects: dict[str, Any] | None = None,
    ):
        _ensure_bub()
        self._log = log
        self._inner = original_executor
        self._tool_effects = tool_effects or {}
        # Preserve the context reference
        self.context = original_executor.context
        self.tool_registry = original_executor.tool_registry

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool, routing through effect-log if it's in tool_effects."""
        from bub.agent.tools import ToolResult

        if tool_name in self._tool_effects:
            result = self._log.execute(tool_name, kwargs)
            # Convert effect-log result back to ToolResult for bub compatibility
            if isinstance(result, str):
                return ToolResult(success=True, data=result, error=None)
            return ToolResult(
                success=True,
                data=json.dumps(result) if not isinstance(result, str) else result,
                error=None,
            )

        # Not effect-logged — delegate to original executor
        return self._inner.execute_tool(tool_name, **kwargs)

    def extract_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Delegate to original executor's extraction logic."""
        return self._inner.extract_tool_calls(response)

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> str:
        """Execute tool calls through effect-log where applicable."""
        results: list[str] = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            if not tool_name:
                continue
            result = self.execute_tool(tool_name, **parameters)
            results.append(f"Observation: {result.format_result()}")
        return "\n".join(results) if results else "No tools executed."


def make_tooldefs(tool_specs, mode=None):
    """Create ToolDef entries from bub Tool classes.

    Accepts raw bub Tool classes (auto-classified) or dicts with explicit effects.

    Args:
        tool_specs: List of:
            - bub Tool classes (auto-classified by tool name), or
            - dicts with keys "tool_class" and optional "effect" (EffectKind)
        mode: Optional ClassifyMode for validation.

    Returns:
        List of ToolDef instances ready for EffectLog construction.
    """
    _ensure_bub()
    from bub.agent.tools import Context

    from effect_log import ClassifyMode
    from effect_log import ToolDef as ELToolDef
    from effect_log.classify import classify_from_name

    if mode is None:
        mode = ClassifyMode.HYBRID

    defs = []
    for spec in tool_specs:
        if isinstance(spec, dict):
            tool_class = spec["tool_class"]
            effect = spec.get("effect")
            if mode is ClassifyMode.AUTO and effect is not None:
                raise TypeError(
                    "In AUTO mode, specs must not include an explicit 'effect' key."
                )
            if mode is ClassifyMode.MANUAL and effect is None:
                raise TypeError(
                    "In MANUAL mode, all specs must include an explicit 'effect' key."
                )
        else:
            if mode is ClassifyMode.MANUAL:
                raise TypeError(
                    "In MANUAL mode, all specs must include an explicit 'effect' key."
                )
            tool_class = spec
            effect = None

        info = tool_class.get_tool_info()
        name = info["name"]

        if effect is None:
            effect = classify_from_name(name).effect_kind

        def adapted(args, _cls=tool_class):
            # Instantiate the tool with args, execute with a default context
            instance = _cls(**args)
            ctx = Context()
            result = instance.execute(ctx)
            if result.success:
                return (
                    result.data
                    if isinstance(result.data, str)
                    else json.dumps(result.data)
                )
            raise RuntimeError(result.error or "Tool execution failed")

        defs.append(ELToolDef(name, effect, adapted))
    return defs


def effect_logged_agent(
    log: Any,
    agent: Any,
    tool_effects: dict[str, Any] | None = None,
) -> Any:
    """Wrap a bub Agent's tool executor to route through effect-log.

    Replaces the agent's ToolExecutor with an EffectLoggedToolExecutor that
    intercepts execute_tool for tools listed in tool_effects.

    Args:
        log: An initialized EffectLog instance.
        agent: A bub Agent instance.
        tool_effects: Optional dict mapping tool name -> EffectKind.
                      Only tools in this mapping go through the WAL;
                      others pass through unchanged.

    Returns:
        The agent with its tool_executor replaced.
    """
    _ensure_bub()
    tool_effects = tool_effects or {}

    wrapper = EffectLoggedToolExecutor(log, agent.tool_executor, tool_effects)
    agent.tool_executor = wrapper
    return agent


# -- Suggested effect classifications for bub's builtin tools ----------------


def builtin_effects() -> dict[str, Any]:
    """Return recommended EffectKind for each bub builtin tool.

    Returns a dict mapping tool name -> EffectKind enum value, ready for
    use with effect_logged_agent() or EffectLoggedToolExecutor.

    Usage::

        from effect_log.middleware.bub import builtin_effects
        tool_effects = builtin_effects()
    """
    from effect_log import EffectKind

    return {
        "run_command": EffectKind.IrreversibleWrite,
        "read_file": EffectKind.ReadOnly,
        "write_file": EffectKind.IdempotentWrite,
        "edit_file": EffectKind.IdempotentWrite,
    }
