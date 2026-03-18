"""CrewAI middleware for effect-log.

Wraps CrewAI tools (both @tool-decorated and BaseTool subclasses) so every
invocation goes through the effect-log WAL.

Usage:
    from crewai import Agent, Task, Crew
    from effect_log import EffectLog, EffectKind, ToolDef
    from effect_log.middleware.crewai import effect_logged_tool, effect_logged_crew

    log = EffectLog(execution_id="task-001", tools=[...], storage="sqlite:///effects.db")

    # Option 1: Wrap individual tools
    search = effect_logged_tool(log, search_tool)

    # Option 2: Wrap all tools in a crew
    crew = effect_logged_crew(log, crew, tool_effects={
        "search": EffectKind.ReadOnly,
        "send_email": EffectKind.IrreversibleWrite,
    })
"""

from __future__ import annotations

import json
from typing import Any


def _ensure_crewai():
    try:
        from crewai.tools import BaseTool  # noqa: F401
    except ImportError:
        raise ImportError(
            "CrewAI middleware requires crewai. Install it with: pip install crewai"
        )


class EffectLoggedCrewAITool:
    """A CrewAI-compatible tool wrapper that routes calls through effect-log.

    Wraps any CrewAI BaseTool or @tool-decorated function, intercepting _run
    to go through the effect-log WAL.
    """

    def __init__(self, log: Any, tool: Any, effect_kind: Any = None):
        self._log = log
        self._inner = tool
        self.name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        self.description = getattr(tool, "description", "")
        self.args_schema = getattr(tool, "args_schema", None)

    def _run(self, **kwargs) -> str:
        """Execute through effect-log."""
        result = self._log.execute(self.name, kwargs)
        return json.dumps(result) if not isinstance(result, str) else result

    def run(self, *args, **kwargs) -> str:
        """Public run interface compatible with CrewAI's tool calling."""
        if args and isinstance(args[0], (dict, str)):
            input_data = args[0]
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    input_data = {"input": input_data}
            result = self._log.execute(self.name, input_data)
        else:
            result = self._log.execute(self.name, kwargs)
        return json.dumps(result) if not isinstance(result, str) else result

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def make_tooldefs(tool_specs, mode=None):
    """Create ToolDef entries from CrewAI SDK tool objects.

    Accepts raw LangChain/CrewAI tools (auto-classified) or dicts with explicit effects.

    Args:
        tool_specs: List of:
            - CrewAI BaseTool instances (auto-classified by name), or
            - dicts with keys "tool" and optional "effect" (EffectKind)
        mode: Optional ClassifyMode for validation.

    Returns:
        List of ToolDef instances ready for EffectLog construction.
    """
    from effect_log import ClassifyMode, ToolDef
    from effect_log.classify import classify_from_name

    if mode is None:
        mode = ClassifyMode.HYBRID

    defs = []
    for spec in tool_specs:
        if isinstance(spec, dict):
            tool = spec["tool"]
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
            tool = spec
            effect = None

        name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        if effect is None:
            effect = classify_from_name(name).effect_kind

        fn = getattr(tool, "func", None)
        if fn is not None:

            def adapted(args, _fn=fn):
                return _fn(**args)
        else:

            def adapted(args, _tool=tool):
                return _tool._run(**args)

        defs.append(ToolDef(name, effect, adapted))
    return defs


def effect_logged_tool(
    log: Any,
    tool: Any,
    effect_kind: Any = None,
) -> EffectLoggedCrewAITool:
    """Wrap a single CrewAI tool with effect-log tracking.

    Args:
        log: An initialized EffectLog instance.
        tool: A CrewAI BaseTool or @tool-decorated function.
        effect_kind: Optional EffectKind (for documentation).

    Returns:
        An EffectLoggedCrewAITool wrapper.
    """
    return EffectLoggedCrewAITool(log, tool, effect_kind)


def effect_logged_crew(
    log: Any,
    crew: Any,
    tool_effects: dict[str, Any] | None = None,
) -> Any:
    """Wrap all tools in a CrewAI Crew to route through effect-log.

    Iterates over all agents in the crew and replaces their tools with
    effect-logged wrappers.

    Args:
        log: An initialized EffectLog instance.
        crew: A CrewAI Crew instance.
        tool_effects: Optional dict mapping tool name -> EffectKind.

    Returns:
        The crew with all agent tools replaced by effect-logged wrappers.
    """
    _ensure_crewai()
    tool_effects = tool_effects or {}

    for agent in crew.agents:
        if hasattr(agent, "tools") and agent.tools:
            wrapped = []
            for tool in agent.tools:
                name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
                effect = tool_effects.get(name)
                wrapped.append(effect_logged_tool(log, tool, effect))
            agent.tools = wrapped

    return crew
