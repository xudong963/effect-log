"""Pydantic AI middleware for effect-log.

Wraps pydantic-ai toolsets so every tool invocation goes through the effect-log
WAL, gaining crash recovery and duplicate prevention.

Usage:
    from pydantic_ai import Agent
    from effect_log import EffectLog, EffectKind, ToolDef
    from effect_log.middleware.pydantic_ai import effect_logged_agent

    log = EffectLog(execution_id=run_id, tools=[...], storage="sqlite:///effects.db")

    # Wrap an existing agent's toolset
    agent = effect_logged_agent(log, agent, tool_effects={
        "search_web": EffectKind.ReadOnly,
        "send_email": EffectKind.IrreversibleWrite,
    })
"""

from __future__ import annotations

import json
from typing import Any


def _ensure_pydantic_ai():
    try:
        from pydantic_ai.toolsets import WrapperToolset  # noqa: F401
    except ImportError:
        raise ImportError(
            "Pydantic AI middleware requires pydantic-ai. "
            "Install it with: pip install pydantic-ai"
        )


class EffectLogToolset:
    """A pydantic-ai WrapperToolset subclass that routes calls through effect-log.

    Tools listed in ``tool_effects`` go through the WAL; all other tools
    pass through to the wrapped toolset unchanged.
    """

    def __init__(
        self, log: Any, wrapped: Any, tool_effects: dict[str, Any] | None = None
    ):
        _ensure_pydantic_ai()
        from pydantic_ai.toolsets import WrapperToolset

        self._log = log
        self._tool_effects = tool_effects or {}

        # Dynamically create a WrapperToolset subclass with our call_tool override
        outer = self

        class _EffectLogWrapperToolset(WrapperToolset):
            async def call_tool(
                self_inner, name: str, tool_args: dict[str, Any], ctx: Any, tool: Any
            ) -> Any:
                if name in outer._tool_effects:
                    args = (
                        tool_args
                        if isinstance(tool_args, dict)
                        else {"input": tool_args}
                    )
                    result = outer._log.execute(name, args)
                    return json.dumps(result) if not isinstance(result, str) else result
                return await super().call_tool(name, tool_args, ctx, tool)

        self._wrapper = _EffectLogWrapperToolset(wrapped)

    @property
    def toolset(self):
        """Return the underlying WrapperToolset instance."""
        return self._wrapper


def make_tooldefs(tool_specs):
    """Create ToolDef entries from raw functions for pydantic-ai tools.

    Pydantic-ai accepts raw functions directly as tools, so this helper
    takes the same functions and produces EffectLog ToolDefs — no duplicate
    definitions needed.

    Args:
        tool_specs: List of dicts with keys:
            - "func": A raw callable (the same function passed to pydantic-ai)
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


def effect_logged_agent(
    log: Any,
    agent: Any,
    tool_effects: dict[str, Any] | None = None,
) -> Any:
    """Wrap a pydantic-ai Agent's toolset to route through effect-log.

    Replaces the agent's internal toolset with an EffectLogToolset that
    intercepts call_tool for tools listed in tool_effects.

    Args:
        log: An initialized EffectLog instance.
        agent: A pydantic-ai Agent instance.
        tool_effects: Optional dict mapping tool name → EffectKind.
                      Only tools in this mapping go through the WAL;
                      others pass through unchanged.

    Returns:
        The agent with its toolset replaced by an effect-logged wrapper.
    """
    _ensure_pydantic_ai()
    tool_effects = tool_effects or {}

    wrapper = EffectLogToolset(log, agent._toolset, tool_effects)
    agent._toolset = wrapper.toolset
    return agent
