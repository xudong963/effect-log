"""effect-log: Semantic side-effect tracking for AI agents."""

from enum import Enum

from effect_log.effect_log_native import (
    EffectKind,
    EffectLog as _NativeEffectLog,
    ToolDef,
)
from effect_log.classify import (
    classify_effect_kind,
    classify_from_name,
    classify_tools,
    classify_with_llm,
)


class ClassifyMode(Enum):
    """Controls how tools are classified in an EffectLog.

    AUTO    — all tools are raw callables; effect kinds are inferred automatically.
    MANUAL  — all tools are explicit ToolDef instances; no auto-classification.
    HYBRID  — mixed: callables are auto-classified, ToolDefs pass through as-is.
    """

    AUTO = "auto"
    MANUAL = "manual"
    HYBRID = "hybrid"


__all__ = [
    "ClassifyMode",
    "EffectKind",
    "EffectLog",
    "ToolDef",
    "tool",
    "auto_tool",
    "classify_tools",
    "classify_effect_kind",
    "classify_from_name",
    "classify_with_llm",
    "middleware",
]


def _wrap_callable(func):
    """Wrap a bare callable so it receives **kwargs from args dict."""

    def adapted(args, _fn=func):
        return _fn(**args)

    return adapted


class EffectLog:
    """Python wrapper around _NativeEffectLog with auto-classification.

    Accepts raw callables alongside ToolDef instances. Raw callables are
    auto-classified using heuristic analysis. Use overrides= to correct
    any misclassifications.

    Args:
        execution_id: Unique identifier for this execution.
        tools: List of ToolDef instances, callables, or a mix.
        storage: Storage backend ("memory" or "sqlite:///path").
        recover: Whether to recover from a previous execution.
        overrides: Optional dict mapping function name -> EffectKind
                   to override auto-classification.
        mode: ClassifyMode controlling validation (default HYBRID).
    """

    def __init__(
        self,
        execution_id: str,
        tools: list,
        storage: str = "memory",
        recover: bool = False,
        overrides: dict[str, EffectKind] | None = None,
        mode: ClassifyMode = ClassifyMode.HYBRID,
    ):
        if mode is ClassifyMode.MANUAL and overrides:
            raise ValueError(
                "overrides= is not supported in MANUAL mode. "
                "Remove overrides or use HYBRID mode."
            )

        overrides = overrides or {}
        processed = []
        for t in tools:
            if isinstance(t, ToolDef):
                if mode is ClassifyMode.AUTO:
                    raise TypeError(
                        "In AUTO mode, pass raw callables instead of ToolDef. "
                        "Use MANUAL or HYBRID mode for explicit ToolDef."
                    )
                processed.append(t)
            elif callable(t):
                if mode is ClassifyMode.MANUAL:
                    raise TypeError(
                        "In MANUAL mode, all tools must be ToolDef instances. "
                        "Use EffectLog.auto() or HYBRID mode for raw callables."
                    )
                name = getattr(t, "__name__", str(t))
                kind = overrides.get(name)
                if kind is None:
                    kind = classify_effect_kind(t, name).effect_kind
                processed.append(ToolDef(name, kind, _wrap_callable(t)))
            else:
                raise TypeError(f"Expected ToolDef or callable, got {type(t).__name__}")
        self._inner = _NativeEffectLog(execution_id, processed, storage, recover)

    @classmethod
    def auto(
        cls,
        execution_id: str,
        tools: list,
        storage: str = "memory",
        recover: bool = False,
        overrides: dict[str, EffectKind] | None = None,
    ) -> "EffectLog":
        """Create an EffectLog in AUTO mode — all tools are raw callables."""
        return cls(
            execution_id=execution_id,
            tools=tools,
            storage=storage,
            recover=recover,
            overrides=overrides,
            mode=ClassifyMode.AUTO,
        )

    @classmethod
    def manual(
        cls,
        execution_id: str,
        tools: list,
        storage: str = "memory",
        recover: bool = False,
    ) -> "EffectLog":
        """Create an EffectLog in MANUAL mode — all tools must be ToolDef."""
        return cls(
            execution_id=execution_id,
            tools=tools,
            storage=storage,
            recover=recover,
            mode=ClassifyMode.MANUAL,
        )

    def execute(self, tool_name: str, args: dict):
        """Execute a tool through the effect-log WAL."""
        return self._inner.execute(tool_name, args)

    def history(self) -> list[dict]:
        """Get execution history."""
        return self._inner.history()


def tool(effect=None, compensate=None):
    """Decorator to register a function as an effect-logged tool.

    Supports both ``@tool`` (no parens) and ``@tool()`` / ``@tool(EffectKind.X)``.

    Args:
        effect: The EffectKind classification. If None, auto-classified.
        compensate: Optional compensation function for Compensatable effects.

    Returns:
        A ToolDef wrapping the function.
    """
    # Handle @tool without parens: effect will be the decorated function
    if callable(effect):
        return auto_tool(effect)

    def decorator(func):
        kind = effect
        if kind is None:
            kind = classify_effect_kind(func).effect_kind
        return ToolDef(
            name=func.__name__,
            effect_kind=kind,
            func=func,
            compensate=compensate,
        )

    return decorator


def auto_tool(func):
    """Convenience decorator: auto-classifies effect kind from function metadata.

    Equivalent to @tool() with no arguments.

    Usage:
        @auto_tool
        def search_db(args):
            return db.query(args["query"])
    """
    kind = classify_effect_kind(func).effect_kind
    return ToolDef(
        name=func.__name__,
        effect_kind=kind,
        func=func,
    )
