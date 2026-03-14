"""effect-log: Semantic side-effect tracking for AI agents."""

from effect_log.effect_log_native import EffectKind, EffectLog, ToolDef

__all__ = ["EffectKind", "EffectLog", "ToolDef", "tool", "middleware"]


def tool(effect: EffectKind, compensate=None):
    """Decorator to register a function as an effect-logged tool.

    Args:
        effect: The EffectKind classification for this tool.
        compensate: Optional compensation function for Compensatable effects.

    Returns:
        A ToolDef wrapping the function.
    """

    def decorator(func):
        return ToolDef(
            name=func.__name__,
            effect_kind=effect,
            func=func,
            compensate=compensate,
        )

    return decorator
