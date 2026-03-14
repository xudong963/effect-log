"""Type stubs for the native Rust extension module."""

from enum import IntEnum
from typing import Any, Callable

class EffectKind(IntEnum):
    ReadOnly = 0
    IdempotentWrite = 1
    Compensatable = 2
    IrreversibleWrite = 3
    ReadThenWrite = 4

class ToolDef:
    name: str
    effect_kind: EffectKind
    func: Callable[..., Any]
    compensate: Callable[..., Any] | None

    def __init__(
        self,
        name: str,
        effect_kind: EffectKind,
        func: Callable[..., Any],
        compensate: Callable[..., Any] | None = None,
    ) -> None: ...

class EffectLog:
    def __init__(
        self,
        execution_id: str,
        tools: list[ToolDef],
        storage: str = "memory",
        recover: bool = False,
    ) -> None: ...
    def execute(self, tool_name: str, args: dict[str, Any]) -> Any: ...
    def history(self) -> list[dict[str, Any]]: ...
