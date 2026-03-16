#!/usr/bin/env python3
"""
End-to-end Bub + effect-log example.

Setup:
    pip install ai-effectlog bub

This example demonstrates how effect-log integrates with Bub's tool system.
It uses Bub's real ToolExecutor and Tool classes (FileReadTool, RunCommandTool,
FileWriteTool) and wraps them with effect-log middleware for crash recovery.

RunCommandTool is classified as IrreversibleWrite — on recovery it will NOT
be re-executed. FileReadTool replays safely (ReadOnly).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from bub.agent.tools import Context, ToolRegistry, ToolResult
from bub.tools import FileReadTool, FileWriteTool, RunCommandTool

from effect_log import EffectKind, EffectLog
from effect_log.middleware.bub import (
    EffectLoggedToolExecutor,
    builtin_effects,
    make_tooldefs,
)

# -- Track real execution counts to verify recovery -------------------------

call_counts: dict[str, int] = {"read_file": 0, "run_command": 0, "write_file": 0}

# Monkey-patch execute methods to count calls
_orig_file_read_execute = FileReadTool.execute
_orig_run_command_execute = RunCommandTool.execute
_orig_file_write_execute = FileWriteTool.execute


def _counting_file_read(self, context):
    call_counts["read_file"] += 1
    return _orig_file_read_execute(self, context)


def _counting_run_command(self, context):
    call_counts["run_command"] += 1
    return _orig_run_command_execute(self, context)


def _counting_file_write(self, context):
    call_counts["write_file"] += 1
    return _orig_file_write_execute(self, context)


# -- Build a minimal ToolExecutor (no LLM needed) ---------------------------


class SimpleToolExecutor:
    """Minimal bub-compatible ToolExecutor for direct tool invocation."""

    def __init__(self, context: Context):
        self.context = context
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_default_tools()

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        tool_class = self.tool_registry.get_tool(tool_name)
        if not tool_class:
            return ToolResult(
                success=False, data=None, error=f"Tool not found: {tool_name}"
            )
        try:
            instance = tool_class(**kwargs)
            return instance.execute(self.context)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def extract_tool_calls(self, response: str) -> list:
        return []


# -- Main --------------------------------------------------------------------


def main():
    workspace = Path(tempfile.mkdtemp())
    db = f"sqlite:///{os.path.join(str(workspace), 'effects.db')}"

    # Create a test file for reading
    test_file = workspace / "config.toml"
    test_file.write_text('[deploy]\nregion = "us-east-1"\n')

    # 1. Create ToolDefs from bub's real tool classes
    tool_specs = [
        {"tool_class": FileReadTool, "effect": EffectKind.ReadOnly},
        {"tool_class": RunCommandTool, "effect": EffectKind.IrreversibleWrite},
        {"tool_class": FileWriteTool, "effect": EffectKind.IdempotentWrite},
    ]
    tooldefs = make_tooldefs(tool_specs)

    # 2. Build EffectLog
    log = EffectLog(execution_id="bub-deploy-run-1", tools=tooldefs, storage=db)

    # 3. Set up bub ToolExecutor + effect-log wrapper
    context = Context(workspace_path=workspace)
    executor = SimpleToolExecutor(context)

    tool_effects = builtin_effects()
    wrapped_executor = EffectLoggedToolExecutor(log, executor, tool_effects)

    # 4. Simulate agent workflow with real bub tools
    print("=== Bub agent workflow (real tools) ===\n")

    with (
        patch.object(FileReadTool, "execute", _counting_file_read),
        patch.object(RunCommandTool, "execute", _counting_run_command),
        patch.object(FileWriteTool, "execute", _counting_file_write),
    ):
        r1 = wrapped_executor.execute_tool("read_file", path=str(test_file))
        print(f"  [1] read_file -> success={r1.success}, data={str(r1.data)[:60]}")

        r2 = wrapped_executor.execute_tool("run_command", command="echo deployed")
        print(f"  [2] run_command -> success={r2.success}, data={str(r2.data)[:60]}")

        r3 = wrapped_executor.execute_tool(
            "write_file", path=str(workspace / "status.txt"), content="deployed"
        )
        print(f"  [3] write_file -> success={r3.success}, data={str(r3.data)[:60]}")

    # 5. Show effect-log history
    print("\n=== Effect-log history ===\n")
    for entry in log.history():
        print(
            f"  seq={entry['sequence']}  tool={entry['tool_name']:<15}  "
            f"effect={entry['effect_kind']:<20}  outcome={entry['outcome']}"
        )

    print(f"\n  read_file called: {call_counts['read_file']} time(s)")
    print(f"  run_command called: {call_counts['run_command']} time(s)")
    print(f"  write_file called: {call_counts['write_file']} time(s)")

    # 6. Simulate crash recovery
    print("\n=== Simulating crash recovery ===\n")
    print("  (Agent crashed after deploying. Restarting with same execution_id...)\n")

    for k in call_counts:
        call_counts[k] = 0

    tooldefs2 = make_tooldefs(tool_specs)
    log2 = EffectLog(
        execution_id="bub-deploy-run-1", tools=tooldefs2, storage=db, recover=True
    )

    executor2 = SimpleToolExecutor(context)
    wrapped_executor2 = EffectLoggedToolExecutor(log2, executor2, tool_effects)

    with (
        patch.object(FileReadTool, "execute", _counting_file_read),
        patch.object(RunCommandTool, "execute", _counting_run_command),
        patch.object(FileWriteTool, "execute", _counting_file_write),
    ):
        r1 = wrapped_executor2.execute_tool("read_file", path=str(test_file))
        print(f"  [1] read_file -> success={r1.success}, data={str(r1.data)[:60]}")

        r2 = wrapped_executor2.execute_tool("run_command", command="echo deployed")
        print(f"  [2] run_command -> success={r2.success}, data={str(r2.data)[:60]}")

        r3 = wrapped_executor2.execute_tool(
            "write_file", path=str(workspace / "status.txt"), content="deployed"
        )
        print(f"  [3] write_file -> success={r3.success}, data={str(r3.data)[:60]}")

    # 7. Verify recovery semantics
    print(
        f"\n  read_file re-executed: {call_counts['read_file']} (ReadOnly -> replayed)"
    )
    print(
        f"  run_command re-executed: {call_counts['run_command']} (IrreversibleWrite -> SEALED)"
    )
    print(
        f"  write_file re-executed: {call_counts['write_file']} (IdempotentWrite -> SEALED)"
    )

    assert call_counts["run_command"] == 0, (
        "run_command must not re-execute on recovery!"
    )
    assert call_counts["write_file"] == 0, "write_file must not re-execute on recovery!"
    print("\n  PASS: irreversible commands were NOT re-executed on recovery")


if __name__ == "__main__":
    main()
