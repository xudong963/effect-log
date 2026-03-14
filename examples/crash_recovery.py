#!/usr/bin/env python3
"""
Phase 1 Milestone Demo: Crash Recovery with effect-log.

Demonstrates a 5-step agent task where:
- Step 3 is an IrreversibleWrite (send_email)
- The process "crashes" after step 3
- On recovery, step 3's sealed result is returned without re-execution
- Steps 4 and 5 execute normally
"""

import os
import tempfile

from effect_log import EffectKind, EffectLog, ToolDef

# --- Tool definitions ---

execution_counts = {}


def make_counting_tool(name, effect_kind):
    """Create a tool that counts how many times it's been called."""
    execution_counts[name] = 0

    def tool_fn(args):
        execution_counts[name] += 1
        return {
            "tool": name,
            "call_number": execution_counts[name],
            "args": args,
        }

    compensate = None
    if effect_kind == EffectKind.Compensatable:

        def _noop(args):
            return None

        compensate = _noop

    return ToolDef(name, effect_kind, tool_fn, compensate)


def main():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "agent_effects.db")
    storage = f"sqlite:///{db_path}"

    print("=" * 60)
    print("effect-log Phase 1 Milestone: Crash Recovery Demo")
    print("=" * 60)

    # Define tools
    tools = [
        make_counting_tool("fetch_data", EffectKind.ReadOnly),
        make_counting_tool("transform", EffectKind.ReadOnly),
        make_counting_tool("send_email", EffectKind.IrreversibleWrite),
        make_counting_tool("update_db", EffectKind.IdempotentWrite),
        make_counting_tool("log_result", EffectKind.ReadOnly),
    ]

    # === First execution: steps 1-3, then "crash" ===
    print("\n--- First Execution (steps 1-3, then crash) ---\n")

    log = EffectLog(execution_id="demo-001", tools=tools, storage=storage)

    r1 = log.execute("fetch_data", {"source": "https://api.example.com/data"})
    print(f"Step 1 [fetch_data]:  {r1}")

    r2 = log.execute("transform", {"data": r1, "format": "csv"})
    print(f"Step 2 [transform]:   {r2}")

    r3 = log.execute(
        "send_email",
        {"to": "ceo@company.com", "subject": "Weekly Report", "body": "See attached."},
    )
    print(f"Step 3 [send_email]:  {r3}")

    print("\n*** CRASH! Process dies after step 3 ***")
    print(f"    send_email executed {execution_counts['send_email']} time(s)")

    # === Recovery: resume execution ===
    print("\n--- Recovery (resuming from WAL) ---\n")

    # Reset counters to track re-execution
    for name in execution_counts:
        execution_counts[name] = 0

    # Recreate tools (simulating new process)
    tools2 = [
        make_counting_tool("fetch_data", EffectKind.ReadOnly),
        make_counting_tool("transform", EffectKind.ReadOnly),
        make_counting_tool("send_email", EffectKind.IrreversibleWrite),
        make_counting_tool("update_db", EffectKind.IdempotentWrite),
        make_counting_tool("log_result", EffectKind.ReadOnly),
    ]

    log2 = EffectLog(
        execution_id="demo-001",
        tools=tools2,
        storage=storage,
        recover=True,
    )

    r1 = log2.execute("fetch_data", {"source": "https://api.example.com/data"})
    print(
        f"Step 1 [fetch_data]:  REPLAYED (ReadOnly) -> executed={execution_counts['fetch_data']} time(s)"
    )

    r2 = log2.execute("transform", {"data": r1, "format": "csv"})
    print(
        f"Step 2 [transform]:   REPLAYED (ReadOnly) -> executed={execution_counts['transform']} time(s)"
    )

    r3 = log2.execute(
        "send_email",
        {"to": "ceo@company.com", "subject": "Weekly Report", "body": "See attached."},
    )
    print(
        f"Step 3 [send_email]:  SEALED (IrreversibleWrite) -> executed={execution_counts['send_email']} time(s)"
    )
    print(f"       Sealed result: {r3}")

    r4 = log2.execute("update_db", {"table": "reports", "status": "sent"})
    print(f"Step 4 [update_db]:   NEW EXECUTION -> {r4}")

    r5 = log2.execute("log_result", {"message": "Task completed successfully"})
    print(f"Step 5 [log_result]:  NEW EXECUTION -> {r5}")

    # === Verify ===
    print("\n--- Verification ---\n")
    print(f"send_email re-executions during recovery: {execution_counts['send_email']}")
    assert execution_counts["send_email"] == 0, "FAIL: send_email was re-executed!"
    print("PASS: IrreversibleWrite was NOT re-executed on recovery")

    history = log2.history()
    print(f"\nExecution history ({len(history)} steps):")
    for entry in history:
        print(
            f"  seq={entry['sequence']} tool={entry['tool_name']} "
            f"kind={entry['effect_kind']} outcome={entry['outcome']}"
        )

    print(f"\n{'=' * 60}")
    print("Phase 1 Milestone: PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
