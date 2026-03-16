#!/usr/bin/env python3
"""
End-to-end Anthropic Claude API example with effect-log.

Demonstrates the full Claude tool_use loop with crash recovery.
Requires: pip install ai-effectlog anthropic
Requires: ANTHROPIC_API_KEY environment variable

Usage:
    ANTHROPIC_API_KEY=sk-... python examples/e2e_anthropic.py
"""

import os
import sys
import tempfile

try:
    import anthropic
except ImportError:
    print("This example requires the anthropic SDK: pip install anthropic")
    sys.exit(1)

from effect_log import EffectKind, EffectLog
from effect_log.middleware.anthropic import (
    effect_logged_tool_executor,
    make_tooldefs,
    process_tool_calls,
)

# ── Define tools ─────────────────────────────────────────────────────────────

call_counts = {"search_db": 0, "send_email": 0}


def search_db(query: str = "", limit: int = 5) -> dict:
    """Search the database for records matching the query."""
    call_counts["search_db"] += 1
    return {
        "results": [f"Record matching '{query}' #{i}" for i in range(1, limit + 1)],
        "total": limit,
    }


def send_email(to: str = "", subject: str = "", body: str = "") -> dict:
    """Send an email to the specified recipient."""
    call_counts["send_email"] += 1
    return {"sent": True, "to": to, "subject": subject, "message_id": "msg-001"}


# ── Claude API tool definitions ──────────────────────────────────────────────

claude_tools = [
    {
        "name": "search_db",
        "description": "Search the database for records matching a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

# ── Setup ─────────────────────────────────────────────────────────────────────

tmpdir = tempfile.mkdtemp()
db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

tool_map = {"search_db": search_db, "send_email": send_email}
tool_specs = [
    {"func": search_db, "effect": EffectKind.ReadOnly},
    {"func": send_email, "effect": EffectKind.IrreversibleWrite},
]
tooldefs = make_tooldefs(tool_specs)

# ── First execution ──────────────────────────────────────────────────────────

print("=== First execution ===\n")

log = EffectLog(execution_id="e2e-anthropic", tools=tooldefs, storage=db)
executor = effect_logged_tool_executor(log, tool_map)

client = anthropic.Anthropic()
messages = [
    {
        "role": "user",
        "content": "Search for Q4 revenue data, then email the results to ceo@co.com",
    }
]

# Claude API conversation loop
while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=claude_tools,
        messages=messages,
    )

    print(f"  Claude stop_reason: {response.stop_reason}")

    if response.stop_reason == "end_turn":
        # Extract final text
        for block in response.content:
            if hasattr(block, "text"):
                print(f"  Claude: {block.text}")
        break

    if response.stop_reason == "tool_use":
        # Process tool calls through effect-log
        messages.append({"role": "assistant", "content": response.content})

        tool_results = process_tool_calls(log, tool_map, response)
        for tr in tool_results:
            print(f"  Tool result ({tr['tool_use_id'][:8]}...): {tr['content'][:80]}")

        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  Unexpected stop_reason: {response.stop_reason}")
        break

print(f"\n  send_email called: {call_counts['send_email']} time(s)")

# ── Simulate crash and recovery ──────────────────────────────────────────────

print("\n=== Simulating crash and recovery ===\n")

# Reset call counts (new process)
call_counts["search_db"] = 0
call_counts["send_email"] = 0

# Create new EffectLog with recovery
tooldefs2 = make_tooldefs(tool_specs)
log2 = EffectLog(
    execution_id="e2e-anthropic", tools=tooldefs2, storage=db, recover=True
)
executor2 = effect_logged_tool_executor(log2, tool_map)

# Re-execute the same tool calls that Claude made
# In a real app, Claude would re-issue these; here we replay them directly
history = log2.history()
print(f"  WAL has {len(history)} entries from first execution")

for entry in history:
    tool_name = entry["tool_name"]
    # Re-execute through effect-log (will be sealed for IrreversibleWrite)
    result = executor2(tool_name, entry.get("input", {}), f"recovery-{tool_name}")
    print(f"  Recovered {tool_name}: is_error={result.get('is_error', False)}")

print(f"\n  search_db  re-executed: {call_counts['search_db']} (ReadOnly -> replayed)")
print(
    f"  send_email re-executed: {call_counts['send_email']} (IrreversibleWrite -> SEALED)"
)

assert call_counts["send_email"] == 0, (
    "FAIL: send_email should NOT be re-executed on recovery!"
)
print("\n  PASS: IrreversibleWrite was sealed on recovery")
