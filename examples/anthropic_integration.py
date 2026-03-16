#!/usr/bin/env python3
"""
Anthropic Claude API integration example for effect-log.

This example runs standalone (no API key required). With the anthropic SDK
installed, you can also use the middleware wrappers shown in comments.

    pip install ai-effectlog anthropic
"""

import os
import tempfile

from effect_log import EffectKind, EffectLog, ToolDef

# ── Define tools with semantic classification ────────────────────────────────

call_counts = {}


def counting(name):
    call_counts[name] = 0

    def fn(args):
        call_counts[name] += 1
        return {"tool": name, "args": args, "call": call_counts[name]}

    return fn


tools = [
    ToolDef("search_db", EffectKind.ReadOnly, counting("search_db")),
    ToolDef("send_email", EffectKind.IrreversibleWrite, counting("send_email")),
    ToolDef("upsert_record", EffectKind.IdempotentWrite, counting("upsert_record")),
]

# ── With Anthropic middleware (requires anthropic SDK): ──────────────────────
#
#   from effect_log.middleware.anthropic import (
#       effect_logged_tool_executor,
#       make_tooldefs,
#       process_tool_calls,
#   )
#
#   tool_specs = [
#       {"func": search_db, "effect": EffectKind.ReadOnly},
#       {"func": send_email, "effect": EffectKind.IrreversibleWrite},
#   ]
#   tools = make_tooldefs(tool_specs)
#   log = EffectLog(execution_id="task-001", tools=tools, storage=db)
#   executor = effect_logged_tool_executor(log, {"search_db": search_db, ...})
#
#   # In the Claude API tool_use loop:
#   for block in response.content:
#       if block.type == "tool_use":
#           result = executor(block.name, block.input, block.id)
#
# ─────────────────────────────────────────────────────────────────────────────

tmpdir = tempfile.mkdtemp()
db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

print("=== First execution (search + email, then crash before upsert) ===\n")

log = EffectLog(execution_id="anthropic-demo", tools=tools, storage=db)

search_result = log.execute("search_db", {"query": "Q4 revenue"})
print(f"  1. search_db:     {search_result}")

email_result = log.execute(
    "send_email", {"to": "ceo@co.com", "subject": "Report", "body": "Q4 looks great"}
)
print(f"  2. send_email:    {email_result}")

print(f"\n  *** CRASH ***  send_email called {call_counts['send_email']} time(s)\n")

# Reset counters (simulating new process)
for k in call_counts:
    call_counts[k] = 0

tools2 = [
    ToolDef("search_db", EffectKind.ReadOnly, counting("search_db")),
    ToolDef("send_email", EffectKind.IrreversibleWrite, counting("send_email")),
    ToolDef("upsert_record", EffectKind.IdempotentWrite, counting("upsert_record")),
]

print("=== Recovery ===\n")

log2 = EffectLog(execution_id="anthropic-demo", tools=tools2, storage=db, recover=True)
log2.execute("search_db", {"query": "Q4 revenue"})
log2.execute(
    "send_email", {"to": "ceo@co.com", "subject": "Report", "body": "Q4 looks great"}
)
log2.execute(
    "upsert_record",
    {"table": "reports", "data": {"id": "Q4-2024", "status": "sent"}},
)

print(f"  search_db     re-executed: {call_counts['search_db']} (ReadOnly -> replayed)")
print(
    f"  send_email    re-executed: {call_counts['send_email']} (IrreversibleWrite -> SEALED)"
)
print(f"  upsert_record re-executed: {call_counts['upsert_record']} (new step)")

assert call_counts["send_email"] == 0, "send_email should NOT be re-executed!"
print("\n  PASS: email not re-sent on recovery")
