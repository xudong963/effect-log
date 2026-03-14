#!/usr/bin/env python3
"""
LangGraph integration example for effect-log.

This example runs standalone. With langchain-core/langgraph installed,
you can also use the middleware wrappers shown in comments.

    pip install effect-log langchain-core langgraph
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
    ToolDef("search_web", EffectKind.ReadOnly, counting("search_web")),
    ToolDef("send_email", EffectKind.IrreversibleWrite, counting("send_email")),
    ToolDef("upsert_record", EffectKind.IdempotentWrite, counting("upsert_record")),
]

# ── With LangGraph middleware (requires langchain-core): ─────────────────────
#
#   from effect_log.middleware.langgraph import EffectLogToolNode
#
#   tool_node = EffectLogToolNode(log, [
#       {"tool": search_tool, "effect": EffectKind.ReadOnly},
#       {"tool": email_tool,  "effect": EffectKind.IrreversibleWrite},
#   ])
#   graph.add_node("tools", tool_node)
#
# ─────────────────────────────────────────────────────────────────────────────

tmpdir = tempfile.mkdtemp()
db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

print("=== First execution (steps 1-3, then crash) ===\n")

log = EffectLog(execution_id="lg-demo", tools=tools, storage=db)
print(f"  1. search_web:    {log.execute('search_web', {'query': 'revenue'})}")
print(
    f"  2. send_email:    {log.execute('send_email', {'to': 'ceo@co.com', 'subject': 'Report'})}"
)
print(f"  3. upsert_record: {log.execute('upsert_record', {'id': 'r1', 'data': {}})}")
print(f"\n  *** CRASH ***  send_email called {call_counts['send_email']} time(s)\n")

# Reset counters (simulating new process)
for k in call_counts:
    call_counts[k] = 0

tools2 = [
    ToolDef("search_web", EffectKind.ReadOnly, counting("search_web")),
    ToolDef("send_email", EffectKind.IrreversibleWrite, counting("send_email")),
    ToolDef("upsert_record", EffectKind.IdempotentWrite, counting("upsert_record")),
]

print("=== Recovery ===\n")

log2 = EffectLog(execution_id="lg-demo", tools=tools2, storage=db, recover=True)
log2.execute("search_web", {"query": "revenue"})
log2.execute("send_email", {"to": "ceo@co.com", "subject": "Report"})
log2.execute("upsert_record", {"id": "r1", "data": {}})

print(f"  search_web  re-executed: {call_counts['search_web']} (ReadOnly → replayed)")
print(
    f"  send_email  re-executed: {call_counts['send_email']} (IrreversibleWrite → SEALED)"
)
print(
    f"  upsert_record re-executed: {call_counts['upsert_record']} (IdempotentWrite → sealed)"
)

assert call_counts["send_email"] == 0, "send_email should NOT be re-executed!"
print("\n  PASS: IrreversibleWrite was not re-sent on recovery")
