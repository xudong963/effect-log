#!/usr/bin/env python3
"""
OpenAI Agents SDK integration example for effect-log.

This example runs standalone. With openai-agents installed,
you can also use the middleware wrappers shown in comments.

    pip install effect-log openai-agents
"""

import os
import tempfile

from effect_log import EffectKind, EffectLog, ToolDef

# ── Define tools ─────────────────────────────────────────────────────────────

call_counts = {}


def counting(name):
    call_counts[name] = 0

    def fn(args):
        call_counts[name] += 1
        return {"tool": name, "args": args, "call": call_counts[name]}

    return fn


tools = [
    ToolDef("search", EffectKind.ReadOnly, counting("search")),
    ToolDef("send_email", EffectKind.IrreversibleWrite, counting("send_email")),
    ToolDef("create_jira", EffectKind.IdempotentWrite, counting("create_jira")),
]

# ── With OpenAI Agents SDK middleware (requires openai-agents): ──────────────
#
#   from effect_log.middleware.openai_agents import effect_logged_agent
#
#   agent = effect_logged_agent(log, agent, tool_effects={
#       "search":     EffectKind.ReadOnly,
#       "send_email": EffectKind.IrreversibleWrite,
#   })
#   result = await Runner.run(agent, "Send the weekly report")
#
# ─────────────────────────────────────────────────────────────────────────────

tmpdir = tempfile.mkdtemp()
db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

print("=== First execution (2 steps, then crash) ===\n")

log = EffectLog(execution_id="oai-demo", tools=tools, storage=db)
print(f"  1. search:     {log.execute('search', {'query': 'Q4 results'})}")
print(
    f"  2. send_email: {log.execute('send_email', {'to': 'team@co.com', 'subject': 'Q4'})}"
)
print(f"\n  *** CRASH ***  send_email called {call_counts['send_email']} time(s)\n")

# Reset
for k in call_counts:
    call_counts[k] = 0

tools2 = [
    ToolDef("search", EffectKind.ReadOnly, counting("search")),
    ToolDef("send_email", EffectKind.IrreversibleWrite, counting("send_email")),
    ToolDef("create_jira", EffectKind.IdempotentWrite, counting("create_jira")),
]

print("=== Recovery ===\n")

log2 = EffectLog(execution_id="oai-demo", tools=tools2, storage=db, recover=True)
log2.execute("search", {"query": "Q4 results"})
log2.execute("send_email", {"to": "team@co.com", "subject": "Q4"})
log2.execute("create_jira", {"title": "Follow up on Q4"})

print(f"  search      re-executed: {call_counts['search']} (ReadOnly → replayed)")
print(
    f"  send_email  re-executed: {call_counts['send_email']} (IrreversibleWrite → SEALED)"
)
print(f"  create_jira re-executed: {call_counts['create_jira']} (new step)")

assert call_counts["send_email"] == 0
print("\n  PASS: email not re-sent on recovery")
