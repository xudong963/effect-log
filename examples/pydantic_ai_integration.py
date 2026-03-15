#!/usr/bin/env python3
"""
Pydantic AI integration example for effect-log.

This example runs standalone. With pydantic-ai installed,
you can also use the middleware wrappers shown in comments.

    pip install ai-effectlog pydantic-ai
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
    ToolDef("fetch_report", EffectKind.ReadOnly, counting("fetch_report")),
    ToolDef("send_email", EffectKind.IrreversibleWrite, counting("send_email")),
    ToolDef("upsert_db", EffectKind.IdempotentWrite, counting("upsert_db")),
]

# ── With pydantic-ai middleware (requires pydantic-ai): ───────────────────────
#
#   from effect_log.middleware.pydantic_ai import effect_logged_agent, make_tooldefs
#
#   agent = effect_logged_agent(log, agent, tool_effects={
#       "fetch_report": EffectKind.ReadOnly,
#       "send_email":   EffectKind.IrreversibleWrite,
#       "upsert_db":    EffectKind.IdempotentWrite,
#   })
#   result = await agent.run("Fetch the report and send it via email")
#
# ─────────────────────────────────────────────────────────────────────────────

tmpdir = tempfile.mkdtemp()
db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

print("=== First execution (fetch + email, then crash before upsert) ===\n")

log = EffectLog(execution_id="pydantic-ai-demo", tools=tools, storage=db)

report = log.execute("fetch_report", {"report_id": "Q4-2024"})
print(f"  1. fetch_report: {report}")

email_result = log.execute(
    "send_email", {"to": "team@company.com", "body": f"Report: {report}"}
)
print(f"  2. send_email: {email_result}")

print(f"\n  *** CRASH ***  send_email called {call_counts['send_email']} time(s)\n")

# Reset
for k in call_counts:
    call_counts[k] = 0

tools2 = [
    ToolDef("fetch_report", EffectKind.ReadOnly, counting("fetch_report")),
    ToolDef("send_email", EffectKind.IrreversibleWrite, counting("send_email")),
    ToolDef("upsert_db", EffectKind.IdempotentWrite, counting("upsert_db")),
]

print("=== Recovery ===\n")

log2 = EffectLog(
    execution_id="pydantic-ai-demo", tools=tools2, storage=db, recover=True
)
log2.execute("fetch_report", {"report_id": "Q4-2024"})
log2.execute("send_email", {"to": "team@company.com", "body": f"Report: {report}"})
log2.execute(
    "upsert_db", {"table": "reports", "data": {"id": "Q4-2024", "status": "sent"}}
)

print(
    f"  fetch_report   re-executed: {call_counts['fetch_report']} (ReadOnly → replayed)"
)
print(
    f"  send_email     re-executed: {call_counts['send_email']} (IrreversibleWrite → SEALED)"
)
print(f"  upsert_db      re-executed: {call_counts['upsert_db']} (new step)")

assert call_counts["send_email"] == 0
print("\n  PASS: email not re-sent on recovery")
