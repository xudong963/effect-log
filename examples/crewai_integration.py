#!/usr/bin/env python3
"""
CrewAI integration example for effect-log.

This example runs standalone. With crewai installed,
you can also use the middleware wrappers shown in comments.

    pip install effect-log crewai
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
    ToolDef("web_search", EffectKind.ReadOnly, counting("web_search")),
    ToolDef("send_slack", EffectKind.IrreversibleWrite, counting("send_slack")),
    ToolDef("deploy_service", EffectKind.IrreversibleWrite, counting("deploy_service")),
]

# ── With CrewAI middleware (requires crewai): ────────────────────────────────
#
#   from effect_log.middleware.crewai import effect_logged_crew
#
#   crew = effect_logged_crew(log, crew, tool_effects={
#       "web_search":     EffectKind.ReadOnly,
#       "send_slack":     EffectKind.IrreversibleWrite,
#       "deploy_service": EffectKind.IrreversibleWrite,
#   })
#   result = crew.kickoff()
#
# ─────────────────────────────────────────────────────────────────────────────

tmpdir = tempfile.mkdtemp()
db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

print("=== First execution (research + notify, then crash before deploy) ===\n")

log = EffectLog(execution_id="crew-demo", tools=tools, storage=db)
print(f"  1. web_search: {log.execute('web_search', {'query': 'service health'})}")
print(
    f"  2. send_slack: {log.execute('send_slack', {'channel': 'ops', 'message': 'Starting deploy'})}"
)
print(f"\n  *** CRASH ***  send_slack called {call_counts['send_slack']} time(s)\n")

# Reset
for k in call_counts:
    call_counts[k] = 0

tools2 = [
    ToolDef("web_search", EffectKind.ReadOnly, counting("web_search")),
    ToolDef("send_slack", EffectKind.IrreversibleWrite, counting("send_slack")),
    ToolDef("deploy_service", EffectKind.IrreversibleWrite, counting("deploy_service")),
]

print("=== Recovery ===\n")

log2 = EffectLog(execution_id="crew-demo", tools=tools2, storage=db, recover=True)
log2.execute("web_search", {"query": "service health"})
log2.execute("send_slack", {"channel": "ops", "message": "Starting deploy"})
log2.execute("deploy_service", {"service": "api-gw", "env": "prod"})

print(
    f"  web_search     re-executed: {call_counts['web_search']} (ReadOnly → replayed)"
)
print(
    f"  send_slack     re-executed: {call_counts['send_slack']} (IrreversibleWrite → SEALED)"
)
print(f"  deploy_service re-executed: {call_counts['deploy_service']} (new step)")

assert call_counts["send_slack"] == 0
print("\n  PASS: slack message not re-sent on recovery")
