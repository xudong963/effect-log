#!/usr/bin/env python3
"""
End-to-end Pydantic AI + effect-log example.

Setup:
    pip install ai-effectlog pydantic-ai

Set your API key:
    export OPENAI_API_KEY="sk-..."

This example defines each tool function ONCE and uses make_tooldefs() to create
EffectLog ToolDefs automatically — no duplicate definitions needed.

The send_email tool is classified as IrreversibleWrite — on recovery it will
NOT be re-executed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from pydantic_ai import Agent

from effect_log import EffectKind, EffectLog
from effect_log.middleware.pydantic_ai import effect_logged_agent, make_tooldefs

# -- Tool implementations (defined once) -------------------------------------

call_counts: dict[str, int] = {"lookup_order": 0, "send_email": 0}


def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    call_counts["lookup_order"] += 1
    data = {"order_id": order_id, "status": "shipped", "eta": "2024-03-20"}
    return str(data)


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification. This is irreversible."""
    call_counts["send_email"] += 1
    return f"Email sent to {to}: {subject}"


# -- Main --------------------------------------------------------------------


async def main():
    tmpdir = tempfile.mkdtemp()
    db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

    # 1. Create EffectLog ToolDefs from the same functions used as pydantic-ai tools
    tooldefs = make_tooldefs(
        [
            {"func": lookup_order, "effect": EffectKind.ReadOnly},
            {"func": send_email, "effect": EffectKind.IrreversibleWrite},
        ]
    )

    # 2. Build EffectLog with the generated ToolDefs
    log = EffectLog(execution_id="order-agent-run-1", tools=tooldefs, storage=db)

    # 3. Create a pydantic-ai agent with the same functions as tools
    agent = Agent(
        "openai:gpt-4o",
        system_prompt=(
            "You are a helpful order assistant. "
            "Look up orders and send email notifications about their status."
        ),
        tools=[lookup_order, send_email],
    )

    # 4. Wrap with effect-log middleware — all tool calls now go through the WAL
    agent = effect_logged_agent(
        log,
        agent,
        tool_effects={
            "lookup_order": EffectKind.ReadOnly,
            "send_email": EffectKind.IrreversibleWrite,
        },
    )

    # 5. Run the agent
    print("=== Running agent ===\n")
    result = await agent.run(
        "Look up order ORD-12345 and email the status to customer@example.com"
    )
    print(f"Agent output: {result.output}\n")

    # 6. Show effect-log history
    print("=== Effect-log history ===\n")
    for entry in log.history():
        print(
            f"  seq={entry['sequence']}  tool={entry['tool_name']:<15}  "
            f"effect={entry['effect_kind']:<20}  outcome={entry['outcome']}"
        )

    print(f"\n  lookup_order called: {call_counts['lookup_order']} time(s)")
    print(f"  send_email   called: {call_counts['send_email']} time(s)")

    # 7. Demonstrate recovery: re-create the log with recover=True,
    #    IrreversibleWrite tools return their sealed result without re-executing.
    print("\n=== Simulating recovery ===\n")
    for k in call_counts:
        call_counts[k] = 0

    tooldefs2 = make_tooldefs(
        [
            {"func": lookup_order, "effect": EffectKind.ReadOnly},
            {"func": send_email, "effect": EffectKind.IrreversibleWrite},
        ]
    )
    log2 = EffectLog(
        execution_id="order-agent-run-1", tools=tooldefs2, storage=db, recover=True
    )

    # Replay the same calls
    for entry in log.history():
        log2.execute(entry["tool_name"], {})

    print(
        f"  lookup_order re-executed: {call_counts['lookup_order']} (ReadOnly -> replayed)"
    )
    print(
        f"  send_email   re-executed: {call_counts['send_email']} (IrreversibleWrite -> SEALED)"
    )
    assert call_counts["send_email"] == 0, "send_email must not re-execute on recovery"
    print("\n  PASS: email not re-sent on recovery")


if __name__ == "__main__":
    asyncio.run(main())
