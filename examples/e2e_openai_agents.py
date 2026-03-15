#!/usr/bin/env python3
"""
End-to-end OpenAI Agents SDK + effect-log example.

Setup:
    pip install ai-effectlog openai-agents

Set your API key:
    export OPENAI_API_KEY="sk-..."

This example defines each tool function ONCE and uses make_tools() to create
both the SDK FunctionTool and the EffectLog ToolDef automatically — no
duplicate definitions needed.

The send_alert tool is classified as IrreversibleWrite — on recovery it will
NOT be re-executed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from agents import Agent, Runner

from effect_log import EffectKind, EffectLog
from effect_log.middleware.openai_agents import effect_logged_agent, make_tools

# -- Tool implementations (defined once) -------------------------------------

call_counts: dict[str, int] = {"get_weather": 0, "send_alert": 0}


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    call_counts["get_weather"] += 1
    data = {"city": city, "temp_f": 72, "condition": "sunny"}
    return str(data)


def send_alert(message: str) -> str:
    """Send a weather alert notification. This is irreversible."""
    call_counts["send_alert"] += 1
    return f"Alert sent: {message}"


# -- Main --------------------------------------------------------------------


async def main():
    tmpdir = tempfile.mkdtemp()
    db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

    # 1. Create SDK tools + EffectLog ToolDefs from the same functions
    sdk_tools, tooldefs = make_tools(
        [
            {"func": get_weather, "effect": EffectKind.ReadOnly},
            {"func": send_alert, "effect": EffectKind.IrreversibleWrite},
        ]
    )

    # 2. Build EffectLog with the generated ToolDefs
    log = EffectLog(execution_id="weather-agent-run-1", tools=tooldefs, storage=db)

    # 3. Create an OpenAI Agents SDK agent with the generated SDK tools
    agent = Agent(
        name="Weather Assistant",
        instructions=(
            "You are a helpful weather assistant. "
            "Look up the weather and send an alert if it's noteworthy."
        ),
        tools=sdk_tools,
    )

    # 4. Wrap with effect-log middleware — all tool calls now go through the WAL
    agent = effect_logged_agent(log, agent)

    # 5. Run the agent
    print("=== Running agent ===\n")
    result = await Runner.run(
        agent, "Check the weather in San Francisco and send an alert about it."
    )
    print(f"Agent output: {result.final_output}\n")

    # 6. Show effect-log history
    print("=== Effect-log history ===\n")
    for entry in log.history():
        print(
            f"  seq={entry['sequence']}  tool={entry['tool_name']:<15}  "
            f"effect={entry['effect_kind']:<20}  outcome={entry['outcome']}"
        )

    print(f"\n  get_weather called: {call_counts['get_weather']} time(s)")
    print(f"  send_alert  called: {call_counts['send_alert']} time(s)")

    # 7. Demonstrate recovery: re-create the log with recover=True,
    #    IrreversibleWrite tools return their sealed result without re-executing.
    print("\n=== Simulating recovery ===\n")
    for k in call_counts:
        call_counts[k] = 0

    _, tooldefs2 = make_tools(
        [
            {"func": get_weather, "effect": EffectKind.ReadOnly},
            {"func": send_alert, "effect": EffectKind.IrreversibleWrite},
        ]
    )
    log2 = EffectLog(
        execution_id="weather-agent-run-1", tools=tooldefs2, storage=db, recover=True
    )

    # Replay the same calls
    for entry in log.history():
        log2.execute(entry["tool_name"], {})

    print(
        f"  get_weather re-executed: {call_counts['get_weather']} (ReadOnly -> replayed)"
    )
    print(
        f"  send_alert  re-executed: {call_counts['send_alert']} (IrreversibleWrite -> SEALED)"
    )
    assert call_counts["send_alert"] == 0, "send_alert must not re-execute on recovery"
    print("\n  PASS: alert not re-sent on recovery")


if __name__ == "__main__":
    asyncio.run(main())
