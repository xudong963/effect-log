#!/usr/bin/env python3
"""
End-to-end CrewAI + effect-log example.

Setup:
    cd bindings/python && maturin develop   # install effect-log from source
    pip install crewai crewai-tools         # install the CrewAI SDK

Set your API key (CrewAI uses OpenAI by default):
    export OPENAI_API_KEY="sk-..."

This example defines each tool function ONCE using @crewai_tool and uses
make_tooldefs() to create EffectLog ToolDef entries automatically — no
duplicate definitions needed.

The send_report tool is classified as IrreversibleWrite — on recovery it will
NOT be re-executed.
"""

from __future__ import annotations

import os
import tempfile

from crewai import Agent, Crew, Task
from crewai.tools import tool as crewai_tool

from effect_log import EffectKind, EffectLog
from effect_log.middleware.crewai import effect_logged_crew, make_tooldefs

# -- Tool implementations (defined once) -------------------------------------

call_counts: dict[str, int] = {"web_search": 0, "send_report": 0}


@crewai_tool
def web_search(query: str) -> str:
    """Search the web for information."""
    call_counts["web_search"] += 1
    return f"Search results for '{query}': [Result 1: Latest findings..., Result 2: Key insights...]"


@crewai_tool
def send_report(summary: str) -> str:
    """Send a research report via email. This action is irreversible."""
    call_counts["send_report"] += 1
    return f"Report sent successfully: {summary[:80]}..."


# -- Main --------------------------------------------------------------------


def main():
    tmpdir = tempfile.mkdtemp()
    db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

    # 1. Create EffectLog ToolDefs from the same @crewai_tool functions
    tool_specs = [
        {"tool": web_search, "effect": EffectKind.ReadOnly},
        {"tool": send_report, "effect": EffectKind.IrreversibleWrite},
    ]
    tooldefs = make_tooldefs(tool_specs)

    # 2. Build EffectLog with the generated ToolDefs
    log = EffectLog(execution_id="crew-research-run-1", tools=tooldefs, storage=db)

    # 3. Create CrewAI agent, task, and crew (tools used directly)
    researcher = Agent(
        role="Research Analyst",
        goal="Find relevant information and send a concise report",
        backstory="You are a senior research analyst who excels at finding key insights.",
        tools=[web_search, send_report],
        verbose=True,
    )

    research_task = Task(
        description=(
            "Research the latest trends in AI agent frameworks. "
            "Search for information, then send a brief report summarizing your findings."
        ),
        expected_output="A confirmation that the report was sent.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    # 4. Wrap the entire crew with effect-log middleware
    crew = effect_logged_crew(log, crew)

    # 5. Run the crew
    print("=== Running CrewAI crew ===\n")
    result = crew.kickoff()
    print(f"\nCrew output: {result}\n")

    # 6. Show effect-log history
    print("=== Effect-log history ===\n")
    for entry in log.history():
        print(
            f"  seq={entry['sequence']}  tool={entry['tool_name']:<15}  "
            f"effect={entry['effect_kind']:<20}  outcome={entry['outcome']}"
        )

    print(f"\n  web_search  called: {call_counts['web_search']} time(s)")
    print(f"  send_report called: {call_counts['send_report']} time(s)")

    # 7. Demonstrate recovery
    print("\n=== Simulating recovery ===\n")
    for k in call_counts:
        call_counts[k] = 0

    tooldefs2 = make_tooldefs(tool_specs)
    log2 = EffectLog(
        execution_id="crew-research-run-1", tools=tooldefs2, storage=db, recover=True
    )

    # Replay the same calls
    for entry in log.history():
        log2.execute(entry["tool_name"], {})

    print(
        f"  web_search  re-executed: {call_counts['web_search']} (ReadOnly -> replayed)"
    )
    print(
        f"  send_report re-executed: {call_counts['send_report']} (IrreversibleWrite -> SEALED)"
    )
    assert call_counts["send_report"] == 0, (
        "send_report must not re-execute on recovery"
    )
    print("\n  PASS: report not re-sent on recovery")


if __name__ == "__main__":
    main()
