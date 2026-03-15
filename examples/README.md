# Examples

## Prerequisites

1. Python 3.8+
2. A Python virtual environment (the project already has one at `.venv/`)

## Setup

Install the `ai-effectlog` package from PyPI:

```bash
pip install ai-effectlog
```

If you're using the project's existing venv, activate it first:

```bash
source .venv/bin/activate
```

## Standalone examples (no SDK required)

These examples simulate tool calls directly — no API keys or extra packages needed:

```bash
python examples/crash_recovery.py
python examples/openai_agents_integration.py
python examples/crewai_integration.py
python examples/langgraph_integration.py
```

| Example | Description |
|---|---|
| `crash_recovery.py` | A 5-step agent task that "crashes" after step 3, then recovers. Shows that `IrreversibleWrite` effects (e.g. `send_email`) are sealed and never re-executed on recovery. |
| `openai_agents_integration.py` | Standalone demo of effect-log with tools modeled after an OpenAI Agents workflow. |
| `crewai_integration.py` | Standalone demo with CrewAI-style tools (`web_search`, `send_slack`, `deploy_service`). |
| `langgraph_integration.py` | Standalone demo with LangGraph-style tools (`search_web`, `send_email`, `upsert_record`). |

## End-to-end examples (real SDK integration)

These examples import the real SDKs and use the effect-log middleware wrappers.
They require the corresponding SDK packages **and** an `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY="sk-..."

# OpenAI Agents SDK
pip install ai-effectlog openai-agents
python examples/e2e_openai_agents.py

# CrewAI
pip install ai-effectlog crewai crewai-tools
python examples/e2e_crewai.py

# LangGraph
pip install ai-effectlog langchain-core langchain-openai langgraph
python examples/e2e_langgraph.py
```

| Example | SDK | Tools | Key demo |
|---|---|---|---|
| `e2e_openai_agents.py` | `openai-agents` | `get_weather` (ReadOnly), `send_alert` (IrreversibleWrite) | `effect_logged_agent()` wraps all FunctionTools on an Agent |
| `e2e_crewai.py` | `crewai` | `web_search` (ReadOnly), `send_report` (IrreversibleWrite) | `effect_logged_crew()` wraps all tools across a Crew's agents |
| `e2e_langgraph.py` | `langgraph` | `lookup_stock` (ReadOnly), `place_order` (IrreversibleWrite) | `EffectLogToolNode` is a drop-in replacement for LangGraph's `ToolNode` |

Each e2e example also demonstrates recovery at the end: after the agent finishes,
it replays the same calls with `recover=True` and asserts that `IrreversibleWrite`
tools are **not** re-executed.
