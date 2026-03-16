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
python examples/pydantic_ai_integration.py
python examples/anthropic_integration.py
```

| Example | Description |
|---|---|
| `crash_recovery.py` | A 5-step agent task that "crashes" after step 3, then recovers. Shows that `IrreversibleWrite` effects (e.g. `send_email`) are sealed and never re-executed on recovery. |
| `openai_agents_integration.py` | Standalone demo of effect-log with tools modeled after an OpenAI Agents workflow. |
| `crewai_integration.py` | Standalone demo with CrewAI-style tools (`web_search`, `send_slack`, `deploy_service`). |
| `langgraph_integration.py` | Standalone demo with LangGraph-style tools (`search_web`, `send_email`, `upsert_record`). |
| `pydantic_ai_integration.py` | Standalone demo with Pydantic AI-style tools (`search_db`, `send_email`, `upsert_record`). |
| `anthropic_integration.py` | Standalone demo with Anthropic Claude-style tools (`search_db`, `send_email`, `upsert_record`). |

## End-to-end examples (real SDK integration)

These examples import the real SDKs and use the effect-log middleware wrappers.
They require the corresponding SDK packages **and** the appropriate API key.

```bash
# OpenAI Agents SDK
export OPENAI_API_KEY="sk-..."
pip install ai-effectlog openai-agents
python examples/e2e_openai_agents.py

# CrewAI
export OPENAI_API_KEY="sk-..."
pip install ai-effectlog crewai crewai-tools
python examples/e2e_crewai.py

# LangGraph
export OPENAI_API_KEY="sk-..."
pip install ai-effectlog langchain-core langchain-openai langgraph
python examples/e2e_langgraph.py

# Pydantic AI
export OPENAI_API_KEY="sk-..."
pip install ai-effectlog pydantic-ai
python examples/e2e_pydantic_ai.py

# Anthropic Claude API
export ANTHROPIC_API_KEY="sk-..."
pip install ai-effectlog anthropic
python examples/e2e_anthropic.py

# Bub
pip install ai-effectlog bub
python examples/e2e_bub.py
```

| Example | SDK | Tools | Key demo |
|---|---|---|---|
| `e2e_openai_agents.py` | `openai-agents` | `get_weather` (ReadOnly), `send_alert` (IrreversibleWrite) | `effect_logged_agent()` wraps all FunctionTools on an Agent |
| `e2e_crewai.py` | `crewai` | `web_search` (ReadOnly), `send_report` (IrreversibleWrite) | `effect_logged_crew()` wraps all tools across a Crew's agents |
| `e2e_langgraph.py` | `langgraph` | `lookup_stock` (ReadOnly), `place_order` (IrreversibleWrite) | `EffectLogToolNode` is a drop-in replacement for LangGraph's `ToolNode` |
| `e2e_pydantic_ai.py` | `pydantic-ai` | `search_db` (ReadOnly), `send_email` (IrreversibleWrite) | `make_tooldefs()` auto-creates ToolDefs from Pydantic AI tool functions |
| `e2e_anthropic.py` | `anthropic` | `search_db` (ReadOnly), `send_email` (IrreversibleWrite) | `process_tool_calls()` handles Claude's `tool_use` loop with effect-log |
| `e2e_bub.py` | `bub` | `read_file` (ReadOnly), `run_command` (IrreversibleWrite), `write_file` (IdempotentWrite) | `EffectLoggedToolExecutor` wraps Bub's real tool classes |

Each e2e example also demonstrates recovery at the end: after the agent finishes,
it replays the same calls with `recover=True` and asserts that `IrreversibleWrite`
tools are **not** re-executed.
