# effect-log

**Semantic side-effect tracking for AI agents.**

When an AI agent crashes mid-task, what happens on restart? Without effect-log, irreversible actions (emails, payments, deployments) get repeated. With effect-log, the system knows what completed and what didn't — it returns sealed results for finished work and resumes from where it left off.

## The Core Idea

Every tool declares its **effect kind** at registration time. This drives all recovery behavior:

| EffectKind | Recovery (completed) | Recovery (crashed) |
|---|---|---|
| **ReadOnly** | Replay for fresh data | Replay safely |
| **IdempotentWrite** | Return sealed result | Replay with same key |
| **Compensatable** | Return sealed result | Compensate, then replay |
| **IrreversibleWrite** | Return sealed result | **Escalate to human** |
| **ReadThenWrite** | Return sealed result | **Escalate to human** |

## Quick Start

```python
from effect_log import EffectKind, EffectLog, ToolDef

def send_email(args):
    return smtp.send(args["to"], args["subject"], args["body"])

tools = [
    ToolDef("read_file",  EffectKind.ReadOnly,         read_file),
    ToolDef("send_email", EffectKind.IrreversibleWrite, send_email),
    ToolDef("upsert",     EffectKind.IdempotentWrite,   upsert_record),
]

log = EffectLog(execution_id="task-001", tools=tools, storage="sqlite:///effects.db")
log.execute("read_file",  {"path": "/tmp/report.csv"})
log.execute("send_email", {"to": "ceo@co.com", "subject": "Report", "body": "..."})
log.execute("upsert",     {"id": "report-001", "data": data})
```

Recovery — just add `recover=True`, re-run the same steps:

```python
log = EffectLog(execution_id="task-001", tools=tools, storage="sqlite:///effects.db", recover=True)
log.execute("read_file",  {"path": "/tmp/report.csv"})  # Replayed (fresh data)
log.execute("send_email", {"to": "ceo@co.com", ...})    # Sealed — NOT re-sent
log.execute("upsert",     {"id": "report-001", ...})    # Replayed (idempotent)
```

## Framework Integration

Built-in middleware for major agent frameworks:

| Framework | Middleware | Entry Point |
|---|---|---|
| **LangGraph** | `effect_log.middleware.langgraph` | `EffectLogToolNode`, `effect_logged_tools` |
| **OpenAI Agents SDK** | `effect_log.middleware.openai_agents` | `effect_logged_agent`, `wrap_function_tool` |
| **CrewAI** | `effect_log.middleware.crewai` | `effect_logged_crew`, `effect_logged_tool` |

See [`examples/`](examples/) for runnable demos:

- [`crash_recovery.py`](examples/crash_recovery.py) — Core crash recovery demo (Phase 1 milestone)
- [`langgraph_integration.py`](examples/langgraph_integration.py) — LangGraph ToolNode + tool wrapping
- [`openai_agents_integration.py`](examples/openai_agents_integration.py) — OpenAI Agents SDK wrapping
- [`crewai_integration.py`](examples/crewai_integration.py) — CrewAI tool + crew wrapping

## How It Works

A write-ahead log with two record types:

1. **Intent** — written *before* execution (tool name, effect kind, input, cursor)
2. **Completion** — written *after* execution (outcome, sealed response)

Intent without completion = crash detected. The recovery engine uses the effect kind to decide: replay, return sealed, compensate, or escalate.

## Building from Source

```bash
# Rust
cargo build --release
cargo test --workspace --all-features

# Python
cd bindings/python
pip install maturin
maturin develop --all-features
pytest tests/ -v
```

## Roadmap

- [x] Core library — WAL engine, recovery engine, SQLite + in-memory backends
- [x] Python bindings — PyO3 + maturin
- [x] Framework middleware — LangGraph, OpenAI Agents SDK, CrewAI
- [ ] TypeScript bindings — napi-rs, Vercel AI SDK
- [ ] Additional backends — RocksDB, S3, Restate journal
- [ ] Auto-classification — infer effect kind from HTTP methods / API metadata

## License

MIT OR Apache-2.0
