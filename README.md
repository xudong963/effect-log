# effect-log

**Semantic side-effect tracking for AI agents.**

When an AI agent crashes mid-task, what happens on restart? Without effect-log, irreversible actions (emails, payments, deployments) get repeated. With effect-log, the system knows what completed and what didn't — it returns sealed results for finished work and resumes from where it left off.

## The Core Idea

Every tool has an **effect kind** that drives all recovery behavior:

| EffectKind | Recovery (completed) | Recovery (crashed) |
|---|---|---|
| **ReadOnly** | Replay for fresh data | Replay safely |
| **IdempotentWrite** | Return sealed result | Replay with same key |
| **Compensatable** | Return sealed result | Compensate, then replay |
| **IrreversibleWrite** | Return sealed result | **Escalate to human** |
| **ReadThenWrite** | Return sealed result | **Escalate to human** |

## Quick Start

### Auto mode — just pass functions

```python
from effect_log import EffectLog

log = EffectLog.auto("task-001", tools=[search_db, send_email, upsert_record])
log.execute("search_db", {"query": "Q4 revenue"})     # auto → ReadOnly
log.execute("send_email", {"to": "ceo@co.com", ...})  # auto → IrreversibleWrite
log.execute("upsert_record", {"id": "r-001", ...})    # auto → IdempotentWrite
```

### Manual mode — explicit ToolDef for full control

```python
from effect_log import EffectKind, EffectLog, ToolDef

tools = [
    ToolDef("read_file",  EffectKind.ReadOnly,         read_file),
    ToolDef("send_email", EffectKind.IrreversibleWrite, send_email),
    ToolDef("upsert",     EffectKind.IdempotentWrite,   upsert_record),
]
log = EffectLog.manual("task-001", tools=tools, storage="sqlite:///effects.db")
```

### Recovery — just add `recover=True`

```python
log = EffectLog.auto("task-001", tools=[search_db, send_email, upsert_record],
                     storage="sqlite:///effects.db", recover=True)
log.execute("search_db", {"query": "Q4 revenue"})     # Replayed (fresh data)
log.execute("send_email", {"to": "ceo@co.com", ...})  # Sealed — NOT re-sent
log.execute("upsert_record", {"id": "r-001", ...})    # Replayed (idempotent)
```

### Override when needed

If the classifier gets something wrong, override just that tool:

```python
from effect_log import EffectKind, EffectLog

log = EffectLog.auto("task-001",
    tools=[search_db, send_email, process_order],
    overrides={"process_order": EffectKind.IdempotentWrite}
)
```

### Inspect classifications

```python
from effect_log import classify_tools

report = classify_tools([search_db, send_email, process_order])
print(report)
# search_db      -> ReadOnly              (0.50)  name
# send_email     -> IrreversibleWrite     (0.50)  name
# process_order  -> IrreversibleWrite     (0.00)  default (no signals)!!!

# Apply with corrections
tools = report.apply(overrides={"process_order": EffectKind.IdempotentWrite})
log = EffectLog("task-001", tools=tools)
```

### Hybrid mode (default constructor)

The default constructor accepts a mix of callables and ToolDefs:

```python
from effect_log import EffectLog, ToolDef, EffectKind

log = EffectLog("task-001", tools=[
    search_db,                                               # auto-classified
    ToolDef("send_email", EffectKind.IrreversibleWrite, send_email),  # explicit
])
```

### Decorators

```python
from effect_log import tool, auto_tool

@tool(effect=EffectKind.ReadOnly)       # explicit
def read_file(args): ...

@tool()                                  # auto-classified
def search_db(args): ...

@auto_tool                               # shorthand for @tool()
def fetch_data(args): ...
```

## Auto-Classification

effect-log classifies tools using a 4-layer weighted heuristic:

| Layer | Signal | Weight | Example |
|---|---|---|---|
| **Name prefix** | `func.__name__` matched against prefix→kind map | 0.50 | `search_` → ReadOnly |
| **Docstring keywords** | `inspect.getdoc()` scanned for keyword families | 0.25 | "irreversible" → IrreversibleWrite |
| **Parameter names** | `inspect.signature()` parameter names | 0.15 | `to`, `recipient` → IrreversibleWrite |
| **Source AST** | `inspect.getsource()` for HTTP/SDK patterns | 0.10 | `requests.post()` → IrreversibleWrite |

**Safety guarantees:**
- Low confidence → defaults to `IrreversibleWrite` (never re-executes ambiguous tools)
- Compensatable auto-downgrades to `IrreversibleWrite` (requires compensation function)
- Explicit always wins: `overrides=`, `ToolDef(kind)`, `@tool(EffectKind.X)` bypass classification
- All classifications logged (`effect_log.classify` logger)

## Framework Integration

Built-in middleware for major agent frameworks. All middleware now accepts raw callables with auto-classification:

| Framework | Middleware | Entry Point |
|---|---|---|
| **LangGraph** | `effect_log.middleware.langgraph` | `EffectLogToolNode`, `effect_logged_tools` |
| **OpenAI Agents SDK** | `effect_log.middleware.openai_agents` | `effect_logged_agent`, `wrap_function_tool` |
| **CrewAI** | `effect_log.middleware.crewai` | `effect_logged_crew`, `effect_logged_tool` |
| **Pydantic AI** | `effect_log.middleware.pydantic_ai` | `effect_logged_agent`, `EffectLogToolset` |
| **Anthropic Claude API** | `effect_log.middleware.anthropic` | `effect_logged_tool_executor`, `process_tool_calls` |
| **Bub** | `effect_log.middleware.bub` | `EffectLoggedToolExecutor`, `effect_logged_agent` |

Middleware `make_tooldefs()` / `make_tools()` now accepts raw callables alongside spec dicts:

```python
from effect_log.middleware.anthropic import make_tooldefs

# Before: always needed explicit effect
make_tooldefs([
    {"func": search_db, "effect": EffectKind.ReadOnly},
    {"func": send_email, "effect": EffectKind.IrreversibleWrite},
])

# After: just pass functions
make_tooldefs([search_db, send_email])
```

See [`examples/`](examples/) for runnable demos:

- [`crash_recovery.py`](examples/crash_recovery.py) — Core crash recovery demo (Phase 1 milestone)
- [`langgraph_integration.py`](examples/langgraph_integration.py) — LangGraph ToolNode + tool wrapping
- [`openai_agents_integration.py`](examples/openai_agents_integration.py) — OpenAI Agents SDK wrapping
- [`crewai_integration.py`](examples/crewai_integration.py) — CrewAI tool + crew wrapping
- [`pydantic_ai_integration.py`](examples/pydantic_ai_integration.py) — Pydantic AI toolset wrapping
- [`anthropic_integration.py`](examples/anthropic_integration.py) — Anthropic Claude API tool_use
- [`e2e_bub.py`](examples/e2e_bub.py) — Bub agent crash recovery for bash/file tools

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
- [x] Framework middleware — LangGraph, OpenAI Agents SDK, CrewAI, Pydantic AI, Anthropic Claude API, Bub
- [x] Auto-classification — infer effect kind from function name, docstring, parameters, and source AST
- [ ] TypeScript bindings — napi-rs, Vercel AI SDK
- [ ] Additional backends — RocksDB, S3, Restate journal

## Inspiration

The idea behind this project was inspired by a blog post from [Guanlan Dai](https://www.linkedin.com/in/guanlandai/). He introduced the concepts of effect log and semantic correctness.

## License

Apache-2.0
