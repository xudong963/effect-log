# Changelog

## v0.1.0

Initial release.

### Features

- **Core WAL engine** — intent/completion write-ahead log with cursor-based recovery
- **Recovery engine** — automatic crash recovery driven by effect kind semantics
- **Storage backends** — SQLite and in-memory
- **Python bindings** — `EffectLog`, `ToolDef`, `EffectKind` via PyO3 + maturin
- **Framework middleware**:
  - LangGraph — `EffectLogToolNode`, `effect_logged_tools`
  - OpenAI Agents SDK — `effect_logged_agent`, `wrap_function_tool`
  - CrewAI — `effect_logged_crew`, `effect_logged_tool`
