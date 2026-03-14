# Show HN: effect-log – Your AI Agent Crashed After Sending the Email. Now What?

Here's a scenario. Your AI agent is running a 5-step task. Step 3 sends an email to your CEO. Step 4 records that the email was sent. The process crashes between step 3 and step 4.

Now what?

The email was sent. There's no record of it. You restart the agent. It replays from the beginning. The CEO gets the email twice.

This problem — ensuring exactly-once side effects across crashes — was solved decades ago in databases with write-ahead logs, and later in distributed systems with durable execution engines like Temporal. AI agent frameworks are starting to address it, but at the wrong level of abstraction. LangGraph, for example, checkpoints graph state between nodes and recently added a `tasks` API to persist individual operation results. But checkpointing and recovery are semantic-blind — a read and an email send get the same treatment. If you want to prevent an email from being re-sent on recovery, you wrap it in a task. If you want a database read to re-execute for fresh data, you... also wrap it in a task, but differently. There's no declaration that drives this automatically.

I built [effect-log](https://github.com/xudong963/effect-log) to fix this.

The idea behind this project was inspired by a blog post from [Guanlan Dai](https://www.linkedin.com/in/guanlandai/). He introduced the concepts of effect log and semantic correctness.

## The Key Insight: Not All Side Effects Are Equal

Most recovery systems treat all operations the same — either replay everything or checkpoint opaquely. But a read and a payment are fundamentally different, and a crash recovery system should treat them differently.

effect-log requires every tool to declare its **effect kind** at registration time. There are five:

| EffectKind | What It Means | Examples |
|---|---|---|
| `ReadOnly` | Pure read, no mutation | File reads, DB queries, GET requests |
| `IdempotentWrite` | Safe to replay with same key | PUT/upsert, Stripe charges with idempotency keys |
| `Compensatable` | Reversible — has a known undo | Creating a VM (undo: delete it), booking a seat (undo: cancel) |
| `IrreversibleWrite` | Cannot be undone once done | Sending emails, fund transfers, deployments |
| `ReadThenWrite` | Reads state, then mutates based on what was read | Read-modify-write cycles |

This classification is the single piece of metadata that drives all recovery behavior. You declare it once per tool, and the system handles the rest.

## How It Works

effect-log maintains a write-ahead log with two record types:

1. **Intent** — written *before* a tool executes (what we're about to do)
2. **Completion** — written *after* it finishes (what happened)

An intent without a matching completion is the signature of a crash. That gap is what triggers the recovery engine.

```python
from effect_log import EffectKind, EffectLog, ToolDef

tools = [
    ToolDef("fetch_data",  EffectKind.ReadOnly,         fetch_data_fn),
    ToolDef("send_email",  EffectKind.IrreversibleWrite, send_email_fn),
    ToolDef("upsert_db",   EffectKind.IdempotentWrite,   upsert_fn),
]

# Normal execution
log = EffectLog(execution_id="task-001", tools=tools, storage="sqlite:///effects.db")

data = log.execute("fetch_data",  {"source": "https://api.example.com/daily-report"})
log.execute("send_email",  {"to": "ceo@co.com", "subject": data["title"], "body": data["report"]})
log.execute("upsert_db",   {"id": data["report_id"], "status": "sent", "sent_to": "ceo@co.com"})
```

Notice how the output of `fetch_data` flows into `send_email` and `upsert_db`. This is the normal case — each step depends on the previous one.

If the process crashes after `send_email` but before `upsert_db`, recovery looks like this:

```python
# Recovery — same code, just add recover=True
log = EffectLog(execution_id="task-001", tools=tools,
                storage="sqlite:///effects.db", recover=True)

# Step 1: ReadOnly + completed → Replayed (re-fetches fresh data from the API)
data = log.execute("fetch_data",  {"source": "https://api.example.com/daily-report"})

# Step 2: IrreversibleWrite + completed → SEALED (returns stored result, function never called)
log.execute("send_email",  {"to": "ceo@co.com", "subject": data["title"], "body": data["report"]})

# Step 3: IdempotentWrite + no completion → Executes normally (picks up where we left off)
log.execute("upsert_db",   {"id": data["report_id"], "status": "sent", "sent_to": "ceo@co.com"})
```

Three tools, three different recovery behaviors — all driven by the effect kind declared at registration time. `fetch_data` re-executes for fresh data because reads are safe to repeat. `send_email` returns the sealed result from the first run — the function is never called again, no duplicate email. `upsert_db` executes normally because it never ran in the first place.

## The Recovery Matrix

The recovery engine is a pure function — no I/O, no side effects. It takes an intent record, an optional completion record, and returns one of four actions:

```rust
pub fn recovery_strategy(
    record: &IntentRecord,
    completion: Option<&CompletionRecord>,
    read_policy: ReadRecoveryPolicy,
) -> RecoveryAction {
    match (&record.effect_kind, completion) {
        // Completed effects → return sealed result
        (EffectKind::IrreversibleWrite, Some(_)) => ReturnSealed,
        (EffectKind::IdempotentWrite, Some(_))   => ReturnSealed,
        (EffectKind::Compensatable, Some(_))     => ReturnSealed,
        (EffectKind::ReadThenWrite, Some(_))     => ReturnSealed,

        // ReadOnly completed → depends on policy
        (EffectKind::ReadOnly, Some(_)) => match read_policy {
            ReplayFresh  => Replay,       // get fresh data
            ReturnSealed => ReturnSealed, // consistency with downstream writes
        },

        // No completion = crashed during execution
        (EffectKind::ReadOnly, None)          => Replay,
        (EffectKind::IdempotentWrite, None)   => Replay,
        (EffectKind::Compensatable, None)     => CompensateThenReplay,
        (EffectKind::IrreversibleWrite, None) => RequireHumanReview,
        (EffectKind::ReadThenWrite, None)     => RequireHumanReview,
    }
}
```

The entire recovery logic fits in one screen. Every branch is exhaustive. Every combination of (effect kind, completion status) maps to exactly one action.

## The Hardest Design Decision: Honest Uncertainty

When an `IrreversibleWrite` has an intent record but no completion, effect-log does not guess. It does not retry. It returns `RequireHumanReview`.

Why? Because we genuinely don't know what happened. The email might have been sent (SMTP accepted it, then we crashed before writing the completion). Or the process might have crashed before the email left. There is no way to tell from the local log alone.

This is the [Two Generals' Problem](https://en.wikipedia.org/wiki/Two_Generals%27_Problem). You cannot distinguish "succeeded then crashed" from "crashed before succeeding" without an acknowledgment that was itself lost in the crash.

Most systems either silently retry (risking duplicates) or silently skip (risking data loss). effect-log chooses a third path: *admit uncertainty and ask a human*. This is the most important design decision in the entire system.

For `Compensatable` effects, we have a better option: call the registered undo function first, then replay. If you crash while creating a VM, we delete the possibly-created VM, then create a fresh one. This is safe because the compensation is designed to be idempotent — deleting a non-existent VM is a no-op.

## What This Is NOT

I want to be explicit about scope, because the most common reaction to projects like this is "just use Temporal."

**Not a workflow engine.** effect-log doesn't schedule, order, or coordinate tool calls. Your agent framework (LangGraph, CrewAI, OpenAI SDK, whatever) owns control flow. effect-log just logs and recovers tool calls within that flow.

**Not distributed transactions.** No two-phase commit, no consensus protocol. effect-log runs in-process with a local SQLite WAL.

**Not a replacement for Temporal or Restate.** If you already run Temporal, great — effect-log could be a complementary semantic layer. Temporal knows step 5 completed; effect-log knows step 5 was an irreversible email send and shouldn't be replayed.

## Architecture

```
Agent Framework (LangGraph / CrewAI / OpenAI SDK / custom)
         │
    ┌────▼────┐
    │effect-log│ ← 5 effect kinds × recovery matrix
    └────┬────┘
         │ Intent (before) / Completion (after)
    ┌────▼────┐
    │ Storage  │ ← SQLite (default), in-memory (test), pluggable
    └─────────┘
```

Core is ~1200 lines of Rust. Python bindings via PyO3. SQLite with WAL mode for durability. The storage trait is pluggable — you could back it with RocksDB, S3, or Restate's journal.

Each tool call gets a monotonically increasing sequence number within an execution. Recovery matches resumed calls to WAL entries by `(execution_id, sequence_number)`, not by argument hashing. This avoids subtle bugs when the agent re-derives arguments slightly differently on the second run (floating-point formatting, key ordering, etc.).

## Current Status

**What works today:**
- Rust core library with full recovery engine
- SQLite and in-memory storage backends
- Python bindings (PyO3 + maturin)
- Middleware for LangGraph, OpenAI Agents SDK, CrewAI
- Parallel tool call support
- Idempotency key deduplication
- Crash recovery end-to-end demo

**What's coming:**
- TypeScript bindings (napi-rs) for Vercel AI SDK
- RocksDB and S3 storage backends
- Auto-inference of effect kind from HTTP methods (GET → ReadOnly, PUT → IdempotentWrite, etc.)

## The Bet

I'm betting that as AI agents move from demos to production, side-effect reliability becomes a hard requirement. Today, most agent frameworks assume tool calls are pure functions. They're not. A `send_email` call that executes twice because of a restart is not a bug in the agent's logic — it's a bug in the infrastructure.

The five-way classification isn't original. Database people will recognize it as a simplification of transaction isolation levels. Distributed systems people will see echoes of saga patterns. The contribution is packaging this into a library that an AI agent developer can adopt in ten minutes.

**Code:** [https://github.com/xudong963/effect-log](https://github.com/xudong963/effect-log)

I'd love feedback on the classification model — are five kinds the right number? Are there tool types that don't fit cleanly? And if you're building agents that take real-world actions, I'm curious what failure modes you've hit.
