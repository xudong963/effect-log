//! Integration test: Phase 1 milestone — crash recovery demonstration.
//!
//! Simulates a 5-step task where step 3 is an IrreversibleWrite.
//! The "process" is killed at step 4. On restart, step 3's sealed result
//! is returned without re-execution, and the task resumes from step 4.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use effect_log_core::*;

fn make_tool(name: &str, kind: EffectKind, counter: Arc<AtomicU32>) -> RegisteredTool {
    let name_owned = name.to_string();
    let builder = ToolBuilder::new(name, kind).func(Arc::new(move |input| {
        let counter = Arc::clone(&counter);
        let name = name_owned.clone();
        Box::pin(async move {
            counter.fetch_add(1, Ordering::SeqCst);
            serde_json::json!({
                "tool": name,
                "input": input,
                "executed": true,
            })
        })
    }));

    let builder = if kind == EffectKind::Compensatable {
        builder.compensate(Arc::new(|_| Box::pin(async { Ok(()) })))
    } else {
        builder
    };

    builder.build().unwrap()
}

#[tokio::test]
async fn crash_recovery_5_step_task() {
    let store = Arc::new(InMemoryStore::new());
    let counter = Arc::new(AtomicU32::new(0));

    // === First execution: run steps 1-3, then "crash" before step 4 completes ===
    {
        let log = EffectLog::builder()
            .store(Arc::clone(&store) as Arc<dyn EffectStore>)
            .execution_id("task-001")
            .register_tool(make_tool(
                "fetch_data",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "transform",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "send_email",
                EffectKind::IrreversibleWrite,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "update_db",
                EffectKind::IdempotentWrite,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "log_result",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .build()
            .unwrap();

        // Step 1: fetch_data (ReadOnly)
        let r1 = log
            .execute("fetch_data", serde_json::json!({"source": "api"}))
            .await
            .unwrap();
        assert_eq!(r1["tool"], "fetch_data");

        // Step 2: transform (ReadOnly)
        let r2 = log
            .execute("transform", serde_json::json!({"data": r1}))
            .await
            .unwrap();
        assert_eq!(r2["tool"], "transform");

        // Step 3: send_email (IrreversibleWrite) - THE CRITICAL STEP
        let r3 = log
            .execute(
                "send_email",
                serde_json::json!({"to": "ceo@company.com", "subject": "Report"}),
            )
            .await
            .unwrap();
        assert_eq!(r3["tool"], "send_email");

        // "CRASH" — we drop the EffectLog here without executing steps 4 and 5
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    // === Recovery: resume the execution ===
    {
        let log = EffectLog::builder()
            .store(Arc::clone(&store) as Arc<dyn EffectStore>)
            .execution_id("task-001")
            .register_tool(make_tool(
                "fetch_data",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "transform",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "send_email",
                EffectKind::IrreversibleWrite,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "update_db",
                EffectKind::IdempotentWrite,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "log_result",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .recover()
            .await
            .unwrap();

        let before_count = counter.load(Ordering::SeqCst);

        // Step 1: fetch_data (ReadOnly, completed) → ReplayFresh by default → re-executes
        let r1 = log
            .execute("fetch_data", serde_json::json!({"source": "api"}))
            .await
            .unwrap();
        assert_eq!(r1["tool"], "fetch_data");
        assert_eq!(counter.load(Ordering::SeqCst), before_count + 1);

        // Step 2: transform (ReadOnly, completed) → ReplayFresh → re-executes
        let r2 = log
            .execute("transform", serde_json::json!({"data": r1}))
            .await
            .unwrap();
        assert_eq!(r2["tool"], "transform");
        assert_eq!(counter.load(Ordering::SeqCst), before_count + 2);

        // Step 3: send_email (IrreversibleWrite, completed) → ReturnSealed → NO re-execution!
        let count_before_email = counter.load(Ordering::SeqCst);
        let r3 = log
            .execute(
                "send_email",
                serde_json::json!({"to": "ceo@company.com", "subject": "Report"}),
            )
            .await
            .unwrap();
        // The sealed result is from the original execution
        assert_eq!(r3["tool"], "send_email");
        assert_eq!(r3["executed"], true);
        // Counter did NOT increment — email was NOT re-sent
        assert_eq!(counter.load(Ordering::SeqCst), count_before_email);

        // Step 4: update_db (IdempotentWrite, was never executed) → normal execution
        let r4 = log
            .execute(
                "update_db",
                serde_json::json!({"table": "reports", "status": "sent"}),
            )
            .await
            .unwrap();
        assert_eq!(r4["tool"], "update_db");

        // Step 5: log_result (ReadOnly, was never executed) → normal execution
        let r5 = log
            .execute("log_result", serde_json::json!({"message": "done"}))
            .await
            .unwrap();
        assert_eq!(r5["tool"], "log_result");

        // Verify history has all 5 steps
        let history = log.history().await.unwrap();
        assert_eq!(history.len(), 5);
    }
}

#[tokio::test]
async fn crash_during_irreversible_write_escalates() {
    let store = Arc::new(InMemoryStore::new());

    // Simulate: write intent for an IrreversibleWrite but NO completion (crash mid-execution)
    {
        use chrono::Utc;

        let intent = IntentRecord {
            effect_id: EffectId::new(),
            tool_call_id: "task-002-1".into(),
            tool_name: "send_email".into(),
            effect_kind: EffectKind::IrreversibleWrite,
            input: serde_json::json!({"to": "ceo@company.com"}),
            idempotency_key: None,
            impact_scope: None,
            timestamp: Utc::now(),
            cursor: ExecutionCursor {
                execution_id: "task-002".into(),
                sequence_number: 1,
                parent_effect_id: None,
            },
        };
        store.write_intent(&intent).await.unwrap();
        // NO completion — simulating crash during execution
    }

    // Recovery: should escalate to human review
    let counter = Arc::new(AtomicU32::new(0));
    let log = EffectLog::builder()
        .store(Arc::clone(&store) as Arc<dyn EffectStore>)
        .execution_id("task-002")
        .register_tool(make_tool(
            "send_email",
            EffectKind::IrreversibleWrite,
            Arc::clone(&counter),
        ))
        .unwrap()
        .recover()
        .await
        .unwrap();

    let result = log
        .execute("send_email", serde_json::json!({"to": "ceo@company.com"}))
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        EffectLogError::RequiresHumanReview { .. } => {} // Expected
        e => panic!("Expected RequiresHumanReview, got: {e}"),
    }

    // Tool was NOT re-executed
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn crash_recovery_with_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db").to_str().unwrap().to_string();
    let counter = Arc::new(AtomicU32::new(0));

    // First execution: run 3 steps
    {
        let store = Arc::new(SqliteStore::open(&db_path).await.unwrap());
        let log = EffectLog::builder()
            .store(store as Arc<dyn EffectStore>)
            .execution_id("task-003")
            .register_tool(make_tool(
                "read_db",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "send_email",
                EffectKind::IrreversibleWrite,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "write_log",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .build()
            .unwrap();

        log.execute("read_db", serde_json::json!({})).await.unwrap();
        log.execute("send_email", serde_json::json!({"to": "a@b.com"}))
            .await
            .unwrap();
        // "crash" before step 3
    }

    assert_eq!(counter.load(Ordering::SeqCst), 2);

    // Recovery from SQLite
    {
        let store = Arc::new(SqliteStore::open(&db_path).await.unwrap());
        let log = EffectLog::builder()
            .store(store as Arc<dyn EffectStore>)
            .execution_id("task-003")
            .register_tool(make_tool(
                "read_db",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "send_email",
                EffectKind::IrreversibleWrite,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(make_tool(
                "write_log",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .recover()
            .await
            .unwrap();

        let before = counter.load(Ordering::SeqCst);

        // Step 1: read_db → replayed (ReadOnly + ReplayFresh)
        log.execute("read_db", serde_json::json!({})).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), before + 1);

        // Step 2: send_email → sealed (IrreversibleWrite, completed)
        let count_before_email = counter.load(Ordering::SeqCst);
        log.execute("send_email", serde_json::json!({"to": "a@b.com"}))
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), count_before_email); // NOT re-executed

        // Step 3: write_log → normal (never executed before)
        log.execute("write_log", serde_json::json!({}))
            .await
            .unwrap();
    }
}
