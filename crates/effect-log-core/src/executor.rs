use std::sync::Arc;

use chrono::Utc;
use tracing::{info, warn};

use crate::error::{EffectLogError, Result};
use crate::recovery::recovery_strategy;
use crate::registry::{RegisteredTool, ToolRegistry};
use crate::storage::EffectStore;
use crate::types::*;
use crate::wal::WalEngine;

/// The main user-facing API for effect-log.
pub struct EffectLog {
    wal: Arc<WalEngine>,
    registry: ToolRegistry,
    execution_id: String,
    read_policy: ReadRecoveryPolicy,
}

/// Builder for constructing an EffectLog.
pub struct EffectLogBuilder {
    store: Option<Arc<dyn EffectStore>>,
    execution_id: Option<String>,
    read_policy: ReadRecoveryPolicy,
    registry: ToolRegistry,
}

impl EffectLogBuilder {
    pub fn new() -> Self {
        Self {
            store: None,
            execution_id: None,
            read_policy: ReadRecoveryPolicy::default(),
            registry: ToolRegistry::new(),
        }
    }

    pub fn store(mut self, store: Arc<dyn EffectStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn execution_id(mut self, id: impl Into<String>) -> Self {
        self.execution_id = Some(id.into());
        self
    }

    pub fn read_policy(mut self, policy: ReadRecoveryPolicy) -> Self {
        self.read_policy = policy;
        self
    }

    pub fn register_tool(mut self, tool: RegisteredTool) -> Result<Self> {
        self.registry.register(tool)?;
        Ok(self)
    }

    /// Build the EffectLog for a fresh execution.
    pub fn build(self) -> Result<EffectLog> {
        let store = self
            .store
            .ok_or_else(|| EffectLogError::Builder("store is required".into()))?;
        let execution_id = self
            .execution_id
            .ok_or_else(|| EffectLogError::Builder("execution_id is required".into()))?;

        Ok(EffectLog {
            wal: Arc::new(WalEngine::new(store)),
            registry: self.registry,
            execution_id,
            read_policy: self.read_policy,
        })
    }

    /// Build the EffectLog and recover from a prior execution.
    pub async fn recover(self) -> Result<EffectLog> {
        let store = self
            .store
            .ok_or_else(|| EffectLogError::Builder("store is required".into()))?;
        let execution_id = self
            .execution_id
            .ok_or_else(|| EffectLogError::Builder("execution_id is required".into()))?;

        let wal = Arc::new(WalEngine::new(store));
        let plan = wal.recover(&execution_id, self.read_policy).await?;

        info!(
            execution_id = %execution_id,
            actions = plan.actions.len(),
            resume_from = plan.resume_from,
            "Recovery plan built"
        );

        Ok(EffectLog {
            wal,
            registry: self.registry,
            execution_id,
            read_policy: self.read_policy,
        })
    }
}

impl Default for EffectLogBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectLog {
    pub fn builder() -> EffectLogBuilder {
        EffectLogBuilder::new()
    }

    pub fn execution_id(&self) -> &str {
        &self.execution_id
    }

    /// Execute a tool call through the effect log.
    ///
    /// During recovery, this checks the WAL for a prior intent at the current
    /// cursor position and applies the recovery strategy. During normal execution,
    /// it writes intent/completion records around the tool call.
    pub async fn execute(
        &self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let tool = self
            .registry
            .get(tool_name)
            .ok_or_else(|| EffectLogError::ToolNotFound(tool_name.to_string()))?
            .clone();

        let seq = self.wal.advance_cursor(&self.execution_id).await;

        // Check WAL for existing intent at this cursor position
        if let Some(intent) = self.wal.lookup_intent(&self.execution_id, seq).await? {
            // Validate tool name matches
            if intent.tool_name != tool_name {
                return Err(EffectLogError::ToolMismatch {
                    sequence: seq,
                    expected: intent.tool_name.clone(),
                    actual: tool_name.to_string(),
                });
            }

            // Log warning if input differs
            if intent.input != input {
                warn!(
                    sequence = seq,
                    tool = tool_name,
                    "Input differs from original execution during recovery"
                );
            }

            let completion = self.wal.get_completion(&intent.effect_id).await?;
            return self
                .apply_recovery(&tool, &intent, completion.as_ref(), input)
                .await;
        }

        // Normal execution path
        self.execute_with_logging(&tool, input, seq).await
    }

    /// Execute multiple tool calls in parallel.
    pub async fn execute_parallel(
        &self,
        calls: Vec<(&str, serde_json::Value)>,
    ) -> Result<Vec<serde_json::Value>> {
        let mut handles = Vec::new();

        for (tool_name, input) in calls {
            let tool_name = tool_name.to_string();
            let wal = Arc::clone(&self.wal);
            let execution_id = self.execution_id.clone();
            let read_policy = self.read_policy;

            let tool = self
                .registry
                .get(&tool_name)
                .ok_or_else(|| EffectLogError::ToolNotFound(tool_name.clone()))?
                .clone();

            let seq = wal.advance_cursor(&execution_id).await;

            handles.push(tokio::spawn(async move {
                // Check WAL for existing intent
                if let Some(intent) = wal.lookup_intent(&execution_id, seq).await? {
                    if intent.tool_name != tool_name {
                        return Err(EffectLogError::ToolMismatch {
                            sequence: seq,
                            expected: intent.tool_name.clone(),
                            actual: tool_name,
                        });
                    }
                    let completion = wal.get_completion(&intent.effect_id).await?;
                    let action = recovery_strategy(&intent, completion.as_ref(), read_policy);
                    return apply_recovery_action(
                        &tool,
                        &intent,
                        completion.as_ref(),
                        input,
                        action,
                        &wal,
                    )
                    .await;
                }

                execute_tool_with_logging(&tool, input, seq, &execution_id, &wal).await
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            let result = handle
                .await
                .map_err(|e| EffectLogError::ToolExecutionFailed(e.to_string()))??;
            results.push(result);
        }

        Ok(results)
    }

    /// Get the execution history as a list of (intent, optional completion) pairs.
    pub async fn history(&self) -> Result<Vec<(IntentRecord, Option<CompletionRecord>)>> {
        let entries = self
            .wal
            .recover(&self.execution_id, self.read_policy)
            .await?;

        Ok(entries
            .actions
            .into_iter()
            .map(|(intent, completion, _)| (intent, completion))
            .collect())
    }

    async fn apply_recovery(
        &self,
        tool: &RegisteredTool,
        intent: &IntentRecord,
        completion: Option<&CompletionRecord>,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let action = recovery_strategy(intent, completion, self.read_policy);
        apply_recovery_action(tool, intent, completion, input, action, &self.wal).await
    }

    async fn execute_with_logging(
        &self,
        tool: &RegisteredTool,
        input: serde_json::Value,
        seq: u64,
    ) -> Result<serde_json::Value> {
        execute_tool_with_logging(tool, input, seq, &self.execution_id, &self.wal).await
    }
}

/// Shared logic for applying a recovery action.
async fn apply_recovery_action(
    tool: &RegisteredTool,
    intent: &IntentRecord,
    completion: Option<&CompletionRecord>,
    input: serde_json::Value,
    action: RecoveryAction,
    wal: &WalEngine,
) -> Result<serde_json::Value> {
    match action {
        RecoveryAction::ReturnSealed => {
            let completion = completion.ok_or_else(|| {
                EffectLogError::InternalInvariant(format!(
                    "ReturnSealed requires completion for effect {}",
                    intent.effect_id
                ))
            })?;
            info!(
                tool = %intent.tool_name,
                effect_id = %intent.effect_id,
                "Returning sealed result"
            );
            Ok(completion.sealed_response.clone())
        }
        RecoveryAction::Replay => {
            info!(
                tool = %intent.tool_name,
                effect_id = %intent.effect_id,
                "Replaying tool call"
            );
            let result = (tool.func)(input).await;

            // Write new completion for the replay
            let completion = CompletionRecord {
                effect_id: intent.effect_id,
                outcome: Outcome::Success,
                sealed_response: result.clone(),
                version_fingerprint: None,
                has_irreversible_change: false,
                compensation_info: None,
                completed_at: Utc::now(),
            };
            wal.write_completion(&completion).await?;

            Ok(result)
        }
        RecoveryAction::CompensateThenReplay => {
            info!(
                tool = %intent.tool_name,
                effect_id = %intent.effect_id,
                "Compensating then replaying"
            );

            if let Some(compensate_fn) = &tool.compensate {
                match compensate_fn(intent.input.clone()).await {
                    Ok(()) => {}
                    Err(e) => {
                        warn!(
                            tool = %intent.tool_name,
                            effect_id = %intent.effect_id,
                            error = %e,
                            "Compensation failed, escalating to human review"
                        );
                        return Err(EffectLogError::RequiresHumanReview {
                            effect_id: intent.effect_id,
                        });
                    }
                }
            }

            let result = (tool.func)(intent.input.clone()).await;

            let completion = CompletionRecord {
                effect_id: intent.effect_id,
                outcome: Outcome::Success,
                sealed_response: result.clone(),
                version_fingerprint: None,
                has_irreversible_change: false,
                compensation_info: None,
                completed_at: Utc::now(),
            };
            wal.write_completion(&completion).await?;

            Ok(result)
        }
        RecoveryAction::RequireHumanReview => Err(EffectLogError::RequiresHumanReview {
            effect_id: intent.effect_id,
        }),
    }
}

/// Extract a human-readable message from a panic payload.
fn extract_panic_message(join_error: tokio::task::JoinError) -> String {
    if join_error.is_panic() {
        let panic = join_error.into_panic();
        if let Some(s) = panic.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        }
    } else {
        join_error.to_string()
    }
}

/// Execute a tool with intent/completion logging.
///
/// This handles: idempotency key enforcement, panic catching via `tokio::spawn`,
/// and optional per-tool timeouts.
async fn execute_tool_with_logging(
    tool: &RegisteredTool,
    input: serde_json::Value,
    seq: u64,
    execution_id: &str,
    wal: &WalEngine,
) -> Result<serde_json::Value> {
    let effect_id = EffectId::new();

    // Compute idempotency key if applicable
    let idempotency_key = tool.idempotency_key_fn.as_ref().map(|f| f(&input));

    // Enforce idempotency: if a prior intent exists with the same key, return its result
    if let Some(ref key) = idempotency_key {
        if let Some(prior_intent) = wal.lookup_by_idempotency_key(execution_id, key).await? {
            if let Some(prior_completion) = wal.get_completion(&prior_intent.effect_id).await? {
                info!(
                    tool = %tool.name,
                    idempotency_key = %key,
                    prior_effect_id = %prior_intent.effect_id,
                    "Returning prior result for duplicate idempotency key"
                );
                return Ok(prior_completion.sealed_response.clone());
            }
        }
    }

    // Write intent BEFORE execution
    let intent = IntentRecord {
        effect_id,
        tool_call_id: format!("{}-{}", execution_id, seq),
        tool_name: tool.name.clone(),
        effect_kind: tool.effect_kind,
        input: input.clone(),
        idempotency_key,
        impact_scope: tool.impact_scope.clone(),
        timestamp: Utc::now(),
        cursor: ExecutionCursor {
            execution_id: execution_id.to_string(),
            sequence_number: seq,
            parent_effect_id: None,
        },
    };

    wal.write_intent(&intent).await?;

    // Execute the tool in a spawned task to catch panics
    let tool_func = Arc::clone(&tool.func);
    let input_clone = input.clone();
    let handle = tokio::spawn(async move { (tool_func)(input_clone).await });

    // Apply optional timeout, then await the result
    let exec_result = if let Some(timeout_duration) = tool.timeout {
        match tokio::time::timeout(timeout_duration, handle).await {
            Ok(join_result) => join_result,
            Err(_elapsed) => {
                warn!(
                    tool = %tool.name,
                    effect_id = %effect_id,
                    "Tool execution timed out"
                );
                let completion = CompletionRecord {
                    effect_id,
                    outcome: Outcome::Timeout,
                    sealed_response: serde_json::json!({"error": "tool execution timed out"}),
                    version_fingerprint: None,
                    has_irreversible_change: false,
                    compensation_info: None,
                    completed_at: Utc::now(),
                };
                wal.write_completion(&completion).await?;
                return Err(EffectLogError::ToolTimeout { effect_id });
            }
        }
    } else {
        handle.await
    };

    match exec_result {
        Ok(result) => {
            // Write completion AFTER successful execution
            let completion = CompletionRecord {
                effect_id,
                outcome: Outcome::Success,
                sealed_response: result.clone(),
                version_fingerprint: None,
                has_irreversible_change: matches!(
                    tool.effect_kind,
                    EffectKind::IrreversibleWrite | EffectKind::ReadThenWrite
                ),
                compensation_info: None,
                completed_at: Utc::now(),
            };
            wal.write_completion(&completion).await?;
            Ok(result)
        }
        Err(join_error) => {
            // Task panicked or was cancelled — write error completion
            let message = extract_panic_message(join_error);
            warn!(
                tool = %tool.name,
                effect_id = %effect_id,
                message = %message,
                "Tool panicked during execution"
            );
            let completion = CompletionRecord {
                effect_id,
                outcome: Outcome::Panicked {
                    message: message.clone(),
                },
                sealed_response: serde_json::json!({"error": "tool panicked", "message": message}),
                version_fingerprint: None,
                has_irreversible_change: false,
                compensation_info: None,
                completed_at: Utc::now(),
            };
            wal.write_completion(&completion).await?;
            Err(EffectLogError::ToolPanicked { effect_id, message })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ToolBuilder;
    use crate::storage::memory::InMemoryStore;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    fn make_store() -> Arc<InMemoryStore> {
        Arc::new(InMemoryStore::new())
    }

    fn counting_tool(name: &str, kind: EffectKind, counter: Arc<AtomicU32>) -> RegisteredTool {
        let builder = ToolBuilder::new(name, kind).func(Arc::new(move |input| {
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                counter.fetch_add(1, Ordering::SeqCst);
                serde_json::json!({"echoed": input, "executed": true})
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
    async fn basic_execute() {
        let store = make_store();
        let counter = Arc::new(AtomicU32::new(0));

        let log = EffectLog::builder()
            .store(store)
            .execution_id("test-1")
            .register_tool(counting_tool(
                "read_file",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .build()
            .unwrap();

        let result = log
            .execute("read_file", serde_json::json!({"path": "/tmp/a"}))
            .await
            .unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(result["executed"], true);
    }

    #[tokio::test]
    async fn tool_not_found() {
        let store = make_store();
        let log = EffectLog::builder()
            .store(store)
            .execution_id("test-1")
            .build()
            .unwrap();

        let result = log.execute("nonexistent", serde_json::json!({})).await;
        assert!(matches!(result, Err(EffectLogError::ToolNotFound(_))));
    }

    #[tokio::test]
    async fn history_records_executions() {
        let store = make_store();
        let counter = Arc::new(AtomicU32::new(0));

        let log = EffectLog::builder()
            .store(store)
            .execution_id("test-1")
            .register_tool(counting_tool(
                "tool_a",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(counting_tool(
                "tool_b",
                EffectKind::IrreversibleWrite,
                Arc::clone(&counter),
            ))
            .unwrap()
            .build()
            .unwrap();

        log.execute("tool_a", serde_json::json!({})).await.unwrap();
        log.execute("tool_b", serde_json::json!({})).await.unwrap();

        let history = log.history().await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].0.tool_name, "tool_a");
        assert_eq!(history[1].0.tool_name, "tool_b");
        assert!(history[0].1.is_some()); // both completed
        assert!(history[1].1.is_some());
    }

    #[tokio::test]
    async fn recovery_returns_sealed_for_irreversible() {
        let store = make_store();
        let counter = Arc::new(AtomicU32::new(0));

        // First execution: run 2 steps
        {
            let log = EffectLog::builder()
                .store(Arc::clone(&store) as Arc<dyn EffectStore>)
                .execution_id("test-1")
                .register_tool(counting_tool(
                    "send_email",
                    EffectKind::IrreversibleWrite,
                    Arc::clone(&counter),
                ))
                .unwrap()
                .register_tool(counting_tool(
                    "read_db",
                    EffectKind::ReadOnly,
                    Arc::clone(&counter),
                ))
                .unwrap()
                .build()
                .unwrap();

            log.execute("send_email", serde_json::json!({"to": "a@b.com"}))
                .await
                .unwrap();
            log.execute("read_db", serde_json::json!({"q": "SELECT 1"}))
                .await
                .unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 2);

        // Recovery: same execution_id
        let log = EffectLog::builder()
            .store(Arc::clone(&store) as Arc<dyn EffectStore>)
            .execution_id("test-1")
            .register_tool(counting_tool(
                "send_email",
                EffectKind::IrreversibleWrite,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(counting_tool(
                "read_db",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .recover()
            .await
            .unwrap();

        // send_email: IrreversibleWrite completed → ReturnSealed (no re-execution)
        let result = log
            .execute("send_email", serde_json::json!({"to": "a@b.com"}))
            .await
            .unwrap();
        assert_eq!(result["executed"], true); // sealed result from first run

        // read_db: ReadOnly completed with ReplayFresh → Replay (re-executes)
        let _result = log
            .execute("read_db", serde_json::json!({"q": "SELECT 1"}))
            .await
            .unwrap();

        // send_email was NOT re-executed (counter stayed at 2 after sealed return)
        // read_db WAS re-executed (ReplayFresh policy)
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn parallel_execute() {
        let store = make_store();
        let counter = Arc::new(AtomicU32::new(0));

        let log = EffectLog::builder()
            .store(store)
            .execution_id("test-1")
            .register_tool(counting_tool(
                "tool_a",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .register_tool(counting_tool(
                "tool_b",
                EffectKind::ReadOnly,
                Arc::clone(&counter),
            ))
            .unwrap()
            .build()
            .unwrap();

        let results = log
            .execute_parallel(vec![
                ("tool_a", serde_json::json!({"a": 1})),
                ("tool_b", serde_json::json!({"b": 2})),
            ])
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn idempotency_key_dedup() {
        let store = make_store();
        let counter = Arc::new(AtomicU32::new(0));

        let tool = ToolBuilder::new("upsert", EffectKind::IdempotentWrite)
            .func({
                let counter = Arc::clone(&counter);
                Arc::new(move |input| {
                    let counter = Arc::clone(&counter);
                    Box::pin(async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        serde_json::json!({"written": input})
                    })
                })
            })
            .idempotency_key(Arc::new(|input| {
                input["id"].as_str().unwrap_or("unknown").to_string()
            }))
            .build()
            .unwrap();

        let log = EffectLog::builder()
            .store(store)
            .execution_id("test-idemp")
            .register_tool(tool)
            .unwrap()
            .build()
            .unwrap();

        // First call executes
        let r1 = log
            .execute("upsert", serde_json::json!({"id": "abc", "value": 1}))
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Second call with same idempotency key returns prior result
        let r2 = log
            .execute("upsert", serde_json::json!({"id": "abc", "value": 2}))
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1); // NOT re-executed
        assert_eq!(r1, r2); // Same result returned

        // Third call with different key executes
        let _r3 = log
            .execute("upsert", serde_json::json!({"id": "def", "value": 3}))
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn tool_timeout() {
        let store = make_store();

        let tool = ToolBuilder::new("slow_tool", EffectKind::ReadOnly)
            .func(Arc::new(|_input| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    serde_json::json!({"should": "not reach"})
                })
            }))
            .timeout(Duration::from_millis(50))
            .build()
            .unwrap();

        let log = EffectLog::builder()
            .store(store)
            .execution_id("test-timeout")
            .register_tool(tool)
            .unwrap()
            .build()
            .unwrap();

        let result = log.execute("slow_tool", serde_json::json!({})).await;
        assert!(matches!(result, Err(EffectLogError::ToolTimeout { .. })));
    }

    #[tokio::test]
    async fn compensation_failure_escalates_to_human_review() {
        let store = make_store();
        let counter = Arc::new(AtomicU32::new(0));

        // Create a compensatable tool with a failing compensate function
        let tool = ToolBuilder::new("create_vm", EffectKind::Compensatable)
            .func({
                let counter = Arc::clone(&counter);
                Arc::new(move |input| {
                    let counter = Arc::clone(&counter);
                    Box::pin(async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        serde_json::json!({"created": input})
                    })
                })
            })
            .compensate(Arc::new(|_| {
                Box::pin(async { Err("VM deletion API is down".to_string()) })
            }))
            .build()
            .unwrap();

        // Simulate a crash: write intent but no completion
        {
            use chrono::Utc;
            let intent = IntentRecord {
                effect_id: EffectId::new(),
                tool_call_id: "test-comp-1".into(),
                tool_name: "create_vm".into(),
                effect_kind: EffectKind::Compensatable,
                input: serde_json::json!({"name": "vm-1"}),
                idempotency_key: None,
                impact_scope: None,
                timestamp: Utc::now(),
                cursor: ExecutionCursor {
                    execution_id: "test-comp".into(),
                    sequence_number: 1,
                    parent_effect_id: None,
                },
            };
            store.write_intent(&intent).await.unwrap();
        }

        // Recovery should escalate to human review when compensation fails
        let log = EffectLog::builder()
            .store(Arc::clone(&store) as Arc<dyn EffectStore>)
            .execution_id("test-comp")
            .register_tool(tool)
            .unwrap()
            .recover()
            .await
            .unwrap();

        let result = log
            .execute("create_vm", serde_json::json!({"name": "vm-1"}))
            .await;

        assert!(matches!(
            result,
            Err(EffectLogError::RequiresHumanReview { .. })
        ));
        assert_eq!(counter.load(Ordering::SeqCst), 0); // Tool was NOT re-executed
    }
}
