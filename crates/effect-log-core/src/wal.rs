use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::{EffectLogError, Result};
use crate::recovery::{recovery_strategy, RecoveryPlan};
use crate::storage::{EffectStore, WalEntry};
use crate::types::*;

/// The WAL engine manages cursor state and delegates persistence to an EffectStore.
pub struct WalEngine {
    store: Arc<dyn EffectStore>,
    /// Current execution state, keyed by execution_id.
    cursors: Mutex<HashMap<String, u64>>,
}

impl WalEngine {
    pub fn new(store: Arc<dyn EffectStore>) -> Self {
        Self {
            store,
            cursors: Mutex::new(HashMap::new()),
        }
    }

    /// Advance the cursor for the given execution and return the new sequence number.
    pub async fn advance_cursor(&self, execution_id: &str) -> u64 {
        let mut cursors = self.cursors.lock().await;
        let seq = cursors.entry(execution_id.to_string()).or_insert(0);
        *seq += 1;
        *seq
    }

    /// Get the current sequence number without advancing.
    pub async fn current_sequence(&self, execution_id: &str) -> u64 {
        let cursors = self.cursors.lock().await;
        cursors.get(execution_id).copied().unwrap_or(0)
    }

    /// Write an intent record to the WAL.
    pub async fn write_intent(&self, record: &IntentRecord) -> Result<()> {
        self.store
            .write_intent(record)
            .await
            .map_err(EffectLogError::Storage)
    }

    /// Write a completion record to the WAL.
    pub async fn write_completion(&self, record: &CompletionRecord) -> Result<()> {
        self.store
            .write_completion(record)
            .await
            .map_err(EffectLogError::Storage)
    }

    /// Look up an intent at a specific cursor position.
    pub async fn lookup_intent(
        &self,
        execution_id: &str,
        sequence_number: u64,
    ) -> Result<Option<IntentRecord>> {
        self.store
            .lookup_intent(execution_id, sequence_number)
            .await
            .map_err(EffectLogError::Storage)
    }

    /// Get the completion for a given effect.
    pub async fn get_completion(&self, effect_id: &EffectId) -> Result<Option<CompletionRecord>> {
        self.store
            .get_completion(effect_id)
            .await
            .map_err(EffectLogError::Storage)
    }

    /// Look up an intent by idempotency key.
    pub async fn lookup_by_idempotency_key(
        &self,
        execution_id: &str,
        key: &str,
    ) -> Result<Option<IntentRecord>> {
        self.store
            .lookup_by_idempotency_key(execution_id, key)
            .await
            .map_err(EffectLogError::Storage)
    }

    /// Load and analyze all WAL entries for an execution to produce a recovery plan.
    pub async fn recover(
        &self,
        execution_id: &str,
        read_policy: ReadRecoveryPolicy,
    ) -> Result<RecoveryPlan> {
        let entries = self
            .store
            .load_execution(execution_id)
            .await
            .map_err(EffectLogError::Storage)?;

        // Reconstruct intents and completions
        let mut intents: Vec<IntentRecord> = Vec::new();
        let mut completions: HashMap<EffectId, CompletionRecord> = HashMap::new();

        for entry in entries {
            match entry {
                WalEntry::Intent(i) => intents.push(i),
                WalEntry::Completion(c) => {
                    completions.insert(c.effect_id, c);
                }
            }
        }

        // Sort by sequence_number
        intents.sort_by_key(|i| i.cursor.sequence_number);

        // Reset cursor to 0 so that subsequent execute() calls via advance_cursor
        // will yield 1, 2, 3... matching the WAL sequence numbers for recovery.
        {
            let mut cursors = self.cursors.lock().await;
            cursors.insert(execution_id.to_string(), 0);
        }

        // Build recovery plan
        let mut actions = Vec::new();
        let mut resume_from = u64::MAX;

        for intent in intents {
            let completion = completions.remove(&intent.effect_id);
            let action = recovery_strategy(&intent, completion.as_ref(), read_policy);

            // The first non-ReturnSealed action is where we resume from
            if action != RecoveryAction::ReturnSealed && intent.cursor.sequence_number < resume_from
            {
                resume_from = intent.cursor.sequence_number;
            }

            actions.push((intent, completion, action));
        }

        if resume_from == u64::MAX {
            // All effects are sealed — resume from after the last one
            resume_from = actions
                .last()
                .map(|(i, _, _)| i.cursor.sequence_number + 1)
                .unwrap_or(1);
        }

        Ok(RecoveryPlan {
            actions,
            resume_from,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::InMemoryStore;
    use chrono::Utc;

    fn make_wal() -> WalEngine {
        WalEngine::new(Arc::new(InMemoryStore::new()))
    }

    fn make_intent(execution_id: &str, seq: u64, kind: EffectKind) -> IntentRecord {
        IntentRecord {
            effect_id: EffectId::new(),
            tool_call_id: format!("call-{seq}"),
            tool_name: format!("tool_{seq}"),
            effect_kind: kind,
            input: serde_json::json!({"step": seq}),
            idempotency_key: None,
            impact_scope: None,
            timestamp: Utc::now(),
            cursor: ExecutionCursor {
                execution_id: execution_id.into(),
                sequence_number: seq,
                parent_effect_id: None,
            },
        }
    }

    fn make_completion(effect_id: EffectId) -> CompletionRecord {
        CompletionRecord {
            effect_id,
            outcome: Outcome::Success,
            sealed_response: serde_json::json!({"done": true}),
            version_fingerprint: None,
            has_irreversible_change: false,
            compensation_info: None,
            completed_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn cursor_advances_sequentially() {
        let wal = make_wal();
        assert_eq!(wal.advance_cursor("exec-1").await, 1);
        assert_eq!(wal.advance_cursor("exec-1").await, 2);
        assert_eq!(wal.advance_cursor("exec-1").await, 3);
        // Different execution has its own cursor
        assert_eq!(wal.advance_cursor("exec-2").await, 1);
    }

    #[tokio::test]
    async fn write_and_lookup_intent() {
        let wal = make_wal();
        let intent = make_intent("exec-1", 1, EffectKind::ReadOnly);

        wal.write_intent(&intent).await.unwrap();

        let found = wal.lookup_intent("exec-1", 1).await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().effect_id, intent.effect_id);
    }

    #[tokio::test]
    async fn write_and_get_completion() {
        let wal = make_wal();
        let intent = make_intent("exec-1", 1, EffectKind::ReadOnly);
        let completion = make_completion(intent.effect_id);

        wal.write_intent(&intent).await.unwrap();
        wal.write_completion(&completion).await.unwrap();

        let found = wal.get_completion(&intent.effect_id).await.unwrap();
        assert!(found.is_some());
    }

    #[tokio::test]
    async fn recover_all_completed() {
        let wal = make_wal();

        // Write 3 completed intents
        for seq in 1..=3 {
            let intent = make_intent("exec-1", seq, EffectKind::IrreversibleWrite);
            wal.write_intent(&intent).await.unwrap();
            wal.write_completion(&make_completion(intent.effect_id))
                .await
                .unwrap();
        }

        let plan = wal
            .recover("exec-1", ReadRecoveryPolicy::ReplayFresh)
            .await
            .unwrap();

        assert_eq!(plan.actions.len(), 3);
        // All sealed → resume from after last
        assert_eq!(plan.resume_from, 4);
        for (_, _, action) in &plan.actions {
            assert_eq!(*action, RecoveryAction::ReturnSealed);
        }
    }

    #[tokio::test]
    async fn recover_with_crash_at_irreversible() {
        let wal = make_wal();

        // Step 1: ReadOnly, completed
        let i1 = make_intent("exec-1", 1, EffectKind::ReadOnly);
        wal.write_intent(&i1).await.unwrap();
        wal.write_completion(&make_completion(i1.effect_id))
            .await
            .unwrap();

        // Step 2: IrreversibleWrite, completed
        let i2 = make_intent("exec-1", 2, EffectKind::IrreversibleWrite);
        wal.write_intent(&i2).await.unwrap();
        wal.write_completion(&make_completion(i2.effect_id))
            .await
            .unwrap();

        // Step 3: IrreversibleWrite, NO completion (crash!)
        let i3 = make_intent("exec-1", 3, EffectKind::IrreversibleWrite);
        wal.write_intent(&i3).await.unwrap();

        let plan = wal
            .recover("exec-1", ReadRecoveryPolicy::ReplayFresh)
            .await
            .unwrap();

        assert_eq!(plan.actions.len(), 3);
        assert_eq!(plan.resume_from, 1); // ReadOnly with ReplayFresh → Replay

        // Step 1: ReadOnly completed with ReplayFresh → Replay
        assert_eq!(plan.actions[0].2, RecoveryAction::Replay);
        // Step 2: IrreversibleWrite completed → ReturnSealed
        assert_eq!(plan.actions[1].2, RecoveryAction::ReturnSealed);
        // Step 3: IrreversibleWrite crashed → RequireHumanReview
        assert_eq!(plan.actions[2].2, RecoveryAction::RequireHumanReview);
    }

    #[tokio::test]
    async fn recover_sets_cursor() {
        let wal = make_wal();

        for seq in 1..=3 {
            let intent = make_intent("exec-1", seq, EffectKind::ReadOnly);
            wal.write_intent(&intent).await.unwrap();
            wal.write_completion(&make_completion(intent.effect_id))
                .await
                .unwrap();
        }

        let _ = wal
            .recover("exec-1", ReadRecoveryPolicy::ReplayFresh)
            .await
            .unwrap();

        // Cursor should be reset to 0 for recovery replay
        assert_eq!(wal.current_sequence("exec-1").await, 0);
    }

    #[tokio::test]
    async fn concurrent_cursor_advances() {
        let wal = Arc::new(make_wal());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let wal = Arc::clone(&wal);
            handles.push(tokio::spawn(
                async move { wal.advance_cursor("exec-1").await },
            ));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap());
        }

        results.sort();
        let expected: Vec<u64> = (1..=10).collect();
        assert_eq!(results, expected);
    }
}
