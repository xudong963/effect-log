use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use async_trait::async_trait;

use crate::error::StorageError;
use crate::types::{CompletionRecord, EffectId, IntentRecord};

use super::{EffectStore, WalEntry};

/// In-memory storage backend for testing and ephemeral workloads.
#[derive(Debug, Clone)]
pub struct InMemoryStore {
    intents: Arc<RwLock<HashMap<EffectId, IntentRecord>>>,
    completions: Arc<RwLock<HashMap<EffectId, CompletionRecord>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            intents: Arc::new(RwLock::new(HashMap::new())),
            completions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EffectStore for InMemoryStore {
    async fn write_intent(&self, record: &IntentRecord) -> Result<(), StorageError> {
        let mut intents = self.intents.write().await;
        if intents.contains_key(&record.effect_id) {
            return Err(StorageError::Duplicate(record.effect_id));
        }
        intents.insert(record.effect_id, record.clone());
        Ok(())
    }

    async fn write_completion(&self, record: &CompletionRecord) -> Result<(), StorageError> {
        let mut completions = self.completions.write().await;
        // Insert or overwrite (matches SQLite INSERT OR REPLACE behavior)
        completions.insert(record.effect_id, record.clone());
        Ok(())
    }

    async fn load_execution(&self, execution_id: &str) -> Result<Vec<WalEntry>, StorageError> {
        let intents = self.intents.read().await;
        let completions = self.completions.read().await;

        let mut entries: Vec<WalEntry> = Vec::new();

        // Collect intents for this execution, sorted by sequence_number
        let mut exec_intents: Vec<&IntentRecord> = intents
            .values()
            .filter(|i| i.cursor.execution_id == execution_id)
            .collect();
        exec_intents.sort_by_key(|i| i.cursor.sequence_number);

        for intent in exec_intents {
            entries.push(WalEntry::Intent(intent.clone()));
            if let Some(completion) = completions.get(&intent.effect_id) {
                entries.push(WalEntry::Completion(completion.clone()));
            }
        }

        Ok(entries)
    }

    async fn lookup_intent(
        &self,
        execution_id: &str,
        sequence_number: u64,
    ) -> Result<Option<IntentRecord>, StorageError> {
        let intents = self.intents.read().await;
        let found = intents.values().find(|i| {
            i.cursor.execution_id == execution_id && i.cursor.sequence_number == sequence_number
        });
        Ok(found.cloned())
    }

    async fn get_completion(
        &self,
        effect_id: &EffectId,
    ) -> Result<Option<CompletionRecord>, StorageError> {
        let completions = self.completions.read().await;
        Ok(completions.get(effect_id).cloned())
    }

    async fn lookup_by_idempotency_key(
        &self,
        execution_id: &str,
        key: &str,
    ) -> Result<Option<IntentRecord>, StorageError> {
        let intents = self.intents.read().await;
        let found = intents.values().find(|i| {
            i.cursor.execution_id == execution_id
                && i.idempotency_key.as_deref() == Some(key)
        });
        Ok(found.cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;

    fn make_intent(execution_id: &str, seq: u64, tool: &str) -> IntentRecord {
        IntentRecord {
            effect_id: EffectId::new(),
            tool_call_id: format!("call-{seq}"),
            tool_name: tool.into(),
            effect_kind: EffectKind::ReadOnly,
            input: serde_json::json!({"seq": seq}),
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
            sealed_response: serde_json::json!({"ok": true}),
            version_fingerprint: None,
            has_irreversible_change: false,
            compensation_info: None,
            completed_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn write_and_read_intent() {
        let store = InMemoryStore::new();
        let intent = make_intent("exec-1", 1, "read_file");

        store.write_intent(&intent).await.unwrap();

        let found = store.lookup_intent("exec-1", 1).await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().tool_name, "read_file");
    }

    #[tokio::test]
    async fn duplicate_intent_rejected() {
        let store = InMemoryStore::new();
        let intent = make_intent("exec-1", 1, "read_file");

        store.write_intent(&intent).await.unwrap();
        let result = store.write_intent(&intent).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn write_and_read_completion() {
        let store = InMemoryStore::new();
        let intent = make_intent("exec-1", 1, "read_file");
        let effect_id = intent.effect_id;

        store.write_intent(&intent).await.unwrap();

        let completion = make_completion(effect_id);
        store.write_completion(&completion).await.unwrap();

        let found = store.get_completion(&effect_id).await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().outcome, Outcome::Success);
    }

    #[tokio::test]
    async fn load_execution_ordered() {
        let store = InMemoryStore::new();

        // Write intents out of order
        let i3 = make_intent("exec-1", 3, "tool_c");
        let i1 = make_intent("exec-1", 1, "tool_a");
        let i2 = make_intent("exec-1", 2, "tool_b");

        store.write_intent(&i3).await.unwrap();
        store.write_intent(&i1).await.unwrap();
        store.write_intent(&i2).await.unwrap();

        // Complete only the first two
        store
            .write_completion(&make_completion(i1.effect_id))
            .await
            .unwrap();
        store
            .write_completion(&make_completion(i2.effect_id))
            .await
            .unwrap();

        let entries = store.load_execution("exec-1").await.unwrap();
        // 3 intents + 2 completions = 5 entries
        assert_eq!(entries.len(), 5);

        // Verify order: intent1, completion1, intent2, completion2, intent3
        match &entries[0] {
            WalEntry::Intent(i) => assert_eq!(i.cursor.sequence_number, 1),
            _ => panic!("expected intent"),
        }
        match &entries[1] {
            WalEntry::Completion(c) => assert_eq!(c.effect_id, i1.effect_id),
            _ => panic!("expected completion"),
        }
        match &entries[4] {
            WalEntry::Intent(i) => assert_eq!(i.cursor.sequence_number, 3),
            _ => panic!("expected intent"),
        }
    }

    #[tokio::test]
    async fn lookup_missing_returns_none() {
        let store = InMemoryStore::new();
        assert!(store.lookup_intent("nope", 1).await.unwrap().is_none());
        assert!(store
            .get_completion(&EffectId::new())
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn separate_executions_isolated() {
        let store = InMemoryStore::new();

        let i1 = make_intent("exec-1", 1, "tool_a");
        let i2 = make_intent("exec-2", 1, "tool_b");

        store.write_intent(&i1).await.unwrap();
        store.write_intent(&i2).await.unwrap();

        let entries = store.load_execution("exec-1").await.unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            WalEntry::Intent(i) => assert_eq!(i.tool_name, "tool_a"),
            _ => panic!("expected intent"),
        }
    }
}
