pub mod memory;
#[cfg(feature = "sqlite")]
pub mod sqlite;

use crate::error::StorageError;
use crate::types::{CompletionRecord, EffectId, IntentRecord};
use async_trait::async_trait;

/// A WAL entry is either an intent or a completion.
#[derive(Debug, Clone)]
pub enum WalEntry {
    Intent(IntentRecord),
    Completion(CompletionRecord),
}

/// Pluggable storage backend for the write-ahead log.
///
/// Implementations must guarantee that `write_intent` and `write_completion`
/// are durable before returning (e.g., fsync for file-based backends).
#[async_trait]
pub trait EffectStore: Send + Sync {
    /// Durably write an intent record.
    async fn write_intent(&self, record: &IntentRecord) -> std::result::Result<(), StorageError>;

    /// Durably write a completion record.
    async fn write_completion(
        &self,
        record: &CompletionRecord,
    ) -> std::result::Result<(), StorageError>;

    /// Load all records for a given execution, ordered by sequence_number.
    async fn load_execution(
        &self,
        execution_id: &str,
    ) -> std::result::Result<Vec<WalEntry>, StorageError>;

    /// Look up a single intent by execution_id + sequence_number.
    async fn lookup_intent(
        &self,
        execution_id: &str,
        sequence_number: u64,
    ) -> std::result::Result<Option<IntentRecord>, StorageError>;

    /// Look up the completion for a given effect_id.
    async fn get_completion(
        &self,
        effect_id: &EffectId,
    ) -> std::result::Result<Option<CompletionRecord>, StorageError>;

    /// Look up an intent by idempotency key within an execution.
    /// Returns the first matching intent (if any).
    async fn lookup_by_idempotency_key(
        &self,
        execution_id: &str,
        key: &str,
    ) -> std::result::Result<Option<IntentRecord>, StorageError>;
}
