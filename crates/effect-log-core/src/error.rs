use crate::types::EffectId;

/// Errors originating from the storage backend.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("storage I/O error: {0}")]
    Io(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("record not found: {0}")]
    NotFound(String),

    #[error("duplicate record: effect_id={0}")]
    Duplicate(EffectId),

    #[cfg(feature = "sqlite")]
    #[error("sqlite error: {0}")]
    Sqlite(String),
}

/// Top-level errors for the effect-log system.
#[derive(Debug, thiserror::Error)]
pub enum EffectLogError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("tool not found: {0}")]
    ToolNotFound(String),

    #[error("tool already registered: {0}")]
    ToolAlreadyRegistered(String),

    #[error(
        "tool name mismatch at cursor position {sequence}: expected '{expected}', got '{actual}'"
    )]
    ToolMismatch {
        sequence: u64,
        expected: String,
        actual: String,
    },

    #[error("execution requires human review for effect {effect_id}")]
    RequiresHumanReview { effect_id: EffectId },

    #[error("tool execution failed: {0}")]
    ToolExecutionFailed(String),

    #[error("builder error: {0}")]
    Builder(String),

    #[error("compensation failed for effect {effect_id}: {reason}")]
    CompensationFailed { effect_id: EffectId, reason: String },

    #[error("tool execution timed out for effect {effect_id}")]
    ToolTimeout { effect_id: EffectId },

    #[error("tool panicked for effect {effect_id}: {message}")]
    ToolPanicked { effect_id: EffectId, message: String },

    #[error("duplicate idempotency key: {key}")]
    DuplicateIdempotencyKey { key: String },

    #[error("internal invariant violated: {0}")]
    InternalInvariant(String),
}

pub type Result<T> = std::result::Result<T, EffectLogError>;
