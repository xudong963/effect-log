use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Globally unique, time-ordered identifier for an effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectId(pub Uuid);

impl EffectId {
    /// Create a new time-ordered (UUIDv7) effect id.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for EffectId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for EffectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Semantic classification of a tool's side effects.
///
/// Every tool declares its `EffectKind` at registration time, and this single
/// declaration drives all recovery behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectKind {
    /// Pure read, can be replayed freely.
    ReadOnly,
    /// Idempotent write, safe to replay with same key.
    IdempotentWrite,
    /// Reversible mutation with a known compensation action.
    Compensatable,
    /// Non-reversible write, sealed on completion.
    IrreversibleWrite,
    /// Read-then-write compound operation.
    ReadThenWrite,
}

/// Position of a tool call within the execution graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionCursor {
    /// Unique identifier for the overall execution run.
    pub execution_id: String,
    /// Monotonically increasing sequence number within the execution.
    pub sequence_number: u64,
    /// Optional parent effect_id for nested/sub-tool-call scenarios.
    pub parent_effect_id: Option<EffectId>,
}

/// Written before a tool executes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRecord {
    pub effect_id: EffectId,
    pub tool_call_id: String,
    pub tool_name: String,
    pub effect_kind: EffectKind,
    pub input: serde_json::Value,
    /// For IdempotentWrite tools: the computed idempotency key.
    pub idempotency_key: Option<String>,
    pub impact_scope: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub cursor: ExecutionCursor,
}

/// Written after a tool finishes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRecord {
    pub effect_id: EffectId,
    pub outcome: Outcome,
    pub sealed_response: serde_json::Value,
    pub version_fingerprint: Option<String>,
    /// Whether an irreversible external change occurred during this call.
    pub has_irreversible_change: bool,
    /// For Compensatable effects: serialized compensation info.
    pub compensation_info: Option<serde_json::Value>,
    pub completed_at: DateTime<Utc>,
}

/// The outcome of a tool execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Success,
    Failed { error: String, retryable: bool },
    Timeout,
    Panicked { message: String },
}

/// Policy for recovering completed ReadOnly effects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadRecoveryPolicy {
    /// Always replay for fresh data (default).
    #[default]
    ReplayFresh,
    /// Return the sealed snapshot.
    ReturnSealed,
}

/// Action the recovery engine should take for a given effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Re-execute the tool call.
    Replay,
    /// Return the sealed response from the completion record.
    ReturnSealed,
    /// Invoke compensation, then re-execute.
    CompensateThenReplay,
    /// Cannot determine outcome — require human decision.
    RequireHumanReview,
}

/// A batch of parallel tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelBatch {
    pub batch_id: String,
    pub effect_ids: Vec<EffectId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_id_uniqueness() {
        let ids: Vec<EffectId> = (0..100).map(|_| EffectId::new()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn effect_id_time_ordered() {
        let a = EffectId::new();
        let b = EffectId::new();
        // UUIDv7 embeds timestamp — later IDs are lexicographically greater
        assert!(a.0 <= b.0);
    }

    #[test]
    fn effect_kind_serde_roundtrip() {
        let kinds = vec![
            EffectKind::ReadOnly,
            EffectKind::IdempotentWrite,
            EffectKind::Compensatable,
            EffectKind::IrreversibleWrite,
            EffectKind::ReadThenWrite,
        ];
        for kind in kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let back: EffectKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, back);
        }
    }

    #[test]
    fn outcome_serde_roundtrip() {
        let outcomes = vec![
            Outcome::Success,
            Outcome::Failed {
                error: "boom".into(),
                retryable: true,
            },
            Outcome::Timeout,
            Outcome::Panicked {
                message: "panic!".into(),
            },
        ];
        for outcome in outcomes {
            let json = serde_json::to_string(&outcome).unwrap();
            let back: Outcome = serde_json::from_str(&json).unwrap();
            assert_eq!(outcome, back);
        }
    }

    #[test]
    fn intent_record_serde_roundtrip() {
        let record = IntentRecord {
            effect_id: EffectId::new(),
            tool_call_id: "call-1".into(),
            tool_name: "send_email".into(),
            effect_kind: EffectKind::IrreversibleWrite,
            input: serde_json::json!({"to": "a@b.com"}),
            idempotency_key: None,
            impact_scope: Some("email".into()),
            timestamp: Utc::now(),
            cursor: ExecutionCursor {
                execution_id: "exec-1".into(),
                sequence_number: 1,
                parent_effect_id: None,
            },
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: IntentRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record.effect_id, back.effect_id);
        assert_eq!(record.tool_name, back.tool_name);
        assert_eq!(record.effect_kind, back.effect_kind);
    }

    #[test]
    fn completion_record_serde_roundtrip() {
        let record = CompletionRecord {
            effect_id: EffectId::new(),
            outcome: Outcome::Success,
            sealed_response: serde_json::json!({"status": "sent"}),
            version_fingerprint: Some("v1".into()),
            has_irreversible_change: true,
            compensation_info: None,
            completed_at: Utc::now(),
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: CompletionRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record.effect_id, back.effect_id);
        assert_eq!(record.outcome, back.outcome);
    }
}
