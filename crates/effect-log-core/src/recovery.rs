use crate::types::{
    CompletionRecord, EffectKind, IntentRecord, ReadRecoveryPolicy, RecoveryAction,
};

/// Determines the recovery action for a given effect based on its kind and
/// completion status.
///
/// This is a pure function — no I/O, no side effects. It implements the
/// recovery strategy matrix from the design doc.
pub fn recovery_strategy(
    record: &IntentRecord,
    completion: Option<&CompletionRecord>,
    read_policy: ReadRecoveryPolicy,
) -> RecoveryAction {
    match (&record.effect_kind, completion) {
        // Completed → return sealed result (for all non-ReadOnly kinds)
        (EffectKind::IrreversibleWrite, Some(_)) => RecoveryAction::ReturnSealed,
        (EffectKind::ReadThenWrite, Some(_)) => RecoveryAction::ReturnSealed,
        (EffectKind::IdempotentWrite, Some(_)) => RecoveryAction::ReturnSealed,
        (EffectKind::Compensatable, Some(_)) => RecoveryAction::ReturnSealed,

        // ReadOnly completed → policy-dependent
        (EffectKind::ReadOnly, Some(_)) => match read_policy {
            ReadRecoveryPolicy::ReplayFresh => RecoveryAction::Replay,
            ReadRecoveryPolicy::ReturnSealed => RecoveryAction::ReturnSealed,
        },

        // No completion → crashed during execution
        (EffectKind::ReadOnly, None) => RecoveryAction::Replay,
        (EffectKind::IdempotentWrite, None) => RecoveryAction::Replay,
        (EffectKind::Compensatable, None) => RecoveryAction::CompensateThenReplay,
        (EffectKind::IrreversibleWrite, None) => RecoveryAction::RequireHumanReview,
        (EffectKind::ReadThenWrite, None) => RecoveryAction::RequireHumanReview,
    }
}

/// A recovery plan for an entire execution, produced by scanning the WAL.
#[derive(Debug)]
pub struct RecoveryPlan {
    pub actions: Vec<(IntentRecord, Option<CompletionRecord>, RecoveryAction)>,
    /// The sequence number to resume from (first incomplete or requiring action).
    pub resume_from: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;

    fn make_intent(kind: EffectKind) -> IntentRecord {
        IntentRecord {
            effect_id: EffectId::new(),
            tool_call_id: "call-1".into(),
            tool_name: "test_tool".into(),
            effect_kind: kind,
            input: serde_json::json!({}),
            idempotency_key: None,
            impact_scope: None,
            timestamp: Utc::now(),
            cursor: ExecutionCursor {
                execution_id: "exec-1".into(),
                sequence_number: 1,
                parent_effect_id: None,
            },
        }
    }

    fn make_completion(effect_id: EffectId) -> CompletionRecord {
        CompletionRecord {
            effect_id,
            outcome: Outcome::Success,
            sealed_response: serde_json::json!({"result": "ok"}),
            version_fingerprint: None,
            has_irreversible_change: false,
            compensation_info: None,
            completed_at: Utc::now(),
        }
    }

    // === Completed effects (with completion record) ===

    #[test]
    fn readonly_completed_replay_fresh() {
        let intent = make_intent(EffectKind::ReadOnly);
        let completion = make_completion(intent.effect_id);
        assert_eq!(
            recovery_strategy(&intent, Some(&completion), ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::Replay
        );
    }

    #[test]
    fn readonly_completed_return_sealed() {
        let intent = make_intent(EffectKind::ReadOnly);
        let completion = make_completion(intent.effect_id);
        assert_eq!(
            recovery_strategy(&intent, Some(&completion), ReadRecoveryPolicy::ReturnSealed),
            RecoveryAction::ReturnSealed
        );
    }

    #[test]
    fn idempotent_write_completed() {
        let intent = make_intent(EffectKind::IdempotentWrite);
        let completion = make_completion(intent.effect_id);
        assert_eq!(
            recovery_strategy(&intent, Some(&completion), ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::ReturnSealed
        );
    }

    #[test]
    fn compensatable_completed() {
        let intent = make_intent(EffectKind::Compensatable);
        let completion = make_completion(intent.effect_id);
        assert_eq!(
            recovery_strategy(&intent, Some(&completion), ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::ReturnSealed
        );
    }

    #[test]
    fn irreversible_write_completed() {
        let intent = make_intent(EffectKind::IrreversibleWrite);
        let completion = make_completion(intent.effect_id);
        assert_eq!(
            recovery_strategy(&intent, Some(&completion), ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::ReturnSealed
        );
    }

    #[test]
    fn read_then_write_completed() {
        let intent = make_intent(EffectKind::ReadThenWrite);
        let completion = make_completion(intent.effect_id);
        assert_eq!(
            recovery_strategy(&intent, Some(&completion), ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::ReturnSealed
        );
    }

    // === Crashed effects (no completion record) ===

    #[test]
    fn readonly_crashed() {
        let intent = make_intent(EffectKind::ReadOnly);
        assert_eq!(
            recovery_strategy(&intent, None, ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::Replay
        );
    }

    #[test]
    fn readonly_crashed_sealed_policy() {
        let intent = make_intent(EffectKind::ReadOnly);
        // Even with ReturnSealed policy, a crashed ReadOnly replays
        assert_eq!(
            recovery_strategy(&intent, None, ReadRecoveryPolicy::ReturnSealed),
            RecoveryAction::Replay
        );
    }

    #[test]
    fn idempotent_write_crashed() {
        let intent = make_intent(EffectKind::IdempotentWrite);
        assert_eq!(
            recovery_strategy(&intent, None, ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::Replay
        );
    }

    #[test]
    fn compensatable_crashed() {
        let intent = make_intent(EffectKind::Compensatable);
        assert_eq!(
            recovery_strategy(&intent, None, ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::CompensateThenReplay
        );
    }

    #[test]
    fn irreversible_write_crashed() {
        let intent = make_intent(EffectKind::IrreversibleWrite);
        assert_eq!(
            recovery_strategy(&intent, None, ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::RequireHumanReview
        );
    }

    #[test]
    fn read_then_write_crashed() {
        let intent = make_intent(EffectKind::ReadThenWrite);
        assert_eq!(
            recovery_strategy(&intent, None, ReadRecoveryPolicy::ReplayFresh),
            RecoveryAction::RequireHumanReview
        );
    }

    // === Verify all 5 kinds x 2 completion states x relevant policies ===

    #[test]
    fn exhaustive_matrix() {
        let kinds = [
            EffectKind::ReadOnly,
            EffectKind::IdempotentWrite,
            EffectKind::Compensatable,
            EffectKind::IrreversibleWrite,
            EffectKind::ReadThenWrite,
        ];
        let policies = [
            ReadRecoveryPolicy::ReplayFresh,
            ReadRecoveryPolicy::ReturnSealed,
        ];

        for kind in &kinds {
            let intent = make_intent(*kind);
            let completion = make_completion(intent.effect_id);

            for policy in &policies {
                // With completion
                let action = recovery_strategy(&intent, Some(&completion), *policy);
                match kind {
                    EffectKind::ReadOnly => match policy {
                        ReadRecoveryPolicy::ReplayFresh => {
                            assert_eq!(action, RecoveryAction::Replay)
                        }
                        ReadRecoveryPolicy::ReturnSealed => {
                            assert_eq!(action, RecoveryAction::ReturnSealed)
                        }
                    },
                    _ => assert_eq!(action, RecoveryAction::ReturnSealed),
                }

                // Without completion (crashed)
                let action = recovery_strategy(&intent, None, *policy);
                match kind {
                    EffectKind::ReadOnly => assert_eq!(action, RecoveryAction::Replay),
                    EffectKind::IdempotentWrite => assert_eq!(action, RecoveryAction::Replay),
                    EffectKind::Compensatable => {
                        assert_eq!(action, RecoveryAction::CompensateThenReplay)
                    }
                    EffectKind::IrreversibleWrite => {
                        assert_eq!(action, RecoveryAction::RequireHumanReview)
                    }
                    EffectKind::ReadThenWrite => {
                        assert_eq!(action, RecoveryAction::RequireHumanReview)
                    }
                }
            }
        }
    }
}
