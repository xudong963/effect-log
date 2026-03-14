use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{EffectLogError, Result};
use crate::types::{EffectKind, ReadRecoveryPolicy};

/// A tool function: takes JSON input, returns JSON output.
pub type ToolFn = Arc<
    dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = serde_json::Value> + Send>>
        + Send
        + Sync,
>;

/// A compensation function: takes the original input, returns a result.
pub type CompensateFn = Arc<
    dyn Fn(
            serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = std::result::Result<(), String>> + Send>>
        + Send
        + Sync,
>;

/// An idempotency key generator: takes the input, returns a key string.
pub type IdempotencyKeyFn = Arc<dyn Fn(&serde_json::Value) -> String + Send + Sync>;

/// A registered tool with its semantic classification and optional handlers.
#[derive(Clone)]
pub struct RegisteredTool {
    pub name: String,
    pub effect_kind: EffectKind,
    pub func: ToolFn,
    pub compensate: Option<CompensateFn>,
    pub idempotency_key_fn: Option<IdempotencyKeyFn>,
    pub read_recovery_policy: ReadRecoveryPolicy,
    pub impact_scope: Option<String>,
    /// Per-tool execution timeout. If the tool does not complete within this
    /// duration, the execution is aborted and a `Timeout` outcome is recorded.
    pub timeout: Option<Duration>,
}

impl std::fmt::Debug for RegisteredTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredTool")
            .field("name", &self.name)
            .field("effect_kind", &self.effect_kind)
            .finish()
    }
}

/// Builder for constructing a RegisteredTool.
pub struct ToolBuilder {
    name: String,
    effect_kind: EffectKind,
    func: Option<ToolFn>,
    compensate: Option<CompensateFn>,
    idempotency_key_fn: Option<IdempotencyKeyFn>,
    read_recovery_policy: ReadRecoveryPolicy,
    impact_scope: Option<String>,
    timeout: Option<Duration>,
}

impl ToolBuilder {
    pub fn new(name: impl Into<String>, effect_kind: EffectKind) -> Self {
        Self {
            name: name.into(),
            effect_kind,
            func: None,
            compensate: None,
            idempotency_key_fn: None,
            read_recovery_policy: ReadRecoveryPolicy::default(),
            impact_scope: None,
            timeout: None,
        }
    }

    pub fn func(mut self, f: ToolFn) -> Self {
        self.func = Some(f);
        self
    }

    pub fn compensate(mut self, f: CompensateFn) -> Self {
        self.compensate = Some(f);
        self
    }

    pub fn idempotency_key(mut self, f: IdempotencyKeyFn) -> Self {
        self.idempotency_key_fn = Some(f);
        self
    }

    pub fn read_recovery_policy(mut self, policy: ReadRecoveryPolicy) -> Self {
        self.read_recovery_policy = policy;
        self
    }

    pub fn impact_scope(mut self, scope: impl Into<String>) -> Self {
        self.impact_scope = Some(scope.into());
        self
    }

    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    pub fn build(self) -> Result<RegisteredTool> {
        let func = self
            .func
            .ok_or_else(|| EffectLogError::Builder("tool function is required".into()))?;

        // Validate: Compensatable requires a compensation function
        if self.effect_kind == EffectKind::Compensatable && self.compensate.is_none() {
            return Err(EffectLogError::Builder(
                "Compensatable tools must provide a compensate function".into(),
            ));
        }

        Ok(RegisteredTool {
            name: self.name,
            effect_kind: self.effect_kind,
            func,
            compensate: self.compensate,
            idempotency_key_fn: self.idempotency_key_fn,
            read_recovery_policy: self.read_recovery_policy,
            impact_scope: self.impact_scope,
            timeout: self.timeout,
        })
    }
}

/// Registry of tools with their semantic classifications.
pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. Returns error if a tool with the same name already exists.
    pub fn register(&mut self, tool: RegisteredTool) -> Result<()> {
        if self.tools.contains_key(&tool.name) {
            return Err(EffectLogError::ToolAlreadyRegistered(tool.name));
        }
        self.tools.insert(tool.name.clone(), tool);
        Ok(())
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name)
    }

    /// List all registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_fn() -> ToolFn {
        Arc::new(|_input| Box::pin(async { serde_json::json!(null) }))
    }

    fn noop_compensate() -> CompensateFn {
        Arc::new(|_input| Box::pin(async { Ok(()) }))
    }

    #[test]
    fn build_readonly_tool() {
        let tool = ToolBuilder::new("read_file", EffectKind::ReadOnly)
            .func(noop_fn())
            .build()
            .unwrap();

        assert_eq!(tool.name, "read_file");
        assert_eq!(tool.effect_kind, EffectKind::ReadOnly);
        assert!(tool.compensate.is_none());
    }

    #[test]
    fn build_compensatable_requires_compensate_fn() {
        let result = ToolBuilder::new("create_vm", EffectKind::Compensatable)
            .func(noop_fn())
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("compensate function"));
    }

    #[test]
    fn build_compensatable_with_compensate_fn() {
        let tool = ToolBuilder::new("create_vm", EffectKind::Compensatable)
            .func(noop_fn())
            .compensate(noop_compensate())
            .build()
            .unwrap();

        assert!(tool.compensate.is_some());
    }

    #[test]
    fn build_requires_func() {
        let result = ToolBuilder::new("empty", EffectKind::ReadOnly).build();
        assert!(result.is_err());
    }

    #[test]
    fn registry_register_and_lookup() {
        let mut registry = ToolRegistry::new();
        let tool = ToolBuilder::new("read_file", EffectKind::ReadOnly)
            .func(noop_fn())
            .build()
            .unwrap();

        registry.register(tool).unwrap();

        assert_eq!(registry.len(), 1);
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn registry_rejects_duplicate() {
        let mut registry = ToolRegistry::new();
        let tool1 = ToolBuilder::new("read_file", EffectKind::ReadOnly)
            .func(noop_fn())
            .build()
            .unwrap();
        let tool2 = ToolBuilder::new("read_file", EffectKind::ReadOnly)
            .func(noop_fn())
            .build()
            .unwrap();

        registry.register(tool1).unwrap();
        let result = registry.register(tool2);
        assert!(result.is_err());
    }

    #[test]
    fn registry_tool_names() {
        let mut registry = ToolRegistry::new();
        for name in ["a", "b", "c"] {
            let tool = ToolBuilder::new(name, EffectKind::ReadOnly)
                .func(noop_fn())
                .build()
                .unwrap();
            registry.register(tool).unwrap();
        }

        let mut names = registry.tool_names();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn builder_with_idempotency_key() {
        let tool = ToolBuilder::new("upsert", EffectKind::IdempotentWrite)
            .func(noop_fn())
            .idempotency_key(Arc::new(|input| {
                input["id"].as_str().unwrap_or("unknown").to_string()
            }))
            .build()
            .unwrap();

        let key_fn = tool.idempotency_key_fn.as_ref().unwrap();
        let key = key_fn(&serde_json::json!({"id": "abc"}));
        assert_eq!(key, "abc");
    }

    #[test]
    fn builder_with_impact_scope() {
        let tool = ToolBuilder::new("send_email", EffectKind::IrreversibleWrite)
            .func(noop_fn())
            .impact_scope("email:external")
            .build()
            .unwrap();

        assert_eq!(tool.impact_scope.as_deref(), Some("email:external"));
    }
}
