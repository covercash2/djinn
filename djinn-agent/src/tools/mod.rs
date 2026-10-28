//! Tools the agent can call.
//!
//! A tool has a name, a description and a JSON-schema parameter description (all shown to the
//! model), and an async `call` that takes the model's JSON arguments and returns text. The built-in
//! tools are read-only: they never write or delete files, run processes, or use the network. A tool
//! that does any of those needs its own change that also specifies user confirmation.

use std::sync::Arc;

use djinn_core::chat::ToolSpec;
use djinn_core::values::ToolName;
use futures::future::BoxFuture;
use serde_json::Value;
use thiserror::Error;

mod calculator;
mod current_time;
mod read_file;

pub use calculator::Calculator;
pub use current_time::{CurrentTime, rfc3339_utc};
pub use read_file::{MAX_READ_BYTES, ReadFile};

/// Why a tool call failed. The message is shown to the model, so it should say what to fix.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ToolError {
    /// The arguments do not match the tool's declared parameters.
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    /// The arguments were fine but the tool could not do what was asked.
    #[error("{0}")]
    Failed(String),
}

/// Something the model can ask the agent to do.
pub trait Tool: Send + Sync {
    /// Unique name the model uses to call the tool.
    fn name(&self) -> &ToolName;

    /// What the tool does and when to use it, written for the model.
    fn description(&self) -> &str;

    /// JSON Schema (an object schema) describing the arguments.
    fn parameters(&self) -> Value;

    /// Runs the tool. `arguments` is the JSON object the model produced.
    fn call(&self, arguments: Value) -> BoxFuture<'_, Result<String, ToolError>>;
}

/// Registration failed because a tool with this name already exists.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("a tool named `{0}` is already registered")]
pub struct DuplicateTool(pub ToolName);

/// The tools available to an agent, in registration order.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a tool; fails if its name is taken.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), DuplicateTool> {
        if self.get(tool.name().as_str()).is_some() {
            return Err(DuplicateTool(tool.name().clone()));
        }
        self.tools.push(tool);
        Ok(())
    }

    /// Adds a tool, returning the registry (for building one in a single expression).
    pub fn with(mut self, tool: Arc<dyn Tool>) -> Result<Self, DuplicateTool> {
        self.register(tool)?;
        Ok(self)
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|tool| tool.name().as_str() == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name().as_str()).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The declarations sent to the model with every request.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .map(|tool| {
                ToolSpec::function(tool.name().clone(), tool.description(), tool.parameters())
            })
            .collect()
    }

    /// The built-in read-only tools: `current_time`, `calculator`, and `read_file` confined to
    /// `root`.
    pub fn builtin(root: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let read_file = ReadFile::new(root)?;
        let mut registry = Self::new();
        for tool in [
            Arc::new(CurrentTime) as Arc<dyn Tool>,
            Arc::new(Calculator),
            Arc::new(read_file),
        ] {
            registry
                .register(tool)
                .expect("built-in tool names are distinct");
        }
        Ok(registry)
    }
}

/// Reads a required string argument.
pub(crate) fn string_arg<'a>(arguments: &'a Value, name: &str) -> Result<&'a str, ToolError> {
    match arguments.get(name) {
        None | Some(Value::Null) => Err(ToolError::InvalidArguments(format!(
            "missing required parameter `{name}`"
        ))),
        Some(Value::String(text)) => Ok(text),
        Some(_) => Err(ToolError::InvalidArguments(format!(
            "`{name}` must be a string"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Named(ToolName);

    impl Tool for Named {
        fn name(&self) -> &ToolName {
            &self.0
        }
        fn description(&self) -> &str {
            "a test tool"
        }
        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        fn call(&self, _arguments: Value) -> BoxFuture<'_, Result<String, ToolError>> {
            Box::pin(async { Ok("ok".to_owned()) })
        }
    }

    #[test]
    fn duplicate_names_are_rejected_with_the_name() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(Named(ToolName::from_static("a"))))
            .unwrap();
        assert_eq!(
            registry.register(Arc::new(Named(ToolName::from_static("a")))),
            Err(DuplicateTool(ToolName::from_static("a")))
        );
        assert_eq!(
            registry.names(),
            vec!["a"],
            "the first registration is kept"
        );
    }

    #[test]
    fn specs_list_every_tool_in_registration_order() {
        let registry = ToolRegistry::new()
            .with(Arc::new(Named(ToolName::from_static("b"))))
            .unwrap()
            .with(Arc::new(Named(ToolName::from_static("a"))))
            .unwrap();
        let specs = registry.specs();
        assert_eq!(
            specs
                .iter()
                .map(|s| s.function.name.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "a"]
        );
        assert_eq!(
            specs[0].function.description.as_deref(),
            Some("a test tool")
        );
        assert_eq!(
            specs[0].function.parameters,
            Some(json!({"type": "object", "properties": {}}))
        );
    }

    #[test]
    fn lookup_finds_registered_tools_only() {
        let registry = ToolRegistry::new()
            .with(Arc::new(Named(ToolName::from_static("a"))))
            .unwrap();
        assert!(registry.get("a").is_some());
        assert!(registry.get("missing").is_none());
        assert!(!registry.is_empty());
        assert!(ToolRegistry::new().is_empty());
    }

    #[test]
    fn builtin_registry_has_the_three_read_only_tools() {
        let dir = tempfile::tempdir().unwrap();
        let registry = ToolRegistry::builtin(dir.path()).unwrap();
        assert_eq!(
            registry.names(),
            vec!["current_time", "calculator", "read_file"]
        );
        for spec in registry.specs() {
            let parameters = spec.function.parameters.unwrap();
            assert_eq!(
                parameters["type"], "object",
                "{} needs an object schema",
                spec.function.name
            );
            assert!(spec.function.description.is_some_and(|d| !d.is_empty()));
        }
    }

    #[test]
    fn builtin_fails_when_the_root_is_not_a_directory() {
        assert!(ToolRegistry::builtin("/definitely/not/a/real/dir").is_err());
    }

    #[test]
    fn string_arg_reports_missing_and_wrongly_typed_values() {
        let args = json!({"a": "x", "b": 1, "c": null});
        assert_eq!(string_arg(&args, "a"), Ok("x"));
        assert!(matches!(
            string_arg(&args, "missing"),
            Err(ToolError::InvalidArguments(m)) if m.contains("missing required parameter `missing`")
        ));
        assert!(matches!(
            string_arg(&args, "c"),
            Err(ToolError::InvalidArguments(m)) if m.contains("missing")
        ));
        assert!(matches!(
            string_arg(&args, "b"),
            Err(ToolError::InvalidArguments(m)) if m.contains("must be a string")
        ));
    }
}
