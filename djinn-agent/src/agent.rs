//! The agent loop: ask the model, run the tools it requests, feed the results back, repeat.

use std::time::{Duration, Instant};

use djinn_core::chat::{ChatMessage, ChatRequest, ToolCall};
use djinn_core::values::{
    MaxTokens, SystemPrompt, Temperature, ToolArguments, ToolCallId, ToolName,
};
use futures::StreamExt;
use futures::future::BoxFuture;
use thiserror::Error;

use crate::backend::{Backend, BackendError, BackendEvent};
use crate::tools::ToolRegistry;

/// Used when no system prompt is configured.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a concise, helpful assistant. Use the provided tools when they help you answer accurately (for calculations, the current time, or reading files); otherwise answer directly. After using a tool, answer the user using its result.";

/// Settings for an [`Agent`].
#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub system_prompt: SystemPrompt,
    /// The most model calls one user message may cause.
    pub max_steps: usize,
    /// How long one tool may run before it is cancelled.
    pub tool_timeout: Duration,
    /// Tool results longer than this many bytes are cut off.
    pub max_tool_output: usize,
    pub temperature: Option<Temperature>,
    pub max_tokens: Option<MaxTokens>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: SystemPrompt::from_static(DEFAULT_SYSTEM_PROMPT),
            max_steps: 8,
            tool_timeout: Duration::from_secs(10),
            max_tool_output: 8 * 1024,
            temperature: None,
            max_tokens: None,
        }
    }
}

/// What happens during a turn, for a user interface to show.
#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    /// A model call is about to start: the `step`-th of this turn (counting from 1).
    StepStarted { step: usize },
    /// A piece of the model's reply, as it streams in.
    Text(String),
    /// The model asked for a tool and it is about to run. `arguments` is the JSON the model wrote.
    ToolStarted {
        id: ToolCallId,
        name: ToolName,
        arguments: ToolArguments,
    },
    /// The user denied a call, so the tool did not run. `reason` is what the user said, if anything.
    ToolDenied {
        id: ToolCallId,
        name: ToolName,
        reason: Option<String>,
    },
    /// A tool call ended. `output` is what the model is shown (an error message when `ok` is false).
    ToolFinished {
        id: ToolCallId,
        name: ToolName,
        ok: bool,
        output: String,
        elapsed: Duration,
    },
    /// The step limit was reached; no further model call will be made this turn.
    StepLimit { steps: usize },
    /// The turn ended successfully.
    TurnFinished,
}

/// A tool call waiting for the user's decision.
#[derive(Clone, Copy, Debug)]
pub struct ApprovalRequest<'a> {
    pub id: &'a ToolCallId,
    pub name: &'a ToolName,
    pub arguments: &'a ToolArguments,
    /// The tool declares that its calls need confirmation (see [`crate::tools::Tool::needs_confirmation`]).
    pub needs_confirmation: bool,
}

/// The user's answer to an [`ApprovalRequest`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Allow,
    /// Do not run the tool. `reason` is free text shown to the model.
    Deny {
        reason: Option<String>,
    },
}

/// Decides whether a requested tool call may run. Asked once per call, in order, before the tool
/// starts; the agent waits for the answer without a time limit, and dropping the turn while it waits
/// cancels it like any other point of the turn.
pub trait Approver: Send + Sync {
    fn decide<'a>(&'a self, request: ApprovalRequest<'a>) -> BoxFuture<'a, Decision>;
}

/// Allows every call. What [`Agent::run_turn`] uses.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllowAll;

impl Approver for AllowAll {
    fn decide<'a>(&'a self, _: ApprovalRequest<'a>) -> BoxFuture<'a, Decision> {
        Box::pin(async { Decision::Allow })
    }
}

/// Why a turn did not produce an answer.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TurnError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(
        "stopped after {0} model calls without a final answer (the step limit); ask again or raise --max-steps"
    )]
    StepLimit(usize),
}

/// A conversation with a model that can use tools.
pub struct Agent<B> {
    backend: B,
    tools: ToolRegistry,
    config: AgentConfig,
    history: Vec<ChatMessage>,
}

impl<B: Backend> Agent<B> {
    pub fn new(backend: B, tools: ToolRegistry, mut config: AgentConfig) -> Self {
        config.max_steps = config.max_steps.max(1);
        let history = vec![ChatMessage::system(config.system_prompt.as_str())];
        Self {
            backend,
            tools,
            config,
            history,
        }
    }

    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// The conversation so far, starting with the system message.
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Forgets the conversation, keeping only the system message.
    pub fn reset(&mut self) {
        self.history.truncate(1);
    }

    /// Answers `user_message`, running tools as the model asks.
    ///
    /// The turn's messages are only added to the history when the turn ends, so a turn that fails
    /// or is cancelled (by dropping this future) leaves the history exactly as it was. The one
    /// exception is hitting the step limit, which keeps what happened so the next message can build
    /// on it.
    pub async fn run_turn(
        &mut self,
        user_message: &str,
        on_event: impl FnMut(AgentEvent),
    ) -> Result<String, TurnError> {
        self.run_turn_with(user_message, on_event, &AllowAll).await
    }

    /// Like [`Agent::run_turn`], but asks `approver` before each tool runs (see [`Approver`]).
    pub async fn run_turn_with(
        &mut self,
        user_message: &str,
        mut on_event: impl FnMut(AgentEvent),
        approver: &dyn Approver,
    ) -> Result<String, TurnError> {
        let mut turn = vec![ChatMessage::user(user_message)];
        let tool_specs = self.tools.specs();

        for step in 1..=self.config.max_steps {
            on_event(AgentEvent::StepStarted { step });
            let request = ChatRequest {
                messages: self.history.iter().chain(&turn).cloned().collect(),
                tools: (!tool_specs.is_empty()).then(|| tool_specs.clone()),
                temperature: self.config.temperature,
                max_tokens: self.config.max_tokens,
                ..ChatRequest::default()
            };

            let mut stream = self.backend.chat(request);
            let mut result = None;
            while let Some(event) = stream.next().await {
                match event? {
                    BackendEvent::Text(text) => on_event(AgentEvent::Text(text)),
                    BackendEvent::Done(done) => result = Some(done),
                }
            }
            drop(stream);
            let result = result.ok_or(BackendError::StreamEndedEarly)?;

            if result.tool_calls.is_empty() {
                let answer = result.content.unwrap_or_default();
                turn.push(ChatMessage::assistant(answer.clone()));
                self.history.append(&mut turn);
                on_event(AgentEvent::TurnFinished);
                return Ok(answer);
            }

            turn.push(ChatMessage::assistant_tool_calls(
                result.content,
                result.tool_calls.clone(),
            ));
            for call in &result.tool_calls {
                let output = self.run_tool(call, approver, &mut on_event).await;
                turn.push(ChatMessage::tool(call.id.clone(), output));
            }

            if step == self.config.max_steps {
                self.history.append(&mut turn);
                on_event(AgentEvent::StepLimit { steps: step });
                return Err(TurnError::StepLimit(step));
            }
        }
        unreachable!("max_steps is at least 1 and the last step returns")
    }

    /// Runs one tool call and returns the text the model is shown. Never fails: every problem is
    /// described to the model so that it can correct itself.
    async fn run_tool(
        &self,
        call: &ToolCall,
        approver: &dyn Approver,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> String {
        let name = call.function.name.clone();
        let tool = self.tools.get(name.as_str());

        // An unknown tool is reported to the model without bothering the user.
        if let Some(tool) = tool {
            let request = ApprovalRequest {
                id: &call.id,
                name: &name,
                arguments: &call.function.arguments,
                needs_confirmation: tool.needs_confirmation(),
            };
            if let Decision::Deny { reason } = approver.decide(request).await {
                let output = denial_message(reason.as_deref());
                on_event(AgentEvent::ToolDenied {
                    id: call.id.clone(),
                    name,
                    reason,
                });
                return output;
            }
        }

        on_event(AgentEvent::ToolStarted {
            id: call.id.clone(),
            name: name.clone(),
            arguments: call.function.arguments.clone(),
        });
        let started = Instant::now();

        let (ok, output) = match tool {
            None => (
                false,
                format!(
                    "Error: unknown tool `{name}`. Available tools: {}",
                    self.tools.names().join(", ")
                ),
            ),
            Some(tool) => {
                let arguments = call.function.arguments.to_value();
                match tokio::time::timeout(self.config.tool_timeout, tool.call(arguments)).await {
                    Ok(Ok(text)) => (true, text),
                    Ok(Err(error)) => (false, format!("Error: {error}")),
                    Err(_) => (
                        false,
                        format!(
                            "Error: the tool timed out after {:.1} seconds",
                            self.config.tool_timeout.as_secs_f64()
                        ),
                    ),
                }
            }
        };

        let output = truncate_output(output, self.config.max_tool_output);
        on_event(AgentEvent::ToolFinished {
            id: call.id.clone(),
            name,
            ok,
            output: output.clone(),
            elapsed: started.elapsed(),
        });
        output
    }
}

/// What the model is told when the user denied a call.
fn denial_message(reason: Option<&str>) -> String {
    match reason.map(str::trim).filter(|reason| !reason.is_empty()) {
        Some(reason) => format!("Error: the user denied this tool call because: {reason}"),
        None => "Error: the user denied this tool call".to_owned(),
    }
}

/// Cuts `text` to at most `limit` bytes (at a character boundary) and says how much was left out.
pub fn truncate_output(text: String, limit: usize) -> String {
    if text.len() <= limit {
        return text;
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n[output truncated: {} bytes omitted]",
        &text[..end],
        text.len() - end
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use std::sync::LazyLock;

    use djinn_core::chat::Role;
    use futures::future::BoxFuture;
    use serde_json::{Value, json};

    use super::*;
    use crate::backend::{ScriptedBackend, ScriptedReply};
    use crate::tools::{Calculator, Tool, ToolError};

    /// Echoes its `text` argument, and records every call.
    struct Echo {
        calls: Arc<Mutex<Vec<Value>>>,
    }

    impl Tool for Echo {
        fn name(&self) -> &ToolName {
            static NAME: LazyLock<ToolName> = LazyLock::new(|| ToolName::from_static("echo"));
            &NAME
        }
        fn description(&self) -> &str {
            "Echo the text back"
        }
        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
        }
        fn call(&self, arguments: Value) -> BoxFuture<'_, Result<String, ToolError>> {
            self.calls.lock().unwrap().push(arguments.clone());
            Box::pin(async move {
                let text = arguments["text"].as_str().ok_or_else(|| {
                    ToolError::InvalidArguments("missing required parameter `text`".into())
                })?;
                Ok(format!("echo: {text}"))
            })
        }
    }

    /// Takes a long time.
    struct Sleepy;

    impl Tool for Sleepy {
        fn name(&self) -> &ToolName {
            static NAME: LazyLock<ToolName> = LazyLock::new(|| ToolName::from_static("sleepy"));
            &NAME
        }
        fn description(&self) -> &str {
            "Sleeps"
        }
        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        fn call(&self, _: Value) -> BoxFuture<'_, Result<String, ToolError>> {
            Box::pin(async {
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok("woke up".into())
            })
        }
    }

    /// Produces whatever it was built with.
    struct Fixed(Result<String, ToolError>);

    impl Tool for Fixed {
        fn name(&self) -> &ToolName {
            static NAME: LazyLock<ToolName> = LazyLock::new(|| ToolName::from_static("fixed"));
            &NAME
        }
        fn description(&self) -> &str {
            "Returns a fixed result"
        }
        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        fn call(&self, _: Value) -> BoxFuture<'_, Result<String, ToolError>> {
            let result = match &self.0 {
                Ok(text) => Ok(text.clone()),
                Err(ToolError::Failed(m)) => Err(ToolError::Failed(m.clone())),
                Err(ToolError::InvalidArguments(m)) => Err(ToolError::InvalidArguments(m.clone())),
            };
            Box::pin(async move { result })
        }
    }

    type Calls = Arc<Mutex<Vec<Value>>>;

    fn registry(extra: Vec<Arc<dyn Tool>>) -> (ToolRegistry, Calls) {
        let calls = Calls::default();
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(Echo {
                calls: calls.clone(),
            }))
            .unwrap();
        registry.register(Arc::new(Calculator)).unwrap();
        for tool in extra {
            registry.register(tool).unwrap();
        }
        (registry, calls)
    }

    fn agent(replies: Vec<ScriptedReply>) -> (Agent<ScriptedBackend>, Calls) {
        let (tools, calls) = registry(vec![]);
        (
            Agent::new(ScriptedBackend::new(replies), tools, AgentConfig::default()),
            calls,
        )
    }

    async fn turn(
        agent: &mut Agent<ScriptedBackend>,
        message: &str,
    ) -> (Result<String, TurnError>, Vec<AgentEvent>) {
        let mut events = Vec::new();
        let result = agent
            .run_turn(message, |e| {
                // Step markers are checked by their own test.
                if !matches!(e, AgentEvent::StepStarted { .. }) {
                    events.push(e);
                }
            })
            .await;
        (result, events)
    }

    fn roles(messages: &[ChatMessage]) -> Vec<Role> {
        messages.iter().map(|m| m.role).collect()
    }

    fn tool_messages(messages: &[ChatMessage]) -> Vec<&ChatMessage> {
        messages.iter().filter(|m| m.role == Role::Tool).collect()
    }

    #[tokio::test]
    async fn a_direct_answer_takes_one_model_call_and_no_tools() {
        let (mut agent, calls) = agent(vec![ScriptedReply::text("Rust is a language.")]);
        let (result, events) = turn(&mut agent, "What is Rust?").await;

        assert_eq!(result.unwrap(), "Rust is a language.");
        assert_eq!(agent.backend().requests().len(), 1);
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(
            events,
            vec![
                AgentEvent::Text("Rust is a language.".into()),
                AgentEvent::TurnFinished
            ]
        );
        assert_eq!(
            roles(agent.history()),
            vec![Role::System, Role::User, Role::Assistant]
        );
    }

    #[tokio::test]
    async fn one_tool_call_then_an_answer() {
        let (mut agent, calls) = agent(vec![
            ScriptedReply::call("c1", "echo", r#"{"text":"hi"}"#),
            ScriptedReply::text("The tool said hi."),
        ]);
        let (result, _) = turn(&mut agent, "say hi").await;

        assert_eq!(result.unwrap(), "The tool said hi.");
        assert_eq!(calls.lock().unwrap().as_slice(), &[json!({"text": "hi"})]);

        let requests = agent.backend().requests();
        assert_eq!(requests.len(), 2);
        let second = &requests[1].messages;
        assert_eq!(
            roles(second),
            vec![Role::System, Role::User, Role::Assistant, Role::Tool]
        );
        assert_eq!(second[2].tool_calls.as_ref().unwrap()[0].id.as_str(), "c1");
        assert_eq!(
            second[3].tool_call_id.as_ref().map(|id| id.as_str()),
            Some("c1")
        );
        assert_eq!(second[3].content.as_deref(), Some("echo: hi"));
        assert_eq!(
            roles(agent.history()),
            vec![
                Role::System,
                Role::User,
                Role::Assistant,
                Role::Tool,
                Role::Assistant
            ]
        );
    }

    #[tokio::test]
    async fn several_calls_in_one_reply_run_in_order_and_all_results_come_before_the_next_call() {
        let (mut agent, calls) = agent(vec![
            ScriptedReply::tool_calls(
                Some("Doing both."),
                vec![
                    ToolCall::from_static("a", "echo", r#"{"text":"first"}"#),
                    ToolCall::from_static("b", "calculator", r#"{"expression":"6*7"}"#),
                    ToolCall::from_static("c", "echo", r#"{"text":"third"}"#),
                ],
            ),
            ScriptedReply::text("done"),
        ]);
        let (result, events) = turn(&mut agent, "go").await;
        result.unwrap();

        let order: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolStarted { id, .. } => Some(format!("start {id}")),
                AgentEvent::ToolFinished { id, .. } => Some(format!("finish {id}")),
                _ => None,
            })
            .collect();
        assert_eq!(
            order,
            vec![
                "start a", "finish a", "start b", "finish b", "start c", "finish c"
            ]
        );
        assert_eq!(calls.lock().unwrap().len(), 2);

        let second = &agent.backend().requests()[1].messages;
        let results = tool_messages(second);
        assert_eq!(results.len(), 3);
        assert_eq!(
            results
                .iter()
                .map(|m| m.tool_call_id.as_ref().unwrap().as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert_eq!(results[1].content.as_deref(), Some("42"));
        assert_eq!(
            second[2].content.as_deref(),
            Some("Doing both."),
            "the preamble stays with the calls"
        );
    }

    #[tokio::test]
    async fn events_for_a_tool_use_are_in_order_and_describe_the_call() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::call("c1", "echo", r#"{"text":"x"}"#),
            ScriptedReply::text("ok"),
        ]);
        let (_, events) = turn(&mut agent, "go").await;

        let AgentEvent::ToolStarted {
            id,
            name,
            arguments,
        } = &events[0]
        else {
            panic!("expected ToolStarted first, got {events:?}")
        };
        assert_eq!(
            (id.as_str(), name.as_str(), arguments.as_str()),
            ("c1", "echo", r#"{"text":"x"}"#)
        );
        let AgentEvent::ToolFinished {
            id,
            name,
            ok,
            output,
            ..
        } = &events[1]
        else {
            panic!("expected ToolFinished second, got {events:?}")
        };
        assert_eq!(
            (id.as_str(), name.as_str(), *ok, output.as_str()),
            ("c1", "echo", true, "echo: x")
        );
        assert_eq!(events.last(), Some(&AgentEvent::TurnFinished));
    }

    #[tokio::test]
    async fn every_model_call_is_announced_with_its_number() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::call("c1", "echo", r#"{"text":"x"}"#),
            ScriptedReply::text("ok"),
        ]);
        let mut steps = Vec::new();
        agent
            .run_turn("go", |e| {
                if let AgentEvent::StepStarted { step } = e {
                    steps.push(step);
                }
            })
            .await
            .unwrap();
        assert_eq!(steps, vec![1, 2]);
    }

    #[tokio::test]
    async fn the_step_limit_stops_a_model_that_never_stops_calling_tools() {
        let always_tool = || ScriptedReply::call("c", "echo", r#"{"text":"again"}"#);
        let (tools, _) = registry(vec![]);
        let config = AgentConfig {
            max_steps: 2,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(
            ScriptedBackend::new([always_tool(), always_tool(), always_tool()]),
            tools,
            config,
        );

        let (result, events) = turn(&mut agent, "loop").await;

        assert_eq!(result, Err(TurnError::StepLimit(2)));
        assert_eq!(agent.backend().requests().len(), 2, "no third model call");
        assert_eq!(agent.backend().remaining(), 1);
        assert!(events.contains(&AgentEvent::StepLimit { steps: 2 }));
        assert!(!events.contains(&AgentEvent::TurnFinished));
        assert!(TurnError::StepLimit(2).to_string().contains("step limit"));
    }

    #[tokio::test]
    async fn after_the_step_limit_the_session_still_works() {
        let (tools, _) = registry(vec![]);
        let config = AgentConfig {
            max_steps: 1,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(
            ScriptedBackend::new([
                ScriptedReply::call("c", "echo", r#"{"text":"x"}"#),
                ScriptedReply::text("fine now"),
            ]),
            tools,
            config,
        );
        assert_eq!(
            turn(&mut agent, "first").await.0,
            Err(TurnError::StepLimit(1))
        );
        assert_eq!(turn(&mut agent, "second").await.0.unwrap(), "fine now");

        // What happened in the limited turn is kept, so the model sees it.
        let last = &agent.backend().requests()[1].messages;
        assert_eq!(
            roles(last),
            vec![
                Role::System,
                Role::User,
                Role::Assistant,
                Role::Tool,
                Role::User
            ]
        );
    }

    #[tokio::test]
    async fn a_limit_of_zero_still_allows_one_call() {
        let (tools, _) = registry(vec![]);
        let config = AgentConfig {
            max_steps: 0,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(
            ScriptedBackend::new([ScriptedReply::text("ok")]),
            tools,
            config,
        );
        assert_eq!(turn(&mut agent, "hi").await.0.unwrap(), "ok");
    }

    #[tokio::test]
    async fn an_unknown_tool_is_reported_to_the_model_with_the_available_names() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::call("c1", "delete_everything", "{}"),
            ScriptedReply::text("sorry"),
        ]);
        let (result, events) = turn(&mut agent, "go").await;
        assert_eq!(result.unwrap(), "sorry");

        let shown = tool_messages(&agent.backend().requests()[1].messages)[0]
            .content
            .clone()
            .unwrap();
        assert!(
            shown.contains("unknown tool `delete_everything`"),
            "{shown}"
        );
        assert!(
            shown.contains("echo") && shown.contains("calculator"),
            "{shown}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolFinished { ok: false, .. }))
        );
    }

    #[tokio::test]
    async fn a_missing_parameter_is_reported_to_the_model() {
        // Arguments that are not a JSON object cannot reach the agent any more: the value type
        // rejects them, and the backend reports them as a malformed response.
        let (mut agent, calls) = agent(vec![
            ScriptedReply::call("c", "calculator", "{}"),
            ScriptedReply::text("giving up"),
        ]);
        turn(&mut agent, "go").await.0.unwrap();

        let messages = &agent.backend().requests()[1].messages;
        let shown: Vec<_> = tool_messages(messages)
            .iter()
            .map(|m| m.content.clone().unwrap())
            .collect();
        assert!(
            shown[0].contains("missing required parameter `expression`"),
            "{}",
            shown[0]
        );
        assert!(calls.lock().unwrap().is_empty(), "echo never ran");
    }

    #[tokio::test]
    async fn empty_arguments_mean_an_empty_object() {
        let (mut agent, calls) = agent(vec![
            ScriptedReply::call("a", "echo", ""),
            ScriptedReply::text("ok"),
        ]);
        turn(&mut agent, "go").await.0.unwrap();
        assert_eq!(calls.lock().unwrap().as_slice(), &[json!({})]);
    }

    #[tokio::test]
    async fn a_failing_tool_does_not_abort_the_turn() {
        let (tools, _) = registry(vec![Arc::new(Fixed(Err(ToolError::Failed(
            "disk on fire".into(),
        ))))]);
        let mut agent = Agent::new(
            ScriptedBackend::new([
                ScriptedReply::call("a", "fixed", "{}"),
                ScriptedReply::text("it failed"),
            ]),
            tools,
            AgentConfig::default(),
        );
        assert_eq!(turn(&mut agent, "go").await.0.unwrap(), "it failed");
        let shown = tool_messages(&agent.backend().requests()[1].messages)[0]
            .content
            .clone()
            .unwrap();
        assert_eq!(shown, "Error: disk on fire");
    }

    #[tokio::test]
    async fn a_slow_tool_is_cancelled_and_reported() {
        let (tools, _) = registry(vec![Arc::new(Sleepy)]);
        let config = AgentConfig {
            tool_timeout: Duration::from_millis(50),
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(
            ScriptedBackend::new([
                ScriptedReply::call("a", "sleepy", "{}"),
                ScriptedReply::text("too slow"),
            ]),
            tools,
            config,
        );
        let started = Instant::now();
        let (result, events) = turn(&mut agent, "go").await;

        assert_eq!(result.unwrap(), "too slow");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the tool was not cancelled"
        );
        let shown = tool_messages(&agent.backend().requests()[1].messages)[0]
            .content
            .clone()
            .unwrap();
        assert!(shown.contains("timed out"), "{shown}");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolFinished { ok: false, .. }))
        );
    }

    #[tokio::test]
    async fn oversized_tool_output_is_truncated_with_a_marker() {
        let (tools, _) = registry(vec![Arc::new(Fixed(Ok("x".repeat(100 * 1024))))]);
        let mut agent = Agent::new(
            ScriptedBackend::new([
                ScriptedReply::call("a", "fixed", "{}"),
                ScriptedReply::text("ok"),
            ]),
            tools,
            AgentConfig::default(),
        );
        turn(&mut agent, "go").await.0.unwrap();

        let shown = tool_messages(&agent.backend().requests()[1].messages)[0]
            .content
            .clone()
            .unwrap();
        let limit = AgentConfig::default().max_tool_output;
        let (body, note) = shown.split_once("\n[output truncated").unwrap();
        assert_eq!(body.len(), limit);
        assert!(
            note.contains(&format!("{} bytes omitted", 100 * 1024 - limit)),
            "{note}"
        );
    }

    #[test]
    fn truncation_respects_character_boundaries() {
        let text = "\u{2603}".repeat(10); // 30 bytes
        let cut = truncate_output(text, 10);
        let (body, _) = cut.split_once("\n[output truncated").unwrap();
        assert_eq!(body.chars().count(), 3, "9 bytes = 3 whole characters");
        assert_eq!(truncate_output("short".into(), 100), "short");
        assert_eq!(truncate_output("exact".into(), 5), "exact");
    }

    #[tokio::test]
    async fn the_conversation_carries_over_between_turns_until_reset() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::text("Paris"),
            ScriptedReply::text("About 2 million"),
            ScriptedReply::text("fresh start"),
        ]);
        turn(&mut agent, "Capital of France?").await.0.unwrap();
        turn(&mut agent, "And its population?").await.0.unwrap();

        let second = &agent.backend().requests()[1].messages;
        assert_eq!(
            roles(second),
            vec![Role::System, Role::User, Role::Assistant, Role::User]
        );
        assert_eq!(second[1].content.as_deref(), Some("Capital of France?"));

        agent.reset();
        assert_eq!(roles(agent.history()), vec![Role::System]);
        turn(&mut agent, "new topic").await.0.unwrap();
        let third = &agent.backend().requests()[2].messages;
        assert_eq!(roles(third), vec![Role::System, Role::User]);
    }

    #[tokio::test]
    async fn the_system_prompt_comes_first() {
        let (tools, _) = registry(vec![]);
        let config = AgentConfig {
            system_prompt: SystemPrompt::from_static("Be brief."),
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(
            ScriptedBackend::new([ScriptedReply::text("ok")]),
            tools,
            config,
        );
        turn(&mut agent, "hi").await.0.unwrap();
        let first = &agent.backend().requests()[0].messages[0];
        assert_eq!(
            (first.role, first.content.as_deref()),
            (Role::System, Some("Be brief."))
        );
    }

    #[tokio::test]
    async fn a_backend_error_mid_turn_discards_the_turn() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::text("first answer"),
            ScriptedReply::call("c", "echo", r#"{"text":"x"}"#),
            ScriptedReply::error(BackendError::Status {
                status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                message: "boom".into(),
            }),
            ScriptedReply::text("recovered"),
        ]);
        turn(&mut agent, "one").await.0.unwrap();
        let before = agent.history().to_vec();

        let (result, _) = turn(&mut agent, "two").await;
        assert_eq!(
            result,
            Err(TurnError::Backend(BackendError::Status {
                status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                message: "boom".into()
            }))
        );
        assert_eq!(
            agent.history(),
            before.as_slice(),
            "nothing from the failed turn is kept"
        );

        assert_eq!(turn(&mut agent, "three").await.0.unwrap(), "recovered");
        let last = &agent.backend().requests()[3].messages;
        assert_eq!(
            roles(last),
            vec![Role::System, Role::User, Role::Assistant, Role::User]
        );
    }

    #[tokio::test]
    async fn partial_text_from_a_failed_request_is_not_kept() {
        let (mut agent, _) = agent(vec![ScriptedReply::error_after(
            "half an ans",
            BackendError::StreamEndedEarly,
        )]);
        let (result, events) = turn(&mut agent, "q").await;
        assert_eq!(
            result,
            Err(TurnError::Backend(BackendError::StreamEndedEarly))
        );
        assert!(
            events.contains(&AgentEvent::Text("half an ans".into())),
            "it was shown as it streamed"
        );
        assert_eq!(roles(agent.history()), vec![Role::System]);
    }

    #[tokio::test]
    async fn cancelling_a_turn_leaves_the_history_untouched() {
        let (mut agent, _) = agent(vec![ScriptedReply::text("one"), ScriptedReply::Hang]);
        turn(&mut agent, "first").await.0.unwrap();
        let before = agent.history().to_vec();

        let cancelled =
            tokio::time::timeout(Duration::from_millis(50), agent.run_turn("second", |_| {})).await;
        assert!(
            cancelled.is_err(),
            "the hanging model should have timed the turn out"
        );
        assert_eq!(agent.history(), before.as_slice());
    }

    #[tokio::test]
    async fn every_request_advertises_exactly_the_registered_tools() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::call("c", "echo", r#"{"text":"x"}"#),
            ScriptedReply::text("ok"),
        ]);
        turn(&mut agent, "go").await.0.unwrap();
        for request in agent.backend().requests() {
            let names: Vec<_> = request
                .tools
                .unwrap()
                .iter()
                .map(|t| t.function.name.to_string())
                .collect();
            assert_eq!(names, vec!["echo", "calculator"]);
        }
    }

    #[tokio::test]
    async fn without_tools_no_tools_are_sent() {
        let mut agent = Agent::new(
            ScriptedBackend::new([ScriptedReply::text("ok")]),
            ToolRegistry::new(),
            AgentConfig::default(),
        );
        turn(&mut agent, "hi").await.0.unwrap();
        assert!(agent.backend().requests()[0].tools.is_none());
    }

    #[tokio::test]
    async fn generation_settings_are_passed_to_the_backend() {
        let (tools, _) = registry(vec![]);
        let config = AgentConfig {
            temperature: Some(Temperature::from_const(0.3)),
            max_tokens: Some(MaxTokens::from_const(77)),
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(
            ScriptedBackend::new([ScriptedReply::text("ok")]),
            tools,
            config,
        );
        turn(&mut agent, "hi").await.0.unwrap();
        let request = &agent.backend().requests()[0];
        assert_eq!(
            (request.temperature, request.max_tokens),
            (
                Some(Temperature::from_const(0.3)),
                Some(MaxTokens::from_const(77))
            )
        );
    }

    #[tokio::test]
    async fn an_empty_reply_is_an_empty_answer() {
        let (mut agent, _) = agent(vec![ScriptedReply::Reply {
            pieces: vec![],
            result: crate::backend::ChatResult {
                content: None,
                tool_calls: vec![],
                finish: djinn_core::chat::FinishReason::Stop,
            },
            delay: None,
        }]);
        assert_eq!(turn(&mut agent, "hi").await.0.unwrap(), "");
    }

    /// Answers from a script and records what it was asked.
    struct Scripted {
        decisions: Mutex<std::collections::VecDeque<Decision>>,
        asked: Mutex<Vec<(String, bool)>>,
    }

    impl Scripted {
        fn new(decisions: Vec<Decision>) -> Self {
            Self {
                decisions: Mutex::new(decisions.into()),
                asked: Mutex::default(),
            }
        }
    }

    impl Approver for Scripted {
        fn decide<'a>(&'a self, request: ApprovalRequest<'a>) -> BoxFuture<'a, Decision> {
            self.asked
                .lock()
                .unwrap()
                .push((request.id.to_string(), request.needs_confirmation));
            let decision = self
                .decisions
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Decision::Allow);
            Box::pin(async move { decision })
        }
    }

    /// Never answers.
    struct Never;

    impl Approver for Never {
        fn decide<'a>(&'a self, _: ApprovalRequest<'a>) -> BoxFuture<'a, Decision> {
            Box::pin(futures::future::pending())
        }
    }

    async fn turn_with(
        agent: &mut Agent<ScriptedBackend>,
        approver: &dyn Approver,
    ) -> (Result<String, TurnError>, Vec<AgentEvent>) {
        let mut events = Vec::new();
        let result = agent
            .run_turn_with(
                "go",
                |e| {
                    if !matches!(e, AgentEvent::StepStarted { .. }) {
                        events.push(e);
                    }
                },
                approver,
            )
            .await;
        (result, events)
    }

    #[tokio::test]
    async fn a_denied_call_does_not_run_and_the_model_is_told_why() {
        let (mut agent, calls) = agent(vec![
            ScriptedReply::call("c1", "echo", r#"{"text":"x"}"#),
            ScriptedReply::text("understood"),
        ]);
        let approver = Scripted::new(vec![Decision::Deny {
            reason: Some("private".into()),
        }]);
        let (result, events) = turn_with(&mut agent, &approver).await;

        assert_eq!(result.unwrap(), "understood");
        assert!(calls.lock().unwrap().is_empty(), "the tool must not run");
        let second = &agent.backend().requests()[1].messages;
        assert_eq!(
            second[3].content.as_deref(),
            Some("Error: the user denied this tool call because: private")
        );
        assert_eq!(
            events[0],
            AgentEvent::ToolDenied {
                id: ToolCallId::from_static("c1"),
                name: ToolName::from_static("echo"),
                reason: Some("private".into()),
            }
        );
        assert!(
            !events.iter().any(|e| matches!(
                e,
                AgentEvent::ToolStarted { .. } | AgentEvent::ToolFinished { .. }
            )),
            "a denied call reports only the denial: {events:?}"
        );
    }

    #[tokio::test]
    async fn a_denial_without_a_reason_says_so_plainly() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::call("c1", "echo", r#"{"text":"x"}"#),
            ScriptedReply::text("ok"),
        ]);
        let approver = Scripted::new(vec![Decision::Deny {
            reason: Some("  ".into()),
        }]);
        turn_with(&mut agent, &approver).await.0.unwrap();
        let second = &agent.backend().requests()[1].messages;
        assert_eq!(
            second[3].content.as_deref(),
            Some("Error: the user denied this tool call")
        );
    }

    #[tokio::test]
    async fn the_decision_comes_before_the_call_is_reported_started() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::call("c1", "echo", r#"{"text":"x"}"#),
            ScriptedReply::text("ok"),
        ]);
        let approver = Scripted::new(vec![Decision::Allow]);
        let (_, events) = turn_with(&mut agent, &approver).await;
        assert_eq!(
            approver.asked.lock().unwrap().as_slice(),
            &[("c1".to_owned(), false)]
        );
        assert!(matches!(events[0], AgentEvent::ToolStarted { .. }));
        assert!(matches!(
            events[1],
            AgentEvent::ToolFinished { ok: true, .. }
        ));
    }

    #[tokio::test]
    async fn the_next_call_is_still_presented_after_a_denial_and_results_stay_in_order() {
        let (mut agent, calls) = agent(vec![
            ScriptedReply::tool_calls(
                None,
                vec![
                    ToolCall::from_static("a", "echo", r#"{"text":"first"}"#),
                    ToolCall::from_static("b", "echo", r#"{"text":"second"}"#),
                ],
            ),
            ScriptedReply::text("done"),
        ]);
        let approver = Scripted::new(vec![Decision::Deny { reason: None }, Decision::Allow]);
        turn_with(&mut agent, &approver).await.0.unwrap();

        assert_eq!(
            calls.lock().unwrap().as_slice(),
            &[json!({"text": "second"})]
        );
        assert_eq!(approver.asked.lock().unwrap().len(), 2);
        let results = agent.backend().requests()[1]
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.tool_call_id.as_ref().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(results, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn an_unknown_tool_does_not_ask() {
        let (mut agent, _) = agent(vec![
            ScriptedReply::call("c1", "missing", "{}"),
            ScriptedReply::text("ok"),
        ]);
        let approver = Scripted::new(vec![Decision::Deny { reason: None }]);
        let (_, events) = turn_with(&mut agent, &approver).await;
        assert!(approver.asked.lock().unwrap().is_empty());
        assert!(matches!(
            events[1],
            AgentEvent::ToolFinished { ok: false, .. }
        ));
    }

    #[tokio::test]
    async fn a_tool_that_needs_confirmation_says_so_in_the_request() {
        struct Careful;
        impl Tool for Careful {
            fn name(&self) -> &ToolName {
                static NAME: LazyLock<ToolName> =
                    LazyLock::new(|| ToolName::from_static("careful"));
                &NAME
            }
            fn description(&self) -> &str {
                "needs confirmation"
            }
            fn parameters(&self) -> Value {
                json!({"type": "object", "properties": {}})
            }
            fn needs_confirmation(&self) -> bool {
                true
            }
            fn call(&self, _: Value) -> BoxFuture<'_, Result<String, ToolError>> {
                Box::pin(async { Ok("done".into()) })
            }
        }
        let (tools, _) = registry(vec![Arc::new(Careful)]);
        let mut agent = Agent::new(
            ScriptedBackend::new([
                ScriptedReply::call("c1", "careful", "{}"),
                ScriptedReply::text("ok"),
            ]),
            tools,
            AgentConfig::default(),
        );
        let approver = Scripted::new(vec![]);
        turn_with(&mut agent, &approver).await.0.unwrap();
        assert_eq!(
            approver.asked.lock().unwrap().as_slice(),
            &[("c1".to_owned(), true)]
        );
    }

    #[tokio::test]
    async fn dropping_the_turn_while_a_decision_is_pending_leaves_the_history_alone() {
        let (mut agent, calls) = agent(vec![ScriptedReply::call("c1", "echo", r#"{"text":"x"}"#)]);
        let before = agent.history().len();
        let waited = tokio::time::timeout(
            Duration::from_millis(50),
            agent.run_turn_with("go", |_| {}, &Never),
        )
        .await;
        assert!(waited.is_err(), "the turn must wait for the user");
        assert_eq!(agent.history().len(), before);
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn allow_all_is_the_default_behaviour() {
        let (mut agent, calls) = agent(vec![
            ScriptedReply::call("c1", "echo", r#"{"text":"x"}"#),
            ScriptedReply::text("ok"),
        ]);
        let (_, events) = turn_with(&mut agent, &AllowAll).await;
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert!(matches!(events[0], AgentEvent::ToolStarted { .. }));
    }
}
