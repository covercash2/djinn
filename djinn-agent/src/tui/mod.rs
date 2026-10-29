//! The full-screen terminal front end: one `select!` loop over terminal events, the running turn and a
//! spinner tick (design D2). Everything but [`terminal`] is testable without a terminal.
//!
//! The loop has two phases. While *idle* it only reacts to the user; when a message is sent it
//! creates the turn future (which borrows the agent) and runs a second loop around it. Cancelling is
//! dropping that future, exactly like the plain REPL, so the history is left as it was.

pub mod app;
pub mod approval;
pub mod completion;
pub mod input;
pub mod markdown;
pub mod terminal;
pub mod text;
pub mod view;

use std::io;
use std::time::Duration;

use crossterm::event::Event;
use futures::{Stream, StreamExt};
use ratatui::Terminal;
use tokio::sync::mpsc;

use self::app::{App, Effect, Level, Msg, Settings, ToolInfo};
use self::approval::{SessionAllow, TuiApprover};
use crate::agent::Agent;
use crate::backend::Backend as AgentBackend;
use crate::settings::{Approval, BaseUrl};
use djinn_core::values::ModelId;

/// How often the spinner moves while a turn runs.
const TICK: Duration = Duration::from_millis(100);

impl Settings {
    /// What the status bar and `/tools` need to know about `agent`.
    pub fn for_agent<B: AgentBackend>(
        agent: &Agent<B>,
        model: ModelId,
        base_url: BaseUrl,
        approval: Approval,
    ) -> Self {
        Self {
            model,
            base_url,
            approval,
            max_steps: agent.config().max_steps,
            tools: agent
                .tools()
                .iter()
                .map(|tool| ToolInfo {
                    name: tool.name().clone(),
                    description: tool.description().to_owned(),
                })
                .collect(),
        }
    }
}

/// Runs the TUI on the real terminal until the user quits.
pub async fn run_terminal<B: AgentBackend>(
    agent: &mut Agent<B>,
    settings: Settings,
    startup: Vec<(Level, String)>,
) -> io::Result<()> {
    let (guard, mut terminal) = terminal::TerminalGuard::enter()?;
    let events = crossterm::event::EventStream::new();
    let result = run(agent, settings, events, &mut terminal, |_| {}, startup).await;
    // Put the terminal back before anything is printed about an error.
    drop(guard);
    result.map(|_| ())
}

fn to_msg(event: Event) -> Option<Msg> {
    match event {
        Event::Key(key) => Some(Msg::Key(key)),
        Event::Paste(text) => Some(Msg::Paste(text)),
        Event::Mouse(mouse) => Some(Msg::Mouse(mouse)),
        // A resize needs no state change: the next loop iteration draws at the new size.
        _ => None,
    }
}

/// What woke the loop up.
enum Input {
    Msg(Msg),
    /// Something happened that only needs a redraw.
    Redraw,
    /// The terminal's event stream ended.
    Closed,
}

async fn next_input<S>(
    events: &mut S,
    messages: &mut mpsc::UnboundedReceiver<Msg>,
) -> io::Result<Input>
where
    S: Stream<Item = io::Result<Event>> + Unpin,
{
    tokio::select! {
        event = events.next() => match event {
            Some(Ok(event)) => Ok(to_msg(event).map_or(Input::Redraw, Input::Msg)),
            Some(Err(error)) => Err(error),
            None => Ok(Input::Closed),
        },
        Some(message) = messages.recv() => Ok(Input::Msg(message)),
    }
}

/// Runs the TUI over any event source and ratatui backend. `observe` sees the state after each
/// draw (tests use it to wait for things). Returns the final state.
pub async fn run<B, S, T>(
    agent: &mut Agent<B>,
    settings: Settings,
    mut events: S,
    terminal: &mut Terminal<T>,
    mut observe: impl FnMut(&App),
    startup: Vec<(Level, String)>,
) -> io::Result<App>
where
    B: AgentBackend,
    S: Stream<Item = io::Result<Event>> + Unpin,
    T: ratatui::backend::Backend,
    T::Error: std::fmt::Display,
{
    let (messages_tx, mut messages) = mpsc::unbounded_channel::<Msg>();
    let allowed = SessionAllow::default();
    let approver = TuiApprover::new(settings.approval, allowed.clone(), messages_tx.clone());
    let mut app = App::new(settings, allowed);
    for (level, text) in startup {
        app.update(Msg::Notice(level, text));
    }

    macro_rules! draw {
        () => {
            terminal
                .draw(|frame| view::render(&mut app, frame))
                .map_err(|error| io::Error::other(error.to_string()))?;
            observe(&app);
        };
    }

    loop {
        // ---- idle: wait for the user to send something ----
        let message = loop {
            draw!();
            let msg = match next_input(&mut events, &mut messages).await? {
                Input::Msg(msg) => msg,
                Input::Redraw => continue,
                Input::Closed => return Ok(app),
            };
            let mut send = None;
            for effect in app.update(msg) {
                match effect {
                    Effect::Send(text) => send = Some(text),
                    Effect::Reset => agent.reset(),
                    Effect::Quit => return Ok(app),
                    Effect::Cancel => {}
                }
            }
            if let Some(text) = send {
                break text;
            }
        };

        // ---- working: run the turn while still serving the UI ----
        let events_tx = messages_tx.clone();
        let turn = agent.run_turn_with(
            &message,
            move |event| {
                let _ = events_tx.send(Msg::Agent(event));
            },
            &approver,
        );
        tokio::pin!(turn);
        let mut ticker = tokio::time::interval(TICK);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let finished = loop {
            draw!();
            tokio::select! {
                result = &mut turn => break Some(result),
                _ = ticker.tick() => {
                    app.update(Msg::Tick);
                }
                input = next_input(&mut events, &mut messages) => {
                    let msg = match input? {
                        Input::Msg(msg) => msg,
                        Input::Redraw => continue,
                        Input::Closed => return Ok(app),
                    };
                    let mut cancelled = false;
                    for effect in app.update(msg) {
                        match effect {
                            Effect::Cancel => cancelled = true,
                            Effect::Quit => return Ok(app),
                            // Sending and resetting are refused while a turn runs.
                            Effect::Send(_) | Effect::Reset => {}
                        }
                    }
                    if cancelled {
                        break None;
                    }
                }
            }
        };

        match finished {
            Some(result) => {
                // The turn's events were queued before it ended; show them first.
                while let Ok(msg) = messages.try_recv() {
                    app.update(msg);
                }
                app.update(Msg::TurnEnded(result));
            }
            None => {
                // The turn was dropped by the cancel; whatever it had queued is stale.
                while messages.try_recv().is_ok() {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use djinn_core::chat::Role;
    use futures::channel::mpsc::{UnboundedSender, unbounded};
    use ratatui::backend::TestBackend;
    use tokio::sync::watch;

    use super::*;
    use crate::agent::AgentConfig;
    use crate::backend::{BackendError, ScriptedBackend, ScriptedReply};
    use crate::tools::ToolRegistry;

    /// What a test can wait for.
    #[derive(Clone, Debug, Default)]
    struct Snapshot {
        working: bool,
        dialog: bool,
        /// How many times a dialog has opened.
        dialogs: usize,
        /// The transcript as plain text, one entry per line.
        transcript: String,
    }

    fn transcript(app: &App) -> String {
        use crate::tui::app::EntryKind;
        app.entries()
            .iter()
            .map(|entry| match entry.kind() {
                EntryKind::User(text) => format!("user: {text}"),
                EntryKind::Assistant { text, .. } => format!("assistant: {text}"),
                EntryKind::Tool(block) => format!("tool {} {:?}", block.name, block.state),
                EntryKind::Notice { text, .. } => format!("notice: {text}"),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    struct Driver {
        keys: UnboundedSender<io::Result<Event>>,
        state: watch::Receiver<Snapshot>,
    }

    impl Driver {
        fn press(&self, code: KeyCode, modifiers: KeyModifiers) {
            self.keys
                .unbounded_send(Ok(Event::Key(KeyEvent::new(code, modifiers))))
                .unwrap();
        }

        fn type_text(&self, text: &str) {
            for c in text.chars() {
                self.press(KeyCode::Char(c), KeyModifiers::NONE);
            }
        }

        fn send(&self, text: &str) {
            self.type_text(text);
            self.press(KeyCode::Enter, KeyModifiers::NONE);
        }

        fn quit(&self) {
            self.press(KeyCode::Char('d'), KeyModifiers::CONTROL);
        }

        async fn wait(&mut self, what: &str, condition: impl Fn(&Snapshot) -> bool) {
            let reached = matches!(
                tokio::time::timeout(
                    Duration::from_secs(10),
                    self.state.wait_for(|snapshot| condition(snapshot)),
                )
                .await,
                Ok(Ok(_))
            );
            if !reached {
                panic!(
                    "timed out waiting for {what}; last state:\n{:#?}",
                    *self.state.borrow()
                );
            }
        }
    }

    struct Outcome {
        app: App,
        screen: String,
    }

    fn settings(agent: &Agent<ScriptedBackend>, approval: Approval) -> Settings {
        Settings::for_agent(
            agent,
            ModelId::from_static("test-model"),
            BaseUrl::from_static("http://localhost:8080/v1"),
            approval,
        )
    }

    fn agent_with(
        replies: Vec<ScriptedReply>,
        config: AgentConfig,
    ) -> (Agent<ScriptedBackend>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let tools = ToolRegistry::builtin(dir.path()).unwrap();
        (
            Agent::new(ScriptedBackend::new(replies), tools, config),
            dir,
        )
    }

    fn new_agent(replies: Vec<ScriptedReply>) -> (Agent<ScriptedBackend>, tempfile::TempDir) {
        agent_with(replies, AgentConfig::default())
    }

    /// Runs the TUI on a 80x24 test terminal while `script` plays the user.
    async fn session<F, Fut>(
        agent: &mut Agent<ScriptedBackend>,
        approval: Approval,
        startup: Vec<(Level, String)>,
        script: F,
    ) -> Outcome
    where
        F: FnOnce(Driver) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let (keys, events) = unbounded();
        let (publish, state) = watch::channel(Snapshot::default());
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        let settings = settings(agent, approval);
        let opened = Arc::new(Mutex::new((0usize, false)));
        let observe = move |app: &App| {
            let mut opened = opened.lock().unwrap();
            let dialog = app.dialog.is_some();
            if dialog && !opened.1 {
                opened.0 += 1;
            }
            opened.1 = dialog;
            publish.send_replace(Snapshot {
                working: app.is_working(),
                dialog,
                dialogs: opened.0,
                transcript: transcript(app),
            });
        };
        let run = run(agent, settings, events, &mut terminal, observe, startup);
        let driver = script(Driver { keys, state });
        let (result, ()) =
            tokio::time::timeout(Duration::from_secs(20), async { tokio::join!(run, driver) })
                .await
                .expect("the session hung");
        let app = result.unwrap();
        let buffer = terminal.backend().buffer().clone();
        let mut screen = String::new();
        for y in 0..24 {
            for x in 0..80 {
                screen.push_str(buffer[(x, y)].symbol());
            }
            screen.push('\n');
        }
        Outcome { app, screen }
    }

    fn roles(agent: &Agent<ScriptedBackend>) -> Vec<Role> {
        agent.history().iter().map(|m| m.role).collect()
    }

    #[tokio::test]
    async fn a_scripted_session_shows_the_message_the_tool_and_the_answer() {
        let (mut agent, _dir) = new_agent(vec![
            ScriptedReply::call("c1", "calculator", r#"{"expression":"2+2"}"#),
            ScriptedReply::text("It is 4."),
        ]);
        let outcome = session(
            &mut agent,
            Approval::Auto,
            vec![],
            |mut driver| async move {
                driver.send("2+2");
                driver
                    .wait("the answer", |s| {
                        !s.working && s.transcript.contains("It is 4.")
                    })
                    .await;
                driver.quit();
            },
        )
        .await;

        assert!(outcome.screen.contains("› 2+2"), "{}", outcome.screen);
        assert!(outcome.screen.contains("calculator"), "{}", outcome.screen);
        assert!(outcome.screen.contains("It is 4."), "{}", outcome.screen);
        assert_eq!(
            roles(&agent),
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
    async fn ask_mode_waits_for_the_user_before_the_tool_runs() {
        let (mut agent, _dir) = new_agent(vec![
            ScriptedReply::call("c1", "calculator", r#"{"expression":"6*7"}"#),
            ScriptedReply::text("42"),
        ]);
        session(&mut agent, Approval::Ask, vec![], |mut driver| async move {
            driver.send("6*7");
            driver.wait("the dialog", |s| s.dialog).await;
            // Still waiting: no request has gone out for the second step.
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(
                driver
                    .state
                    .borrow()
                    .transcript
                    .contains("calculator Waiting")
            );
            driver.type_text("y");
            driver
                .wait("the answer", |s| {
                    !s.working && s.transcript.contains("assistant: 42")
                })
                .await;
            driver.quit();
        })
        .await;
        assert_eq!(agent.backend().requests().len(), 2);
    }

    #[tokio::test]
    async fn allowing_for_the_session_stops_the_questions_for_that_tool() {
        let (mut agent, _dir) = new_agent(vec![
            ScriptedReply::call("c1", "calculator", r#"{"expression":"1+1"}"#),
            ScriptedReply::call("c2", "calculator", r#"{"expression":"2+2"}"#),
            ScriptedReply::text("done"),
        ]);
        let outcome = session(&mut agent, Approval::Ask, vec![], |mut driver| async move {
            driver.send("go");
            driver.wait("the dialog", |s| s.dialog).await;
            driver.type_text("a");
            driver
                .wait("the answer", |s| {
                    !s.working && s.transcript.contains("assistant: done")
                })
                .await;
            assert_eq!(driver.state.borrow().dialogs, 1, "asked only once");
            driver.quit();
        })
        .await;
        assert!(
            outcome
                .app
                .allowed
                .contains(&djinn_core::values::ToolName::from_static("calculator"))
        );
        let results: Vec<_> = agent.backend().requests()[2]
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.content.clone().unwrap())
            .collect();
        assert_eq!(results, vec!["2", "4"]);
    }

    #[tokio::test]
    async fn a_denial_reaches_the_model_with_the_reason() {
        let (mut agent, _dir) = new_agent(vec![
            ScriptedReply::call("c1", "calculator", r#"{"expression":"1+1"}"#),
            ScriptedReply::text("ok, I will not"),
        ]);
        let outcome = session(&mut agent, Approval::Ask, vec![], |mut driver| async move {
            driver.send("go");
            driver.wait("the dialog", |s| s.dialog).await;
            driver.type_text("n");
            driver.type_text("not that file");
            driver.press(KeyCode::Enter, KeyModifiers::NONE);
            driver
                .wait("the answer", |s| {
                    !s.working && s.transcript.contains("will not")
                })
                .await;
            driver.quit();
        })
        .await;
        let second = &agent.backend().requests()[1].messages;
        let tool = second.iter().find(|m| m.role == Role::Tool).unwrap();
        assert_eq!(
            tool.content.as_deref(),
            Some("Error: the user denied this tool call because: not that file")
        );
        assert!(
            outcome.screen.contains("denied: not that file"),
            "{}",
            outcome.screen
        );
    }

    #[tokio::test]
    async fn cancelling_while_an_approval_is_pending_discards_the_turn() {
        let (mut agent, _dir) = new_agent(vec![
            ScriptedReply::call("c1", "calculator", r#"{"expression":"1+1"}"#),
            ScriptedReply::text("second turn answer"),
        ]);
        let outcome = session(&mut agent, Approval::Ask, vec![], |mut driver| async move {
            driver.send("first");
            driver.wait("the dialog", |s| s.dialog).await;
            driver.press(KeyCode::Char('c'), KeyModifiers::CONTROL);
            driver
                .wait("cancellation", |s| {
                    !s.working && !s.dialog && s.transcript.contains("turn cancelled")
                })
                .await;
            driver.send("second");
            driver
                .wait("the second answer", |s| {
                    !s.working && s.transcript.contains("second turn answer")
                })
                .await;
            driver.quit();
        })
        .await;
        // Only the second turn is in the conversation.
        assert_eq!(
            roles(&agent),
            vec![Role::System, Role::User, Role::Assistant]
        );
        assert_eq!(agent.history()[1].content.as_deref(), Some("second"));
        assert!(
            outcome.screen.contains("turn cancelled"),
            "{}",
            outcome.screen
        );
    }

    #[tokio::test]
    async fn cancelling_a_streaming_answer_stops_it_and_keeps_the_session_usable() {
        let (mut agent, _dir) = new_agent(vec![
            ScriptedReply::error_after("partial text", BackendError::StreamEndedEarly),
            ScriptedReply::Hang,
            ScriptedReply::text("fine"),
        ]);
        session(
            &mut agent,
            Approval::Auto,
            vec![],
            |mut driver| async move {
                // The first turn fails after streaming some text.
                driver.send("one");
                driver
                    .wait("the error", |s| {
                        !s.working && s.transcript.contains("error: the stream ended")
                    })
                    .await;
                // The second never answers; cancel it.
                driver.send("two");
                driver.wait("working", |s| s.working).await;
                driver.press(KeyCode::Char('c'), KeyModifiers::CONTROL);
                driver
                    .wait("cancellation", |s| {
                        !s.working && s.transcript.contains("turn cancelled")
                    })
                    .await;
                driver.send("three");
                driver
                    .wait("the answer", |s| {
                        !s.working && s.transcript.contains("assistant: fine")
                    })
                    .await;
                driver.quit();
            },
        )
        .await;
        assert_eq!(
            roles(&agent),
            vec![Role::System, Role::User, Role::Assistant],
            "failed and cancelled turns are not in the history"
        );
        assert_eq!(agent.history()[1].content.as_deref(), Some("three"));
    }

    #[tokio::test]
    async fn the_step_limit_is_shown_and_the_next_message_is_accepted() {
        let config = AgentConfig {
            max_steps: 1,
            ..AgentConfig::default()
        };
        let (mut agent, _dir) = agent_with(
            vec![
                ScriptedReply::call("c1", "calculator", r#"{"expression":"1+1"}"#),
                ScriptedReply::text("after the limit"),
            ],
            config,
        );
        session(
            &mut agent,
            Approval::Auto,
            vec![],
            |mut driver| async move {
                driver.send("loop");
                driver
                    .wait("the limit", |s| {
                        !s.working && s.transcript.contains("step limit")
                    })
                    .await;
                driver.send("again");
                driver
                    .wait("the answer", |s| {
                        !s.working && s.transcript.contains("after the limit")
                    })
                    .await;
                driver.quit();
            },
        )
        .await;
    }

    #[tokio::test]
    async fn a_startup_warning_is_in_the_transcript_and_messages_still_work() {
        let (mut agent, _dir) = new_agent(vec![ScriptedReply::text("hello")]);
        let outcome = session(
            &mut agent,
            Approval::Auto,
            vec![(
                Level::Warn,
                "could not connect to http://localhost:1/v1".into(),
            )],
            |mut driver| async move {
                driver.send("hi");
                driver
                    .wait("the answer", |s| {
                        !s.working && s.transcript.contains("assistant: hello")
                    })
                    .await;
                driver.quit();
            },
        )
        .await;
        assert!(
            outcome.screen.contains("could not connect"),
            "{}",
            outcome.screen
        );
    }

    #[tokio::test]
    async fn reset_forgets_the_conversation_and_the_screen() {
        let (mut agent, _dir) =
            new_agent(vec![ScriptedReply::text("one"), ScriptedReply::text("two")]);
        let outcome = session(
            &mut agent,
            Approval::Auto,
            vec![],
            |mut driver| async move {
                driver.send("first");
                driver
                    .wait("the answer", |s| {
                        !s.working && s.transcript.contains("assistant: one")
                    })
                    .await;
                driver.send("/reset");
                driver
                    .wait("an empty transcript", |s| s.transcript.is_empty())
                    .await;
                driver.send("second");
                driver
                    .wait("the answer", |s| {
                        !s.working && s.transcript.contains("assistant: two")
                    })
                    .await;
                driver.quit();
            },
        )
        .await;
        assert!(!outcome.screen.contains("first"), "{}", outcome.screen);
        // The second request holds only the system message and the new one.
        assert_eq!(agent.backend().requests()[1].messages.len(), 2);
    }

    #[tokio::test]
    async fn exit_and_a_closed_event_stream_end_the_session() {
        let (mut agent, _dir) = new_agent(vec![]);
        session(&mut agent, Approval::Auto, vec![], |driver| async move {
            driver.send("/exit");
        })
        .await;
        let (mut agent, _dir) = new_agent(vec![]);
        session(&mut agent, Approval::Auto, vec![], |driver| async move {
            drop(driver); // the terminal went away
        })
        .await;
    }

    #[tokio::test]
    async fn pasted_text_is_not_sent_and_the_quit_keys_do_not_quit_a_non_empty_input() {
        let (mut agent, _dir) = new_agent(vec![]);
        let outcome = session(
            &mut agent,
            Approval::Auto,
            vec![],
            |mut driver| async move {
                driver
                    .keys
                    .unbounded_send(Ok(Event::Paste("line one\nline two".into())))
                    .unwrap();
                // Ctrl-C when idle only explains how to quit.
                driver.press(KeyCode::Char('c'), KeyModifiers::CONTROL);
                driver
                    .wait("the hint", |s| s.transcript.contains("Ctrl-D"))
                    .await;
                // Ctrl-D on a non-empty input edits instead of quitting.
                driver.quit();
                driver.type_text("!");
                driver.press(KeyCode::Enter, KeyModifiers::ALT);
                // Closing the event stream is what ends the session.
                tokio::time::sleep(Duration::from_millis(30)).await;
                drop(driver);
            },
        )
        .await;
        assert!(agent.backend().requests().is_empty());
        assert_eq!(outcome.app.input.text(), "line one\nline two!\n");
    }

    #[tokio::test]
    async fn a_failing_terminal_event_is_an_error() {
        let (mut agent, _dir) = new_agent(vec![]);
        let (keys, events) = unbounded::<io::Result<Event>>();
        keys.unbounded_send(Err(io::Error::other("boom"))).unwrap();
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        let settings = settings(&agent, Approval::Auto);
        let error = run(&mut agent, settings, events, &mut terminal, |_| {}, vec![])
            .await
            .map(|_| ())
            .unwrap_err();
        assert_eq!(error.to_string(), "boom");
    }

    #[tokio::test]
    async fn a_command_can_be_completed_with_enter_and_then_run_without_a_model_request() {
        let (mut agent, _dir) = new_agent(vec![]);
        let outcome = session(
            &mut agent,
            Approval::Auto,
            vec![],
            |mut driver| async move {
                driver.type_text("/to");
                driver.press(KeyCode::Enter, KeyModifiers::NONE); // completes to /tools
                driver.press(KeyCode::Enter, KeyModifiers::NONE); // runs it
                driver
                    .wait("the tools listing", |s| {
                        s.transcript.contains("calculator:")
                    })
                    .await;
                driver.quit();
            },
        )
        .await;
        assert!(agent.backend().requests().is_empty());
        assert!(outcome.screen.contains("calculator:"), "{}", outcome.screen);
    }

    #[tokio::test]
    async fn the_popup_is_on_screen_while_a_command_is_typed() {
        let (mut agent, _dir) = new_agent(vec![]);
        let outcome = session(&mut agent, Approval::Auto, vec![], |driver| async move {
            driver.type_text("/");
            // Wait for the draw that follows the key.
            tokio::time::sleep(Duration::from_millis(50)).await;
            driver.type_text("t");
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(driver);
        })
        .await;
        assert!(outcome.screen.contains("/tools"), "{}", outcome.screen);
        assert!(!outcome.screen.contains("/reset"), "{}", outcome.screen);
    }
}
