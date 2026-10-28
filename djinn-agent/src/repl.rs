//! The interactive command line: a plain read-eval-print loop, plus a one-shot mode.
//!
//! Answers go to the output writer (stdout); everything else (the prompt, tool activity, warnings,
//! errors) goes to the error writer (stderr), so the output can be piped. Interrupts (Ctrl-C) are
//! delivered through a [`Notify`] so the behaviour can be tested without real signals.

use std::io::Write;
use std::sync::Arc;

use tokio::io::{AsyncBufRead, AsyncBufReadExt};
use tokio::sync::Notify;

use crate::agent::{Agent, AgentEvent, TurnError};
use crate::backend::Backend;

/// The text shown before each input line.
pub const PROMPT: &str = "> ";

const COMMANDS: &str = "/exit (or /quit)  end the session\n/reset            forget the conversation\n/tools            list the available tools\n/help             show this list";

/// Runs the interactive loop until `/exit`, `/quit`, or end of input (Ctrl-D).
///
/// An interrupt during a turn cancels it: the turn is dropped from the history and the prompt comes
/// back. An interrupt at the prompt is ignored apart from a reminder of how to quit.
pub async fn run<B: Backend>(
    agent: &mut Agent<B>,
    input: impl AsyncBufRead + Unpin,
    out: &mut impl Write,
    err: &mut impl Write,
    interrupt: &Arc<Notify>,
) -> std::io::Result<()> {
    // `next_line` is cancellation safe, which matters because it is raced against interrupts.
    let mut lines = input.lines();

    loop {
        write!(err, "{PROMPT}")?;
        err.flush()?;

        let interrupted = interrupt.notified();
        tokio::pin!(interrupted);
        let line = tokio::select! {
            line = lines.next_line() => line?,
            _ = &mut interrupted => {
                writeln!(err, "\n(type /exit or press Ctrl-D to quit)")?;
                continue;
            }
        };
        let Some(line) = line else {
            writeln!(err)?;
            return Ok(()); // end of input
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(command) = line.strip_prefix('/') {
            if !run_command(agent, command, out, err)? {
                return Ok(());
            }
            continue;
        }

        let interrupted = interrupt.notified();
        tokio::pin!(interrupted);
        let cancelled = {
            let turn = run_turn(agent, line, out, err);
            tokio::pin!(turn);
            tokio::select! {
                result = &mut turn => { result?; false }
                _ = &mut interrupted => true,
            }
        };
        if cancelled {
            writeln!(err, "\n^C turn cancelled")?;
        }
    }
}

/// Handles `/command`. Returns `false` when the session should end.
fn run_command<B: Backend>(
    agent: &mut Agent<B>,
    command: &str,
    out: &mut impl Write,
    err: &mut impl Write,
) -> std::io::Result<bool> {
    match command.trim() {
        "exit" | "quit" => return Ok(false),
        "reset" => {
            agent.reset();
            writeln!(err, "conversation reset")?;
        }
        "tools" => {
            if agent.tools().is_empty() {
                writeln!(out, "no tools are registered")?;
            }
            for tool in agent.tools().iter() {
                writeln!(out, "{}: {}", tool.name(), tool.description())?;
            }
        }
        "help" => writeln!(out, "{COMMANDS}")?,
        other => {
            writeln!(err, "unknown command `/{other}`. Commands:\n{COMMANDS}")?;
        }
    }
    Ok(true)
}

/// Runs one turn, streaming the answer to `out` and activity to `err`. Backend errors and the step
/// limit are reported on `err`; only I/O errors fail.
async fn run_turn<B: Backend>(
    agent: &mut Agent<B>,
    message: &str,
    out: &mut impl Write,
    err: &mut impl Write,
) -> std::io::Result<()> {
    let mut io_error: Option<std::io::Error> = None;
    let mut streamed = false;
    let result = agent
        .run_turn(message, |event| {
            if io_error.is_some() {
                return;
            }
            if let Err(error) = show(&event, out, err, &mut streamed) {
                io_error = Some(error);
            }
        })
        .await;
    if let Some(error) = io_error {
        return Err(error);
    }

    // The answer was streamed without a trailing newline.
    if streamed {
        writeln!(out)?;
    }
    match result {
        Ok(_) => {}
        Err(error @ TurnError::StepLimit(_)) => writeln!(err, "{error}")?,
        Err(TurnError::Backend(error)) => writeln!(err, "error: {error}")?,
    }
    Ok(())
}

fn show(
    event: &AgentEvent,
    out: &mut impl Write,
    err: &mut impl Write,
    streamed: &mut bool,
) -> std::io::Result<()> {
    match event {
        AgentEvent::Text(text) => {
            write!(out, "{text}")?;
            out.flush()?;
            *streamed = !text.ends_with('\n') || *streamed;
        }
        AgentEvent::ToolStarted {
            name, arguments, ..
        } => {
            writeln!(err, "[tool] {name}({})", one_line(arguments.as_str(), 160))?;
        }
        AgentEvent::ToolFinished {
            name,
            ok,
            output,
            elapsed,
            ..
        } => {
            let status = if *ok { "ok" } else { "failed" };
            writeln!(
                err,
                "[tool] {name} {status} in {}ms: {}",
                elapsed.as_millis(),
                one_line(output, 120)
            )?;
        }
        AgentEvent::StepLimit { .. } | AgentEvent::TurnFinished => {}
    }
    err.flush()
}

/// Collapses whitespace so a value fits on one line, cutting it at `limit` characters.
fn one_line(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= limit {
        collapsed
    } else {
        collapsed.chars().take(limit).collect::<String>() + "..."
    }
}

/// Answers one message and prints only the answer to `out`. Returns whether it succeeded.
///
/// Nothing is written to `out` on failure, so a script can rely on empty output meaning no answer.
pub async fn run_once<B: Backend>(
    agent: &mut Agent<B>,
    message: &str,
    out: &mut impl Write,
    err: &mut impl Write,
) -> std::io::Result<bool> {
    let mut io_error: Option<std::io::Error> = None;
    let result = agent
        .run_turn(message, |event| {
            if io_error.is_some() {
                return;
            }
            // The answer is printed at the end; only tool activity is shown as it happens.
            if !matches!(event, AgentEvent::Text(_)) {
                if let Err(error) = show(&event, &mut std::io::sink(), err, &mut false) {
                    io_error = Some(error);
                }
            }
        })
        .await;
    if let Some(error) = io_error {
        return Err(error);
    }
    match result {
        Ok(answer) => {
            writeln!(out, "{answer}")?;
            Ok(true)
        }
        Err(TurnError::StepLimit(steps)) => {
            writeln!(err, "{}", TurnError::StepLimit(steps))?;
            Ok(false)
        }
        Err(TurnError::Backend(error)) => {
            writeln!(err, "error: {error}")?;
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use djinn_core::chat::Role;

    use super::*;
    use crate::agent::AgentConfig;
    use crate::backend::{BackendError, ScriptedBackend, ScriptedReply};
    use crate::tools::ToolRegistry;

    fn agent(replies: Vec<ScriptedReply>) -> (Agent<ScriptedBackend>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let tools = ToolRegistry::builtin(dir.path()).unwrap();
        (
            Agent::new(ScriptedBackend::new(replies), tools, AgentConfig::default()),
            dir,
        )
    }

    struct Session {
        out: String,
        err: String,
    }

    async fn session(agent: &mut Agent<ScriptedBackend>, input: &str) -> Session {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let interrupt = Arc::new(Notify::new());
        run(agent, input.as_bytes(), &mut out, &mut err, &interrupt)
            .await
            .unwrap();
        Session {
            out: String::from_utf8(out).unwrap(),
            err: String::from_utf8(err).unwrap(),
        }
    }

    fn roles(agent: &Agent<ScriptedBackend>) -> Vec<Role> {
        agent.history().iter().map(|m| m.role).collect()
    }

    #[tokio::test]
    async fn a_question_gets_a_streamed_answer_and_a_new_prompt() {
        let (mut agent, _dir) = agent(vec![ScriptedReply::Reply {
            pieces: vec!["The answer ".into(), "is 4.".into()],
            result: crate::backend::ChatResult {
                content: Some("The answer is 4.".into()),
                tool_calls: vec![],
                finish: djinn_core::chat::FinishReason::Stop,
            },
            delay: None,
        }]);
        let s = session(&mut agent, "What is 2+2?\n").await;

        assert_eq!(s.out, "The answer is 4.\n");
        assert_eq!(
            s.err.matches(PROMPT).count(),
            2,
            "a prompt before the question and one after: {:?}",
            s.err
        );
    }

    #[tokio::test]
    async fn blank_lines_do_not_reach_the_model() {
        let (mut agent, _dir) = agent(vec![]);
        let s = session(&mut agent, "\n   \n\t\n").await;
        assert_eq!(s.out, "");
        assert!(agent.backend().requests().is_empty());
    }

    #[tokio::test]
    async fn end_of_input_ends_the_session_cleanly() {
        let (mut agent, _dir) = agent(vec![]);
        session(&mut agent, "").await; // returns instead of hanging
    }

    #[tokio::test]
    async fn tool_activity_goes_to_stderr_and_stdout_has_only_the_answer() {
        let (mut agent, _dir) = agent(vec![
            ScriptedReply::call("c1", "calculator", r#"{"expression":"2+2"}"#),
            ScriptedReply::text("It is 4."),
        ]);
        let s = session(&mut agent, "2+2?\n").await;

        assert_eq!(s.out, "It is 4.\n");
        assert!(
            s.err.contains(r#"[tool] calculator({"expression":"2+2"})"#),
            "{}",
            s.err
        );
        assert!(s.err.contains("[tool] calculator ok"), "{}", s.err);
        assert!(s.err.contains(": 4"), "{}", s.err);
    }

    #[tokio::test]
    async fn a_failing_tool_is_shown_as_failed() {
        let (mut agent, _dir) = agent(vec![
            ScriptedReply::call("c1", "calculator", r#"{"expression":"1/0"}"#),
            ScriptedReply::text("cannot do that"),
        ]);
        let s = session(&mut agent, "go\n").await;
        assert!(s.err.contains("calculator failed"), "{}", s.err);
        assert!(s.err.contains("division by zero"), "{}", s.err);
    }

    #[tokio::test]
    async fn long_tool_arguments_and_results_are_shortened_on_one_line() {
        let long = "x".repeat(2000);
        let (mut agent, dir) = agent(vec![
            ScriptedReply::call("c1", "read_file", r#"{"path":"big.txt"}"#),
            ScriptedReply::text("read it"),
        ]);
        std::fs::write(dir.path().join("big.txt"), format!("{long}\nsecond line")).unwrap();
        let s = session(&mut agent, "go\n").await;
        let activity: Vec<&str> = s.err.lines().filter(|l| l.contains("[tool]")).collect();
        assert_eq!(activity.len(), 2, "{}", s.err);
        assert!(activity[1].chars().count() < 250, "{}", activity[1]);
        assert!(activity[1].ends_with("..."));
    }

    #[tokio::test]
    async fn exit_and_quit_end_the_session_before_later_input() {
        for command in ["/exit", "/quit", "/exit  "] {
            let (mut agent, _dir) = agent(vec![]);
            session(&mut agent, &format!("{command}\nthis must not be read\n")).await;
            assert!(agent.backend().requests().is_empty());
        }
    }

    #[tokio::test]
    async fn reset_forgets_the_conversation() {
        let (mut agent, _dir) = agent(vec![ScriptedReply::text("one"), ScriptedReply::text("two")]);
        let s = session(&mut agent, "first\n/reset\nsecond\n").await;
        assert!(s.err.contains("conversation reset"));
        assert_eq!(
            agent.backend().requests()[1].messages.len(),
            2,
            "system + the new message only"
        );
    }

    #[tokio::test]
    async fn tools_lists_every_tool_with_its_description() {
        let (mut agent, _dir) = agent(vec![]);
        let s = session(&mut agent, "/tools\n").await;
        for name in ["current_time", "calculator", "read_file"] {
            assert!(s.out.contains(&format!("{name}: ")), "{}", s.out);
        }
        assert!(agent.backend().requests().is_empty());
    }

    #[tokio::test]
    async fn help_lists_the_commands() {
        let (mut agent, _dir) = agent(vec![]);
        let s = session(&mut agent, "/help\n").await;
        for command in ["/exit", "/reset", "/tools", "/help"] {
            assert!(s.out.contains(command), "{}", s.out);
        }
    }

    #[tokio::test]
    async fn an_unknown_command_is_an_error_and_never_reaches_the_model() {
        let (mut agent, _dir) = agent(vec![]);
        let s = session(&mut agent, "/frobnicate\n").await;
        assert!(s.err.contains("unknown command `/frobnicate`"), "{}", s.err);
        assert!(s.err.contains("/exit"), "{}", s.err);
        assert_eq!(s.out, "");
        assert!(agent.backend().requests().is_empty());
    }

    #[tokio::test]
    async fn a_backend_error_is_reported_and_the_session_continues() {
        let (mut agent, _dir) = agent(vec![
            ScriptedReply::error(BackendError::Status {
                status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                message: "boom".into(),
            }),
            ScriptedReply::text("back again"),
        ]);
        let s = session(&mut agent, "one\ntwo\n").await;
        assert!(
            s.err
                .contains("error: the server returned HTTP 500 Internal Server Error: boom"),
            "{}",
            s.err
        );
        assert_eq!(s.out, "back again\n");
    }

    #[tokio::test]
    async fn the_step_limit_message_is_shown() {
        let (tools, dir) = (ToolRegistry::new(), tempfile::tempdir().unwrap());
        drop(dir);
        let config = AgentConfig {
            max_steps: 1,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(
            ScriptedBackend::new([ScriptedReply::call("c", "nope", "{}")]),
            tools,
            config,
        );
        let s = session(&mut agent, "go\n").await;
        assert!(s.err.contains("step limit"), "{}", s.err);
    }

    #[tokio::test]
    async fn a_streamed_answer_that_already_ends_in_a_newline_gets_no_extra_blank_line() {
        let (mut agent, _dir) = agent(vec![ScriptedReply::text("done\n")]);
        let s = session(&mut agent, "go\n").await;
        assert_eq!(s.out, "done\n");
    }

    // ---- interrupts ------------------------------------------------------------------------

    #[tokio::test]
    async fn ctrl_c_during_a_turn_cancels_it_and_the_prompt_returns() {
        let (mut agent, _dir) = agent(vec![
            ScriptedReply::text("earlier answer"),
            ScriptedReply::Hang,
            ScriptedReply::text("after the interrupt"),
        ]);
        let interrupt = Arc::new(Notify::new());

        // Input arrives over a pipe so the loop can be interrupted between lines.
        let (mut writer, reader) = tokio::io::duplex(1024);
        let mut out = Vec::new();
        let mut err = Vec::new();
        let signal = interrupt.clone();
        let driver = tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            writer
                .write_all(b"first\nhanging question\n")
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            signal.notify_waiters(); // Ctrl-C while the model hangs
            tokio::time::sleep(Duration::from_millis(100)).await;
            writer.write_all(b"third\n").await.unwrap();
            drop(writer);
        });

        run(
            &mut agent,
            tokio::io::BufReader::new(reader),
            &mut out,
            &mut err,
            &interrupt,
        )
        .await
        .unwrap();
        driver.await.unwrap();

        let out = String::from_utf8(out).unwrap();
        let err = String::from_utf8(err).unwrap();
        assert_eq!(out, "earlier answer\nafter the interrupt\n");
        assert!(err.contains("turn cancelled"), "{err}");
        // The cancelled turn is not in the history.
        let contents: Vec<_> = agent
            .history()
            .iter()
            .filter_map(|m| m.content.as_deref())
            .collect();
        assert!(!contents.contains(&"hanging question"), "{contents:?}");
        assert_eq!(
            roles(&agent),
            vec![
                Role::System,
                Role::User,
                Role::Assistant,
                Role::User,
                Role::Assistant
            ]
        );
    }

    #[tokio::test]
    async fn ctrl_c_at_an_empty_prompt_does_not_exit() {
        let (mut agent, _dir) = agent(vec![ScriptedReply::text("still here")]);
        let interrupt = Arc::new(Notify::new());
        let (mut writer, reader) = tokio::io::duplex(1024);
        let mut out = Vec::new();
        let mut err = Vec::new();
        let signal = interrupt.clone();
        let driver = tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            tokio::time::sleep(Duration::from_millis(100)).await;
            signal.notify_waiters(); // Ctrl-C with nothing typed
            tokio::time::sleep(Duration::from_millis(100)).await;
            writer.write_all(b"hello\n").await.unwrap();
            drop(writer);
        });

        run(
            &mut agent,
            tokio::io::BufReader::new(reader),
            &mut out,
            &mut err,
            &interrupt,
        )
        .await
        .unwrap();
        driver.await.unwrap();

        assert_eq!(String::from_utf8(out).unwrap(), "still here\n");
        assert!(String::from_utf8(err).unwrap().contains("/exit"));
    }

    // ---- one-shot mode ---------------------------------------------------------------------

    async fn once(agent: &mut Agent<ScriptedBackend>, message: &str) -> (bool, String, String) {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let ok = run_once(agent, message, &mut out, &mut err).await.unwrap();
        (
            ok,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    #[tokio::test]
    async fn once_prints_only_the_answer_and_succeeds() {
        let (mut agent, _dir) = agent(vec![
            ScriptedReply::call("c1", "calculator", r#"{"expression":"2+2"}"#),
            ScriptedReply::text("4"),
        ]);
        let (ok, out, err) = once(&mut agent, "What is 2+2? Use the calculator.").await;
        assert!(ok);
        assert_eq!(out, "4\n");
        assert!(err.contains("[tool] calculator"), "{err}");
        assert!(!err.contains(PROMPT), "no prompt in one-shot mode: {err}");
    }

    #[tokio::test]
    async fn once_prints_nothing_on_stdout_when_the_backend_fails() {
        let (mut agent, _dir) = agent(vec![ScriptedReply::error_after(
            "partial text that must not leak",
            BackendError::Connection {
                url: "http://nowhere/v1/chat/completions".parse().unwrap(),
                message: "refused".into(),
            },
        )]);
        let (ok, out, err) = once(&mut agent, "hi").await;
        assert!(!ok);
        assert_eq!(out, "");
        assert!(err.contains("http://nowhere/v1/chat/completions"), "{err}");
    }

    #[tokio::test]
    async fn once_fails_at_the_step_limit_without_output() {
        let config = AgentConfig {
            max_steps: 1,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(
            ScriptedBackend::new([ScriptedReply::call("c", "nope", "{}")]),
            ToolRegistry::new(),
            config,
        );
        let (ok, out, err) = once(&mut agent, "go").await;
        assert!(!ok);
        assert_eq!(out, "");
        assert!(err.contains("step limit"), "{err}");
    }

    #[test]
    fn one_line_collapses_whitespace_and_truncates() {
        assert_eq!(one_line("a\n  b\t c", 50), "a b c");
        assert_eq!(one_line("abcdef", 3), "abc...");
        assert_eq!(one_line("", 5), "");
    }
}
