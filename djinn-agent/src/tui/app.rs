//! Application state and the pure `update` function (design D2, D4).
//!
//! [`App::update`] takes one [`Msg`] and returns the [`Effect`]s the runtime must perform; it never
//! touches the terminal, the agent or the clock. (The one side effect it has is answering an open
//! approval dialog through its one-shot channel, which is synchronous.)

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind};
use djinn_core::values::{ModelId, ToolArguments, ToolCallId, ToolName};

use super::approval::{Dialog, Outcome, PendingApproval, SessionAllow};
use super::input::InputBuffer;
use super::text::sanitise;
use crate::agent::{AgentEvent, Decision, TurnError};
use crate::commands::{Command, HELP, UnknownCommand};
use crate::settings::{Approval, BaseUrl};

/// Lines moved by one turn of the mouse wheel.
const WHEEL_LINES: usize = 3;

/// A tool as listed by `/tools`.
#[derive(Clone, Debug)]
pub struct ToolInfo {
    pub name: ToolName,
    pub description: String,
}

/// What the status bar and `/tools` show.
#[derive(Clone, Debug)]
pub struct Settings {
    pub model: ModelId,
    pub base_url: BaseUrl,
    pub approval: Approval,
    pub max_steps: usize,
    pub tools: Vec<ToolInfo>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolState {
    /// Waiting for the user's decision.
    Waiting,
    Running,
    Ok,
    Failed,
    Denied,
    /// The turn was cancelled before the call ended.
    Cancelled,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolBlock {
    pub id: ToolCallId,
    pub name: ToolName,
    /// The arguments as the model wrote them, made safe to show.
    pub arguments: String,
    pub state: ToolState,
    pub elapsed: Option<Duration>,
    /// The result, or the error, made safe to show.
    pub output: String,
    /// Why the user denied the call, if they said.
    pub reason: Option<String>,
    pub expanded: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EntryKind {
    User(String),
    Assistant { text: String, streaming: bool },
    Tool(ToolBlock),
    Notice { level: Level, text: String },
}

/// Source of entry revisions; unique across the program so a cache can never confuse two entries.
static REVISION: AtomicU64 = AtomicU64::new(1);

fn next_revision() -> u64 {
    REVISION.fetch_add(1, Ordering::Relaxed)
}

/// One item of the transcript. `rev` changes whenever the content does, which lets the view reuse
/// the lines it already laid out.
#[derive(Clone, Debug)]
pub struct Entry {
    kind: EntryKind,
    rev: u64,
}

impl Entry {
    fn new(kind: EntryKind) -> Self {
        Self {
            kind,
            rev: next_revision(),
        }
    }

    pub fn kind(&self) -> &EntryKind {
        &self.kind
    }

    pub fn rev(&self) -> u64 {
        self.rev
    }

    /// Mutable access; counts as a change.
    fn change(&mut self) -> &mut EntryKind {
        self.rev = next_revision();
        &mut self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    /// A turn is running; `step` is the number of the current model call (0 before the first).
    Working {
        step: usize,
    },
}

/// Everything that can happen to the UI.
#[derive(Debug)]
pub enum Msg {
    Key(KeyEvent),
    Paste(String),
    Mouse(MouseEvent),
    Tick,
    Agent(AgentEvent),
    Approval(PendingApproval),
    TurnEnded(Result<String, TurnError>),
    Notice(Level, String),
}

/// What the runtime has to do after an update.
#[derive(Debug, PartialEq, Eq)]
pub enum Effect {
    /// Start a turn with this user message.
    Send(String),
    /// Drop the running turn.
    Cancel,
    /// Forget the conversation.
    Reset,
    Quit,
}

pub struct App {
    pub settings: Settings,
    entries: Vec<Entry>,
    pub input: InputBuffer,
    pub phase: Phase,
    /// How many lines the view is scrolled up from the bottom; the view clamps it.
    pub scroll: usize,
    /// Lines that arrived while the view was scrolled up.
    pub new_lines: usize,
    /// Transcript height at the last draw (so growth can be told apart from scrolling).
    pub last_total: usize,
    /// Transcript width at the last draw; a resize is not "new lines".
    pub last_width: u16,
    /// Laid-out lines per entry, kept by the view.
    pub(super) cache: Vec<Option<super::view::Cached>>,
    /// How far page up/down moves; set by the view from the window height.
    pub page: usize,
    /// The tool block selected for expanding, as an entry index.
    pub focus: Option<usize>,
    pub dialog: Option<Dialog>,
    pub allowed: SessionAllow,
    pub spinner: usize,
}

impl App {
    pub fn new(settings: Settings, allowed: SessionAllow) -> Self {
        Self {
            settings,
            entries: Vec::new(),
            input: InputBuffer::new(),
            phase: Phase::Idle,
            scroll: 0,
            new_lines: 0,
            last_total: 0,
            last_width: 0,
            cache: Vec::new(),
            page: 10,
            focus: None,
            dialog: None,
            allowed,
            spinner: 0,
        }
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn is_working(&self) -> bool {
        matches!(self.phase, Phase::Working { .. })
    }

    fn push(&mut self, kind: EntryKind) {
        self.entries.push(Entry::new(kind));
    }

    fn notice(&mut self, level: Level, text: impl Into<String>) {
        self.push(EntryKind::Notice {
            level,
            text: sanitise(&text.into()),
        });
    }

    pub fn update(&mut self, msg: Msg) -> Vec<Effect> {
        match msg {
            Msg::Key(key) => self.on_key(key),
            Msg::Paste(text) => {
                if self.dialog.is_none() {
                    self.input.insert_str(&text);
                }
                vec![]
            }
            Msg::Mouse(mouse) => {
                match mouse.kind {
                    MouseEventKind::ScrollUp => self.scroll_up(WHEEL_LINES),
                    MouseEventKind::ScrollDown => self.scroll_down(WHEEL_LINES),
                    _ => {}
                }
                vec![]
            }
            Msg::Tick => {
                if self.is_working() {
                    self.spinner = self.spinner.wrapping_add(1);
                }
                vec![]
            }
            Msg::Agent(event) => {
                self.on_agent(event);
                vec![]
            }
            Msg::Approval(pending) => {
                self.on_approval(pending);
                vec![]
            }
            Msg::TurnEnded(result) => {
                self.on_turn_ended(result);
                vec![]
            }
            Msg::Notice(level, text) => {
                self.notice(level, text);
                vec![]
            }
        }
    }

    // ---- scrolling ----

    fn scroll_up(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_add(lines);
    }

    fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
        if self.scroll == 0 {
            self.new_lines = 0;
        }
    }

    // ---- keys ----

    fn on_key(&mut self, key: KeyEvent) -> Vec<Effect> {
        if key.kind == KeyEventKind::Release {
            return vec![];
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        // These work in every state.
        match key.code {
            KeyCode::Char('c') if ctrl => return self.interrupt(),
            KeyCode::PageUp => {
                self.scroll_up(self.page.max(1));
                return vec![];
            }
            KeyCode::PageDown => {
                self.scroll_down(self.page.max(1));
                return vec![];
            }
            KeyCode::Home if ctrl => {
                self.scroll = usize::MAX;
                return vec![];
            }
            KeyCode::End if ctrl => {
                self.scroll = 0;
                self.new_lines = 0;
                return vec![];
            }
            _ => {}
        }

        if self.dialog.is_some() {
            self.on_dialog_key(key);
            return vec![];
        }

        if self.focus.is_some() && self.on_focus_key(key) {
            return vec![];
        }

        match key.code {
            KeyCode::Char('d') if ctrl && self.input.is_empty() => {
                return vec![Effect::Quit];
            }
            KeyCode::Enter if alt || shift => self.input.newline(),
            KeyCode::Char('j') if ctrl => self.input.newline(),
            KeyCode::Enter => return self.submit(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            KeyCode::Char('k') if ctrl => self.input.kill_to_end(),
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('w') if ctrl => self.input.delete_word_back(),
            KeyCode::Char('d') if ctrl => self.input.delete(),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Up => self.input.up(),
            KeyCode::Down => self.input.down(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Tab => self.move_focus(true),
            KeyCode::BackTab => self.move_focus(false),
            KeyCode::Char(c) if !ctrl && !alt => self.input.insert_char(c),
            _ => {}
        }
        vec![]
    }

    /// Keys while a tool block is selected. Returns whether the key was used.
    fn on_focus_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Tab => self.move_focus(true),
            KeyCode::BackTab => self.move_focus(false),
            KeyCode::Esc => self.focus = None,
            KeyCode::Enter | KeyCode::Char(' ') if self.input.is_empty() => {
                if let Some(index) = self.focus {
                    self.toggle(index);
                }
            }
            _ => {
                // Any other key is for the input box.
                self.focus = None;
                return false;
            }
        }
        true
    }

    fn tool_indices(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e.kind, EntryKind::Tool(_)))
            .map(|(i, _)| i)
            .collect()
    }

    /// Selects the next (or previous) tool block; walking off either end deselects.
    fn move_focus(&mut self, forward: bool) {
        let tools = self.tool_indices();
        if tools.is_empty() {
            self.focus = None;
            return;
        }
        let position = self
            .focus
            .and_then(|focus| tools.iter().position(|&i| i == focus));
        self.focus = match (position, forward) {
            (None, true) => tools.last().copied(),
            (None, false) => tools.first().copied(),
            (Some(p), true) => p.checked_sub(1).map(|older| tools[older]),
            (Some(p), false) => tools.get(p + 1).copied(),
        };
    }

    fn toggle(&mut self, index: usize) {
        if let Some(entry) = self.entries.get_mut(index)
            && let EntryKind::Tool(block) = entry.change()
        {
            block.expanded = !block.expanded;
        }
    }

    fn interrupt(&mut self) -> Vec<Effect> {
        if self.is_working() {
            self.cancel_turn();
            vec![Effect::Cancel]
        } else {
            self.notice(
                Level::Info,
                "Ctrl-C does not quit while idle. Type /exit or press Ctrl-D to quit.",
            );
            vec![]
        }
    }

    fn cancel_turn(&mut self) {
        self.phase = Phase::Idle;
        self.dialog = None; // dropping the sender tells the waiting turn
        for entry in &mut self.entries {
            let open = match &entry.kind {
                EntryKind::Tool(block) => {
                    matches!(block.state, ToolState::Waiting | ToolState::Running)
                }
                EntryKind::Assistant { streaming, .. } => *streaming,
                _ => false,
            };
            if !open {
                continue;
            }
            match entry.change() {
                EntryKind::Tool(block) => block.state = ToolState::Cancelled,
                EntryKind::Assistant { streaming, .. } => *streaming = false,
                _ => {}
            }
        }
        self.notice(Level::Warn, "turn cancelled");
    }

    // ---- sending and commands ----

    fn submit(&mut self) -> Vec<Effect> {
        if self.is_working() || self.input.is_blank() {
            return vec![]; // the text stays in the box
        }
        let Some(text) = self.input.take() else {
            return vec![];
        };
        let trimmed = text.trim();
        if let Some(command) = trimmed.strip_prefix('/') {
            return self.run_command(command);
        }
        self.push(EntryKind::User(text.clone()));
        self.phase = Phase::Working { step: 0 };
        self.scroll = 0;
        self.new_lines = 0;
        vec![Effect::Send(text)]
    }

    fn run_command(&mut self, command: &str) -> Vec<Effect> {
        match Command::parse(command) {
            Ok(Command::Exit) => return vec![Effect::Quit],
            Ok(Command::Reset) => {
                self.entries.clear();
                self.focus = None;
                self.scroll = 0;
                self.new_lines = 0;
                self.last_total = 0;
                self.cache.clear();
                self.allowed.clear();
                return vec![Effect::Reset];
            }
            Ok(Command::Tools) => {
                let text = if self.settings.tools.is_empty() {
                    "no tools are registered".to_owned()
                } else {
                    self.settings
                        .tools
                        .iter()
                        .map(|tool| format!("{}: {}", tool.name, tool.description))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                self.notice(Level::Info, text);
            }
            Ok(Command::Help) => self.notice(Level::Info, help_text()),
            Err(UnknownCommand(other)) => self.notice(
                Level::Error,
                format!("unknown command `/{other}`. Commands:\n{HELP}"),
            ),
        }
        vec![]
    }

    // ---- approval dialog ----

    fn on_approval(&mut self, pending: PendingApproval) {
        self.push(EntryKind::Tool(ToolBlock {
            id: pending.id.clone(),
            name: pending.name.clone(),
            arguments: sanitise(pending.arguments.as_str()),
            state: ToolState::Waiting,
            elapsed: None,
            output: String::new(),
            reason: None,
            expanded: false,
        }));
        self.dialog = Some(Dialog::new(pending));
    }

    fn on_dialog_key(&mut self, key: KeyEvent) {
        let Some(dialog) = &mut self.dialog else {
            return;
        };
        let decision = match dialog.handle_key(key) {
            Outcome::Open => return,
            Outcome::Allow => Decision::Allow,
            Outcome::AllowSession => {
                self.allowed.allow(dialog.pending.name.clone());
                Decision::Allow
            }
            Outcome::Deny(reason) => Decision::Deny { reason },
        };
        if let Some(dialog) = self.dialog.take() {
            // The turn may be gone already; nothing to tell then.
            let _ = dialog.pending.respond.send(decision);
        }
    }

    // ---- agent events ----

    fn tool_mut(&mut self, id: &ToolCallId) -> Option<&mut ToolBlock> {
        self.entries.iter_mut().rev().find_map(|entry| {
            if matches!(&entry.kind, EntryKind::Tool(block) if block.id == *id) {
                match entry.change() {
                    EntryKind::Tool(block) => Some(block),
                    _ => None,
                }
            } else {
                None
            }
        })
    }

    fn end_streaming(&mut self) {
        if let Some(entry) = self.entries.last_mut()
            && matches!(
                entry.kind,
                EntryKind::Assistant {
                    streaming: true,
                    ..
                }
            )
            && let EntryKind::Assistant { streaming, .. } = entry.change()
        {
            *streaming = false;
        }
    }

    fn new_tool_block(
        &mut self,
        id: ToolCallId,
        name: ToolName,
        arguments: &ToolArguments,
        state: ToolState,
    ) {
        self.push(EntryKind::Tool(ToolBlock {
            id,
            name,
            arguments: sanitise(arguments.as_str()),
            state,
            elapsed: None,
            output: String::new(),
            reason: None,
            expanded: false,
        }));
    }

    fn on_agent(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::StepStarted { step } => self.phase = Phase::Working { step },
            AgentEvent::Text(piece) => {
                if let Some(entry) = self.entries.last_mut()
                    && matches!(
                        entry.kind,
                        EntryKind::Assistant {
                            streaming: true,
                            ..
                        }
                    )
                    && let EntryKind::Assistant { text, .. } = entry.change()
                {
                    text.push_str(&piece);
                } else {
                    self.push(EntryKind::Assistant {
                        text: piece,
                        streaming: true,
                    });
                }
            }
            AgentEvent::ToolStarted {
                id,
                name,
                arguments,
            } => {
                self.end_streaming();
                match self.tool_mut(&id) {
                    Some(block) => block.state = ToolState::Running,
                    None => self.new_tool_block(id, name, &arguments, ToolState::Running),
                }
            }
            AgentEvent::ToolDenied { id, name, reason } => {
                self.end_streaming();
                let reason = reason.map(|r| sanitise(&r));
                match self.tool_mut(&id) {
                    Some(block) => {
                        block.state = ToolState::Denied;
                        block.reason = reason;
                    }
                    None => {
                        self.new_tool_block(id, name, &ToolArguments::empty(), ToolState::Denied);
                        if let Some(EntryKind::Tool(block)) =
                            self.entries.last_mut().map(Entry::change)
                        {
                            block.reason = reason;
                        }
                    }
                }
            }
            AgentEvent::ToolFinished {
                id,
                ok,
                output,
                elapsed,
                ..
            } => {
                if let Some(block) = self.tool_mut(&id) {
                    block.state = if ok { ToolState::Ok } else { ToolState::Failed };
                    block.elapsed = Some(elapsed);
                    block.output = sanitise(&output);
                }
            }
            // The turn's end carries the step-limit message, and the result ends the turn.
            AgentEvent::StepLimit { .. } | AgentEvent::TurnFinished => {}
        }
    }

    fn on_turn_ended(&mut self, result: Result<String, TurnError>) {
        if !self.is_working() {
            return; // cancelled earlier
        }
        self.phase = Phase::Idle;
        self.dialog = None;
        self.end_streaming();
        match result {
            Ok(_) => {}
            Err(error @ TurnError::StepLimit(_)) => self.notice(Level::Warn, error.to_string()),
            Err(TurnError::Backend(error)) => self.notice(Level::Error, format!("error: {error}")),
        }
    }
}

/// The text of `/help`: the commands and the keys.
pub fn help_text() -> String {
    format!(
        "{HELP}\n\nKeys:\n\
         Enter            send the message\n\
         Alt-Enter        new line (also Shift-Enter or Ctrl-J)\n\
         Up/Down          recall earlier messages (first/last line of the input)\n\
         PageUp/PageDown  scroll the conversation (also the mouse wheel)\n\
         Ctrl-Home/End    jump to the start/end of the conversation\n\
         Tab/Shift-Tab    select a tool call; Enter or Space opens or closes it\n\
         Ctrl-C           cancel the running turn\n\
         Ctrl-D           quit (empty input)\n\
         Text selection needs Shift held while dragging (mouse capture is on)."
    )
}

#[cfg(test)]
mod tests {
    use tokio::sync::oneshot;

    use super::*;

    fn settings() -> Settings {
        Settings {
            model: ModelId::from_static("test-model"),
            base_url: BaseUrl::from_static("http://localhost:8080/v1"),
            approval: Approval::Ask,
            max_steps: 8,
            tools: vec![ToolInfo {
                name: ToolName::from_static("calculator"),
                description: "Does math".into(),
            }],
        }
    }

    fn app() -> App {
        App::new(settings(), SessionAllow::default())
    }

    fn key(code: KeyCode) -> Msg {
        Msg::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn ctrl(c: char) -> Msg {
        Msg::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }

    fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            app.update(key(KeyCode::Char(c)));
        }
    }

    fn send(app: &mut App, text: &str) -> Vec<Effect> {
        type_text(app, text);
        app.update(key(KeyCode::Enter))
    }

    fn kinds(app: &App) -> Vec<&EntryKind> {
        app.entries().iter().map(Entry::kind).collect()
    }

    fn tool_started(id: &str, name: &str, arguments: &str) -> Msg {
        Msg::Agent(AgentEvent::ToolStarted {
            id: ToolCallId::new(id).unwrap(),
            name: ToolName::new(name).unwrap(),
            arguments: ToolArguments::new(arguments).unwrap(),
        })
    }

    fn tool_finished(id: &str, name: &str, ok: bool, output: &str) -> Msg {
        Msg::Agent(AgentEvent::ToolFinished {
            id: ToolCallId::new(id).unwrap(),
            name: ToolName::new(name).unwrap(),
            ok,
            output: output.into(),
            elapsed: Duration::from_millis(12),
        })
    }

    fn the_tool(app: &App) -> &ToolBlock {
        app.entries()
            .iter()
            .find_map(|e| match e.kind() {
                EntryKind::Tool(block) => Some(block),
                _ => None,
            })
            .expect("a tool block")
    }

    // ---- sending ----

    #[test]
    fn enter_sends_the_message_and_starts_working() {
        let mut app = app();
        let effects = send(&mut app, "What is 2+2?");
        assert_eq!(effects, vec![Effect::Send("What is 2+2?".into())]);
        assert!(app.is_working());
        assert!(app.input.is_empty());
        assert_eq!(kinds(&app), vec![&EntryKind::User("What is 2+2?".into())]);
    }

    #[test]
    fn a_blank_message_is_not_sent() {
        let mut app = app();
        assert_eq!(send(&mut app, "   "), vec![]);
        assert!(!app.is_working());
        assert!(app.entries().is_empty());
    }

    #[test]
    fn typing_while_busy_keeps_the_text_until_the_turn_ends() {
        let mut app = app();
        send(&mut app, "first");
        type_text(&mut app, "second");
        assert_eq!(app.update(key(KeyCode::Enter)), vec![]);
        assert_eq!(app.input.text(), "second");
        app.update(Msg::TurnEnded(Ok("done".into())));
        assert_eq!(
            app.update(key(KeyCode::Enter)),
            vec![Effect::Send("second".into())]
        );
    }

    #[test]
    fn alt_enter_and_ctrl_j_insert_newlines() {
        let mut app = app();
        type_text(&mut app, "a");
        app.update(Msg::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)));
        type_text(&mut app, "b");
        app.update(ctrl('j'));
        type_text(&mut app, "c");
        assert_eq!(app.input.text(), "a\nb\nc");
        assert!(app.entries().is_empty());
    }

    #[test]
    fn a_paste_fills_the_input_without_sending() {
        let mut app = app();
        let effects = app.update(Msg::Paste("one\ntwo\nthree".into()));
        assert_eq!(effects, vec![]);
        assert_eq!(app.input.line_count(), 3);
        assert!(!app.is_working());
    }

    #[test]
    fn key_releases_are_ignored() {
        let mut app = app();
        let mut release = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        release.kind = KeyEventKind::Release;
        app.update(Msg::Key(release));
        assert!(app.input.is_empty());
    }

    // ---- streaming and tools ----

    #[test]
    fn streamed_text_grows_one_assistant_entry() {
        let mut app = app();
        send(&mut app, "hi");
        app.update(Msg::Agent(AgentEvent::StepStarted { step: 1 }));
        app.update(Msg::Agent(AgentEvent::Text("Hello".into())));
        app.update(Msg::Agent(AgentEvent::Text(" world".into())));
        assert_eq!(
            kinds(&app)[1],
            &EntryKind::Assistant {
                text: "Hello world".into(),
                streaming: true
            }
        );
        app.update(Msg::TurnEnded(Ok("Hello world".into())));
        assert_eq!(
            kinds(&app)[1],
            &EntryKind::Assistant {
                text: "Hello world".into(),
                streaming: false
            }
        );
        assert!(!app.is_working());
    }

    #[test]
    fn the_step_number_follows_the_agent() {
        let mut app = app();
        send(&mut app, "hi");
        app.update(Msg::Agent(AgentEvent::StepStarted { step: 2 }));
        assert_eq!(app.phase, Phase::Working { step: 2 });
    }

    #[test]
    fn a_tool_call_goes_from_running_to_ok_with_its_result() {
        let mut app = app();
        send(&mut app, "2+2");
        app.update(Msg::Agent(AgentEvent::Text("Let me check.".into())));
        app.update(tool_started("c1", "calculator", r#"{"expression":"2+2"}"#));
        assert_eq!(the_tool(&app).state, ToolState::Running);
        app.update(tool_finished("c1", "calculator", true, "4"));
        let block = the_tool(&app);
        assert_eq!(block.state, ToolState::Ok);
        assert_eq!(block.output, "4");
        assert_eq!(block.arguments, r#"{"expression":"2+2"}"#);
        assert_eq!(block.elapsed, Some(Duration::from_millis(12)));
        // The text before the call is finished; later text starts a new entry.
        app.update(Msg::Agent(AgentEvent::Text("It is 4.".into())));
        assert!(matches!(
            kinds(&app)[2..],
            [EntryKind::Tool(_), EntryKind::Assistant { .. }]
        ));
        assert!(matches!(
            kinds(&app)[1],
            EntryKind::Assistant {
                streaming: false,
                ..
            }
        ));
    }

    #[test]
    fn a_failed_tool_is_marked_failed() {
        let mut app = app();
        send(&mut app, "x");
        app.update(tool_started("c1", "calculator", "{}"));
        app.update(tool_finished("c1", "calculator", false, "Error: bad"));
        assert_eq!(the_tool(&app).state, ToolState::Failed);
        assert_eq!(the_tool(&app).output, "Error: bad");
    }

    #[test]
    fn tool_output_and_arguments_are_made_safe() {
        let mut app = app();
        send(&mut app, "x");
        app.update(tool_started("c1", "calculator", r#"{"a":"\u001b[2J"}"#));
        app.update(tool_finished("c1", "calculator", true, "\u{1b}[2Jgone"));
        let block = the_tool(&app);
        assert!(!block.output.contains('\u{1b}') && !block.arguments.contains('\u{1b}'));
    }

    #[test]
    fn a_denied_call_shows_the_reason() {
        let mut app = app();
        send(&mut app, "x");
        app.update(Msg::Agent(AgentEvent::ToolDenied {
            id: ToolCallId::from_static("c1"),
            name: ToolName::from_static("read_file"),
            reason: Some("private".into()),
        }));
        let block = the_tool(&app);
        assert_eq!(block.state, ToolState::Denied);
        assert_eq!(block.reason.as_deref(), Some("private"));
    }

    #[test]
    fn backend_errors_and_the_step_limit_are_shown_and_the_session_continues() {
        let mut app = app();
        send(&mut app, "x");
        app.update(Msg::TurnEnded(Err(TurnError::Backend(
            crate::backend::BackendError::StreamEndedEarly,
        ))));
        let EntryKind::Notice { level, text } = kinds(&app).last().copied().unwrap() else {
            panic!("expected a notice")
        };
        assert_eq!(*level, Level::Error);
        assert!(text.starts_with("error: the stream ended"), "{text}");
        assert!(!app.is_working());

        send(&mut app, "y");
        app.update(Msg::TurnEnded(Err(TurnError::StepLimit(8))));
        let EntryKind::Notice { level, text } = kinds(&app).last().copied().unwrap() else {
            panic!("expected a notice")
        };
        assert_eq!(*level, Level::Warn);
        assert!(text.contains("step limit"), "{text}");
        assert_eq!(send(&mut app, "z"), vec![Effect::Send("z".into())]);
    }

    // ---- commands ----

    #[test]
    fn exit_and_quit_end_the_program() {
        for command in ["/exit", "/quit"] {
            let mut app = app();
            assert_eq!(send(&mut app, command), vec![Effect::Quit]);
        }
    }

    #[test]
    fn ctrl_d_quits_only_on_an_empty_input() {
        let mut app = app();
        assert_eq!(app.update(ctrl('d')), vec![Effect::Quit]);
        type_text(&mut app, "abc");
        app.update(key(KeyCode::Home));
        assert_eq!(app.update(ctrl('d')), vec![]);
        assert_eq!(app.input.text(), "bc", "Ctrl-D deletes a character instead");
    }

    #[test]
    fn unknown_commands_are_an_error_and_send_nothing() {
        let mut app = app();
        assert_eq!(send(&mut app, "/frobnicate"), vec![]);
        let EntryKind::Notice { level, text } = kinds(&app)[0] else {
            panic!("expected a notice")
        };
        assert_eq!(*level, Level::Error);
        assert!(
            text.contains("/frobnicate") && text.contains("/reset"),
            "{text}"
        );
        assert!(!app.is_working());
    }

    #[test]
    fn tools_and_help_are_listed_in_the_transcript() {
        let mut app = app();
        send(&mut app, "/tools");
        send(&mut app, "/help");
        let texts: Vec<_> = app
            .entries()
            .iter()
            .filter_map(|e| match e.kind() {
                EntryKind::Notice { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts[0], "calculator: Does math");
        assert!(
            texts[1].contains("/exit") && texts[1].contains("Ctrl-C"),
            "{}",
            texts[1]
        );
    }

    #[test]
    fn reset_clears_the_transcript_and_the_session_allowance() {
        let mut app = app();
        send(&mut app, "hi");
        app.update(Msg::TurnEnded(Ok("x".into())));
        app.allowed.allow(ToolName::from_static("calculator"));
        assert_eq!(send(&mut app, "/reset"), vec![Effect::Reset]);
        assert!(app.entries().is_empty());
        assert!(!app.allowed.contains(&ToolName::from_static("calculator")));
    }

    // ---- interrupt ----

    #[test]
    fn ctrl_c_cancels_a_running_turn_and_marks_open_calls() {
        let mut app = app();
        send(&mut app, "x");
        app.update(Msg::Agent(AgentEvent::Text("partial".into())));
        app.update(tool_started("c1", "calculator", "{}"));
        assert_eq!(app.update(ctrl('c')), vec![Effect::Cancel]);
        assert!(!app.is_working());
        assert_eq!(the_tool(&app).state, ToolState::Cancelled);
        let EntryKind::Notice { text, .. } = kinds(&app).last().copied().unwrap() else {
            panic!("expected a notice")
        };
        assert_eq!(text, "turn cancelled");
        // A late result from the dropped turn changes nothing.
        app.update(Msg::TurnEnded(Ok("late".into())));
        assert_eq!(app.entries().len(), 4);
        // And the next message is accepted.
        assert_eq!(send(&mut app, "y"), vec![Effect::Send("y".into())]);
    }

    #[test]
    fn ctrl_c_when_idle_explains_how_to_quit_and_does_not_quit() {
        let mut app = app();
        assert_eq!(app.update(ctrl('c')), vec![]);
        let EntryKind::Notice { text, .. } = kinds(&app)[0] else {
            panic!("expected a notice")
        };
        assert!(text.contains("/exit") && text.contains("Ctrl-D"), "{text}");
    }

    // ---- scrolling ----

    #[test]
    fn page_keys_and_the_wheel_scroll_and_never_below_the_bottom() {
        let mut app = app();
        app.page = 5;
        app.update(key(KeyCode::PageUp));
        assert_eq!(app.scroll, 5);
        app.update(Msg::Mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        }));
        assert_eq!(app.scroll, 8);
        app.update(key(KeyCode::PageDown));
        app.update(key(KeyCode::PageDown));
        assert_eq!(app.scroll, 0);
        app.update(Msg::Key(KeyEvent::new(
            KeyCode::Home,
            KeyModifiers::CONTROL,
        )));
        assert_eq!(app.scroll, usize::MAX);
        app.new_lines = 4;
        app.update(Msg::Key(KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL)));
        assert_eq!((app.scroll, app.new_lines), (0, 0));
    }

    #[test]
    fn sending_a_message_returns_to_the_bottom() {
        let mut app = app();
        app.scroll = 30;
        send(&mut app, "hi");
        assert_eq!(app.scroll, 0);
    }

    // ---- focus ----

    fn app_with_two_tools() -> App {
        let mut app = app();
        send(&mut app, "x");
        for id in ["c1", "c2"] {
            app.update(tool_started(id, "calculator", "{}"));
            app.update(tool_finished(id, "calculator", true, "ok"));
        }
        app.update(Msg::TurnEnded(Ok("done".into())));
        app
    }

    #[test]
    fn tab_selects_tool_blocks_and_enter_toggles_them() {
        let mut app = app_with_two_tools();
        app.update(key(KeyCode::Tab));
        assert_eq!(app.focus, Some(2), "the most recent call first");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.entries()[2].kind(), EntryKind::Tool(b) if b.expanded));
        app.update(key(KeyCode::Char(' ')));
        assert!(matches!(app.entries()[2].kind(), EntryKind::Tool(b) if !b.expanded));
        app.update(key(KeyCode::Tab));
        assert_eq!(app.focus, Some(1));
        app.update(key(KeyCode::Tab));
        assert_eq!(app.focus, None, "walking off the end deselects");
        assert!(app.input.is_empty());
    }

    #[test]
    fn typing_or_escape_leaves_the_selection() {
        let mut app = app_with_two_tools();
        app.update(key(KeyCode::Tab));
        app.update(key(KeyCode::Esc));
        assert_eq!(app.focus, None);
        app.update(key(KeyCode::Tab));
        type_text(&mut app, "h");
        assert_eq!(app.focus, None);
        assert_eq!(app.input.text(), "h");
    }

    #[test]
    fn expanding_a_block_changes_its_revision() {
        let mut app = app_with_two_tools();
        let before = app.entries()[2].rev();
        app.update(key(KeyCode::Tab));
        app.update(key(KeyCode::Enter));
        assert_ne!(app.entries()[2].rev(), before);
    }

    // ---- approval ----

    fn approval(
        app: &mut App,
        name: &str,
        needs_confirmation: bool,
    ) -> oneshot::Receiver<Decision> {
        let (respond, receiver) = oneshot::channel();
        app.update(Msg::Approval(PendingApproval {
            id: ToolCallId::from_static("c1"),
            name: ToolName::new(name).unwrap(),
            arguments: ToolArguments::new(r#"{"path":"a.txt"}"#).unwrap(),
            needs_confirmation,
            respond,
        }));
        receiver
    }

    #[test]
    fn a_request_waits_in_a_dialog_and_shows_a_waiting_block() {
        let mut app = app();
        send(&mut app, "x");
        let _answer = approval(&mut app, "read_file", false);
        assert!(app.dialog.is_some());
        assert_eq!(the_tool(&app).state, ToolState::Waiting);
    }

    #[test]
    fn allowing_once_answers_and_the_next_call_asks_again() {
        let mut app = app();
        send(&mut app, "x");
        let mut answer = approval(&mut app, "read_file", false);
        app.update(key(KeyCode::Char('y')));
        assert_eq!(answer.try_recv().unwrap(), Decision::Allow);
        assert!(app.dialog.is_none());
        assert!(!app.allowed.contains(&ToolName::from_static("read_file")));
    }

    #[test]
    fn allowing_for_the_session_remembers_the_tool() {
        let mut app = app();
        send(&mut app, "x");
        let mut answer = approval(&mut app, "read_file", false);
        app.update(key(KeyCode::Char('a')));
        assert_eq!(answer.try_recv().unwrap(), Decision::Allow);
        assert!(app.allowed.contains(&ToolName::from_static("read_file")));
        assert!(!app.allowed.contains(&ToolName::from_static("calculator")));
    }

    #[test]
    fn the_session_choice_is_not_offered_for_tools_that_need_confirmation() {
        let mut app = app();
        send(&mut app, "x");
        let mut answer = approval(&mut app, "danger", true);
        app.update(key(KeyCode::Char('a')));
        assert!(answer.try_recv().is_err(), "still waiting");
        assert!(!app.allowed.contains(&ToolName::from_static("danger")));
        assert!(app.dialog.is_some());
    }

    #[test]
    fn denying_carries_the_typed_reason() {
        let mut app = app();
        send(&mut app, "x");
        let mut answer = approval(&mut app, "read_file", false);
        app.update(key(KeyCode::Char('n')));
        type_text(&mut app, "not that file");
        assert!(
            app.input.is_empty(),
            "typing goes to the dialog, not the input box"
        );
        app.update(key(KeyCode::Enter));
        assert_eq!(
            answer.try_recv().unwrap(),
            Decision::Deny {
                reason: Some("not that file".into())
            }
        );
    }

    #[test]
    fn ctrl_c_with_a_pending_approval_cancels_the_turn_and_drops_the_question() {
        let mut app = app();
        send(&mut app, "x");
        let mut answer = approval(&mut app, "read_file", false);
        assert_eq!(app.update(ctrl('c')), vec![Effect::Cancel]);
        assert!(app.dialog.is_none());
        assert!(matches!(
            answer.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert_eq!(the_tool(&app).state, ToolState::Cancelled);
    }

    #[test]
    fn scrolling_still_works_while_a_dialog_is_open() {
        let mut app = app();
        send(&mut app, "x");
        let _answer = approval(&mut app, "read_file", false);
        app.update(key(KeyCode::PageUp));
        assert!(app.scroll > 0);
        assert!(app.dialog.is_some());
    }

    #[test]
    fn ticks_only_spin_while_working() {
        let mut app = app();
        app.update(Msg::Tick);
        assert_eq!(app.spinner, 0);
        send(&mut app, "x");
        app.update(Msg::Tick);
        assert_eq!(app.spinner, 1);
    }
}
