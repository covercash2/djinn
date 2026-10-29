//! Tool approval policy and dialog state for the TUI (design D3).
//!
//! The policy ([`TuiApprover`]) runs inside the agent's turn and asks the UI through a message with
//! a one-shot reply channel; the dialog ([`Dialog`]) is the UI's half. The session allowance is
//! shared by both, so "allow for this session" takes effect on the very next call.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use djinn_core::values::{ToolArguments, ToolCallId, ToolName};
use futures::future::BoxFuture;
use tokio::sync::{mpsc, oneshot};

use super::app::Msg;
use crate::agent::{ApprovalRequest, Approver, Decision};
use crate::settings::Approval;

/// The tools the user allowed for the rest of the session.
#[derive(Clone, Debug, Default)]
pub struct SessionAllow(Arc<Mutex<HashSet<ToolName>>>);

impl SessionAllow {
    pub fn allow(&self, tool: ToolName) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(tool);
    }

    pub fn contains(&self, tool: &ToolName) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(tool)
    }

    pub fn clear(&self) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// A call waiting for the user, sent from the turn to the UI.
#[derive(Debug)]
pub struct PendingApproval {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: ToolArguments,
    pub needs_confirmation: bool,
    pub respond: oneshot::Sender<Decision>,
}

/// The policy used by the TUI.
pub struct TuiApprover {
    mode: Approval,
    allowed: SessionAllow,
    messages: mpsc::UnboundedSender<Msg>,
}

impl TuiApprover {
    pub fn new(
        mode: Approval,
        allowed: SessionAllow,
        messages: mpsc::UnboundedSender<Msg>,
    ) -> Self {
        Self {
            mode,
            allowed,
            messages,
        }
    }

    /// Whether the user has to be asked about this call.
    fn must_ask(&self, request: &ApprovalRequest<'_>) -> bool {
        if request.needs_confirmation {
            return true;
        }
        match self.mode {
            Approval::Ask => !self.allowed.contains(request.name),
            Approval::Auto => false,
        }
    }
}

impl Approver for TuiApprover {
    fn decide<'a>(&'a self, request: ApprovalRequest<'a>) -> BoxFuture<'a, Decision> {
        Box::pin(async move {
            if !self.must_ask(&request) {
                return Decision::Allow;
            }
            let (respond, answer) = oneshot::channel();
            let pending = PendingApproval {
                id: request.id.clone(),
                name: request.name.clone(),
                arguments: request.arguments.clone(),
                needs_confirmation: request.needs_confirmation,
                respond,
            };
            if self.messages.send(Msg::Approval(pending)).is_err() {
                return closed();
            }
            answer.await.unwrap_or_else(|_| closed())
        })
    }
}

/// What a call gets when nobody is left to ask.
fn closed() -> Decision {
    Decision::Deny {
        reason: Some("the terminal UI was closed".to_owned()),
    }
}

/// Which part of the approval dialog is showing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Choose,
    /// Typing the reason for a denial.
    Reason(String),
}

/// An open approval dialog.
#[derive(Debug)]
pub struct Dialog {
    pub pending: PendingApproval,
    pub stage: Stage,
    pub scroll: usize,
}

/// What a key press in the dialog decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Nothing decided yet.
    Open,
    Allow,
    AllowSession,
    Deny(Option<String>),
}

impl Dialog {
    pub fn new(pending: PendingApproval) -> Self {
        Self {
            pending,
            stage: Stage::Choose,
            scroll: 0,
        }
    }

    pub fn scroll_up(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_add(lines);
    }

    /// Whether "allow for this session" is offered.
    pub fn can_allow_session(&self) -> bool {
        !self.pending.needs_confirmation
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        if key.kind == KeyEventKind::Release {
            return Outcome::Open;
        }
        let plain = !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
        match &mut self.stage {
            Stage::Choose => match key.code {
                KeyCode::Char('y' | 'Y') | KeyCode::Enter if plain => Outcome::Allow,
                KeyCode::Char('a' | 'A') if plain && self.can_allow_session() => {
                    Outcome::AllowSession
                }
                KeyCode::Char('n' | 'N' | 'd' | 'D') if plain => {
                    self.stage = Stage::Reason(String::new());
                    Outcome::Open
                }
                _ => Outcome::Open,
            },
            Stage::Reason(reason) => match key.code {
                KeyCode::Enter => {
                    let reason = reason.trim();
                    Outcome::Deny((!reason.is_empty()).then(|| reason.to_owned()))
                }
                KeyCode::Esc => {
                    self.stage = Stage::Choose;
                    Outcome::Open
                }
                KeyCode::Backspace => {
                    reason.pop();
                    Outcome::Open
                }
                KeyCode::Char(c) if plain && !c.is_control() => {
                    reason.push(c);
                    Outcome::Open
                }
                _ => Outcome::Open,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending(needs_confirmation: bool) -> (PendingApproval, oneshot::Receiver<Decision>) {
        let (respond, receiver) = oneshot::channel();
        (
            PendingApproval {
                id: ToolCallId::from_static("c1"),
                name: ToolName::from_static("read_file"),
                arguments: ToolArguments::new(r#"{"path":"a.txt"}"#).unwrap(),
                needs_confirmation,
                respond,
            },
            receiver,
        )
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_text(dialog: &mut Dialog, text: &str) {
        for c in text.chars() {
            dialog.handle_key(press(KeyCode::Char(c)));
        }
    }

    #[test]
    fn y_and_enter_allow_once() {
        for code in [KeyCode::Char('y'), KeyCode::Enter] {
            let mut dialog = Dialog::new(pending(false).0);
            assert_eq!(dialog.handle_key(press(code)), Outcome::Allow);
        }
    }

    #[test]
    fn a_allows_for_the_session_unless_the_tool_needs_confirmation() {
        let mut dialog = Dialog::new(pending(false).0);
        assert!(dialog.can_allow_session());
        assert_eq!(
            dialog.handle_key(press(KeyCode::Char('a'))),
            Outcome::AllowSession
        );
        let mut careful = Dialog::new(pending(true).0);
        assert!(!careful.can_allow_session());
        assert_eq!(careful.handle_key(press(KeyCode::Char('a'))), Outcome::Open);
    }

    #[test]
    fn denying_asks_for_an_optional_reason() {
        let mut dialog = Dialog::new(pending(false).0);
        assert_eq!(dialog.handle_key(press(KeyCode::Char('n'))), Outcome::Open);
        assert_eq!(dialog.stage, Stage::Reason(String::new()));
        type_text(&mut dialog, "not that fiile");
        dialog.handle_key(press(KeyCode::Left)); // ignored
        for _ in 0..3 {
            dialog.handle_key(press(KeyCode::Backspace));
        }
        type_text(&mut dialog, "le");
        assert_eq!(
            dialog.handle_key(press(KeyCode::Enter)),
            Outcome::Deny(Some("not that file".into()))
        );
    }

    #[test]
    fn a_denial_without_a_reason_has_none() {
        let mut dialog = Dialog::new(pending(false).0);
        dialog.handle_key(press(KeyCode::Char('d')));
        type_text(&mut dialog, "   ");
        assert_eq!(
            dialog.handle_key(press(KeyCode::Enter)),
            Outcome::Deny(None)
        );
    }

    #[test]
    fn escape_goes_back_from_the_reason() {
        let mut dialog = Dialog::new(pending(false).0);
        dialog.handle_key(press(KeyCode::Char('n')));
        type_text(&mut dialog, "oops");
        dialog.handle_key(press(KeyCode::Esc));
        assert_eq!(dialog.stage, Stage::Choose);
        assert_eq!(dialog.handle_key(press(KeyCode::Char('y'))), Outcome::Allow);
    }

    #[test]
    fn other_keys_and_releases_do_nothing() {
        let mut dialog = Dialog::new(pending(false).0);
        assert_eq!(dialog.handle_key(press(KeyCode::Char('x'))), Outcome::Open);
        let mut release = press(KeyCode::Char('y'));
        release.kind = KeyEventKind::Release;
        assert_eq!(dialog.handle_key(release), Outcome::Open);
        let ctrl_y = KeyEvent::new(KeyCode::Char('y'), KeyModifiers::CONTROL);
        assert_eq!(dialog.handle_key(ctrl_y), Outcome::Open);
    }

    #[test]
    fn the_session_allowance_is_shared_between_clones() {
        let allowed = SessionAllow::default();
        let other = allowed.clone();
        let name = ToolName::from_static("calculator");
        assert!(!other.contains(&name));
        allowed.allow(name.clone());
        assert!(other.contains(&name));
        other.clear();
        assert!(!allowed.contains(&name));
    }

    fn request<'a>(
        id: &'a ToolCallId,
        name: &'a ToolName,
        arguments: &'a ToolArguments,
        needs_confirmation: bool,
    ) -> ApprovalRequest<'a> {
        ApprovalRequest {
            id,
            name,
            arguments,
            needs_confirmation,
        }
    }

    struct Call {
        id: ToolCallId,
        name: ToolName,
        arguments: ToolArguments,
    }

    fn call() -> Call {
        Call {
            id: ToolCallId::from_static("c1"),
            name: ToolName::from_static("calculator"),
            arguments: ToolArguments::new("{}").unwrap(),
        }
    }

    #[tokio::test]
    async fn ask_mode_asks_for_every_call_and_returns_the_answer() {
        let (messages, mut ui) = mpsc::unbounded_channel();
        let approver = TuiApprover::new(Approval::Ask, SessionAllow::default(), messages);
        let c = call();
        let decision = approver.decide(request(&c.id, &c.name, &c.arguments, false));
        let answer = async {
            let Some(Msg::Approval(pending)) = ui.recv().await else {
                panic!("expected an approval message")
            };
            assert_eq!(pending.name.as_str(), "calculator");
            pending.respond.send(Decision::Allow).unwrap();
        };
        let (decision, ()) = tokio::join!(decision, answer);
        assert_eq!(decision, Decision::Allow);
    }

    #[tokio::test]
    async fn ask_mode_skips_tools_allowed_for_the_session() {
        let (messages, mut ui) = mpsc::unbounded_channel();
        let allowed = SessionAllow::default();
        let c = call();
        allowed.allow(c.name.clone());
        let approver = TuiApprover::new(Approval::Ask, allowed, messages);
        let decision = approver
            .decide(request(&c.id, &c.name, &c.arguments, false))
            .await;
        assert_eq!(decision, Decision::Allow);
        assert!(ui.try_recv().is_err(), "the user must not be asked");
    }

    #[tokio::test]
    async fn auto_mode_only_asks_about_tools_that_need_confirmation() {
        let (messages, mut ui) = mpsc::unbounded_channel();
        let approver = TuiApprover::new(Approval::Auto, SessionAllow::default(), messages);
        let c = call();
        assert_eq!(
            approver
                .decide(request(&c.id, &c.name, &c.arguments, false))
                .await,
            Decision::Allow
        );
        assert!(ui.try_recv().is_err());

        let careful = approver.decide(request(&c.id, &c.name, &c.arguments, true));
        let answer = async {
            let Some(Msg::Approval(pending)) = ui.recv().await else {
                panic!("expected an approval message")
            };
            assert!(pending.needs_confirmation);
            pending
                .respond
                .send(Decision::Deny { reason: None })
                .unwrap();
        };
        let (decision, ()) = tokio::join!(careful, answer);
        assert_eq!(decision, Decision::Deny { reason: None });
    }

    #[tokio::test]
    async fn a_tool_that_needs_confirmation_is_asked_even_if_allowed_for_the_session() {
        let (messages, mut ui) = mpsc::unbounded_channel();
        let allowed = SessionAllow::default();
        let c = call();
        allowed.allow(c.name.clone());
        let approver = TuiApprover::new(Approval::Ask, allowed, messages);
        let decision = approver.decide(request(&c.id, &c.name, &c.arguments, true));
        let answer = async {
            let Some(Msg::Approval(pending)) = ui.recv().await else {
                panic!("expected an approval message")
            };
            pending.respond.send(Decision::Allow).unwrap();
        };
        let (decision, ()) = tokio::join!(decision, answer);
        assert_eq!(decision, Decision::Allow);
    }

    #[tokio::test]
    async fn a_closed_ui_denies_instead_of_hanging() {
        let (messages, ui) = mpsc::unbounded_channel();
        drop(ui);
        let approver = TuiApprover::new(Approval::Ask, SessionAllow::default(), messages);
        let c = call();
        let decision = approver
            .decide(request(&c.id, &c.name, &c.arguments, false))
            .await;
        assert!(matches!(decision, Decision::Deny { .. }));

        // The UI took the request and went away without answering.
        let (messages, mut ui) = mpsc::unbounded_channel();
        let approver = TuiApprover::new(Approval::Ask, SessionAllow::default(), messages);
        let decision = approver.decide(request(&c.id, &c.name, &c.arguments, false));
        let vanish = async {
            let _ = ui.recv().await; // dropped here
        };
        let (decision, ()) = tokio::join!(decision, vanish);
        assert!(matches!(decision, Decision::Deny { .. }));
    }
}
