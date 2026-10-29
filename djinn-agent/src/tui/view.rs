//! Rendering: a function of the application state (design D4).
//!
//! [`render`] also records what the next update needs to know about the layout (the page size, how
//! far the view may scroll, how much content arrived while the user was reading), which is why it
//! takes the app mutably.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Padding, Paragraph};

use super::app::{App, Entry, EntryKind, Level, Phase, ToolBlock, ToolState};
use super::approval::{Dialog, Stage};
use super::markdown;
use super::text::{hard_wrap, width, wrap_spans};
use crate::settings::Approval;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// The most input rows shown before the box scrolls.
const MAX_INPUT_ROWS: usize = 6;
/// The most commands listed in the completion popup.
const MAX_POPUP_ROWS: usize = 6;

/// Lines laid out for one entry, valid while `rev`, `width` and `focused` stay the same.
#[derive(Clone, Debug)]
pub struct Cached {
    rev: u64,
    width: usize,
    focused: bool,
    lines: Vec<Line<'static>>,
}

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

/// Cuts `text` to `columns` columns, ending with `…` when something was left out.
pub fn truncate_columns(text: &str, columns: usize) -> String {
    if width(text) <= columns {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if used + w + 1 > columns {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('…');
    out
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Pieces of a line that is cut at the end if it is too wide.
fn fit(segments: Vec<(String, Style)>, columns: usize) -> Line<'static> {
    let mut spans = Vec::new();
    let mut remaining = columns;
    let mut iter = segments.into_iter().peekable();
    while let Some((text, style)) = iter.next() {
        let w = width(&text);
        if w <= remaining && (iter.peek().is_some() || w <= remaining) {
            remaining -= w;
            spans.push(Span::styled(text, style));
            continue;
        }
        spans.push(Span::styled(truncate_columns(&text, remaining), style));
        break;
    }
    Line::from(spans)
}

fn state_symbol(state: ToolState, spinner: usize) -> (String, Style) {
    match state {
        ToolState::Waiting => ("?".into(), Style::default().fg(Color::Yellow)),
        ToolState::Running => (
            SPINNER[spinner % SPINNER.len()].into(),
            Style::default().fg(Color::Cyan),
        ),
        ToolState::Ok => ("✓".into(), Style::default().fg(Color::Green)),
        ToolState::Failed => ("✗".into(), Style::default().fg(Color::Red)),
        ToolState::Denied => ("⊘".into(), Style::default().fg(Color::Red)),
        ToolState::Cancelled => ("–".into(), dim()),
    }
}

fn duration_text(block: &ToolBlock) -> Option<String> {
    let elapsed = block.elapsed?;
    Some(if elapsed.as_millis() < 1000 {
        format!("{}ms", elapsed.as_millis())
    } else {
        format!("{:.1}s", elapsed.as_secs_f64())
    })
}

fn tool_lines(
    block: &ToolBlock,
    focused: bool,
    columns: usize,
    spinner: usize,
) -> Vec<Line<'static>> {
    let (symbol, symbol_style) = state_symbol(block.state, spinner);
    let marker = if block.expanded { "▾" } else { "▸" };
    let mut head = vec![
        (format!("{marker} "), dim()),
        (format!("{symbol} "), symbol_style),
        (
            block.name.to_string(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ];
    if let Some(duration) = duration_text(block) {
        head.push((format!(" {duration}"), dim()));
    }
    if !block.arguments.trim().is_empty() && block.arguments.trim() != "{}" {
        head.push((format!(" {}", one_line(&block.arguments)), dim()));
    }
    let tail = match block.state {
        ToolState::Waiting => Some(" waiting for approval".to_owned()),
        ToolState::Running => Some(" running…".to_owned()),
        ToolState::Cancelled => Some(" cancelled".to_owned()),
        ToolState::Denied => Some(match &block.reason {
            Some(reason) => format!(" denied: {}", one_line(reason)),
            None => " denied".to_owned(),
        }),
        // The result is shown in full below an expanded block.
        ToolState::Ok | ToolState::Failed if block.expanded => None,
        ToolState::Ok | ToolState::Failed => {
            let preview = one_line(&block.output);
            (!preview.is_empty()).then(|| format!(" → {preview}"))
        }
    };
    if let Some(tail) = tail {
        head.push((tail, Style::default()));
    }
    let mut header = fit(head, columns);
    if focused {
        header = header.patch_style(Style::default().add_modifier(Modifier::REVERSED));
    }
    let mut lines = vec![header];

    if block.expanded {
        let indent = "    ";
        let inner = columns.saturating_sub(indent.len()).max(1);
        let section = |label: &str, body: &str, lines: &mut Vec<Line<'static>>| {
            lines.push(Line::from(Span::styled(format!("  {label}"), dim())));
            for source in body.split('\n') {
                for row in hard_wrap(source, inner, Style::default()) {
                    let mut spans = vec![Span::raw(indent)];
                    spans.extend(row);
                    lines.push(Line::from(spans));
                }
            }
        };
        if !block.arguments.trim().is_empty() {
            section("arguments", &block.arguments, &mut lines);
        }
        match block.state {
            ToolState::Ok if !block.output.is_empty() => {
                section("result", &block.output, &mut lines)
            }
            ToolState::Failed => section("error", &block.output, &mut lines),
            ToolState::Denied => section(
                "denied",
                block.reason.as_deref().unwrap_or("no reason given"),
                &mut lines,
            ),
            _ => {}
        }
    }
    lines
}

fn text_rows(
    text: &str,
    columns: usize,
    style: Style,
    first: &str,
    rest: &str,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let available = columns.saturating_sub(width(first)).max(1);
    for source in text.split('\n') {
        for row in wrap_spans(&[Span::styled(source.to_owned(), style)], available) {
            let prefix = if lines.is_empty() { first } else { rest };
            let mut spans = vec![Span::styled(prefix.to_owned(), style)];
            spans.extend(row);
            lines.push(Line::from(spans));
        }
    }
    lines
}

fn entry_lines(entry: &Entry, focused: bool, columns: usize, spinner: usize) -> Vec<Line<'static>> {
    match entry.kind() {
        EntryKind::User(text) => text_rows(
            text,
            columns,
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
            "› ",
            "  ",
        ),
        EntryKind::Assistant { text, streaming } => {
            let mut lines = markdown::render(text, columns);
            if *streaming {
                match lines.last_mut() {
                    Some(last) => last.spans.push(Span::styled("▍", dim())),
                    None => lines.push(Line::from(Span::styled("▍", dim()))),
                }
            }
            lines
        }
        EntryKind::Tool(block) => tool_lines(block, focused, columns, spinner),
        EntryKind::Notice { level, text } => {
            let (symbol, color) = match level {
                Level::Info => ("ℹ ", Color::Cyan),
                Level::Warn => ("! ", Color::Yellow),
                Level::Error => ("✗ ", Color::Red),
            };
            text_rows(text, columns, Style::default().fg(color), symbol, "  ")
        }
    }
}

/// All transcript lines, reusing the layout of entries that did not change.
fn transcript(app: &mut App, columns: usize) -> Vec<Line<'static>> {
    let spinner = app.spinner;
    app.cache.resize(app.entries().len(), None);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut previous_was_tool = false;
    for index in 0..app.entries().len() {
        let focused = app.focus == Some(index);
        let entry = &app.entries()[index];
        let is_tool = matches!(entry.kind(), EntryKind::Tool(_));
        let animated = matches!(
            entry.kind(),
            EntryKind::Tool(block) if block.state == ToolState::Running
        );
        let reusable = matches!(
            &app.cache[index],
            Some(cached) if !animated
                && cached.rev == entry.rev()
                && cached.width == columns
                && cached.focused == focused
        );
        if !reusable {
            let laid_out = entry_lines(entry, focused, columns, spinner);
            let rev = entry.rev();
            app.cache[index] = Some(Cached {
                rev,
                width: columns,
                focused,
                lines: laid_out,
            });
        }
        if !lines.is_empty() && !(is_tool && previous_was_tool) {
            lines.push(Line::default());
        }
        if let Some(cached) = &app.cache[index] {
            lines.extend(cached.lines.iter().cloned());
        }
        previous_was_tool = is_tool;
    }
    lines
}

fn status_line(app: &App, columns: usize) -> Line<'static> {
    let mode = match app.settings.approval {
        Approval::Ask => "ask",
        Approval::Auto => "auto",
    };
    let state = match app.phase {
        Phase::Idle => "● idle".to_owned(),
        Phase::Working { step } => format!(
            "{} working {}/{}",
            SPINNER[app.spinner % SPINNER.len()],
            step.max(1),
            app.settings.max_steps
        ),
    };
    let hints = if app.dialog.is_some() {
        "y allow · a session · n deny · Ctrl-C cancel"
    } else if app.is_working() {
        "Ctrl-C cancel · PgUp/PgDn scroll"
    } else {
        "Enter send · Alt-Enter newline · /help"
    };
    let left = format!(
        " {} · {} · approval: {} · {}",
        app.settings.model, app.settings.base_url, mode, state
    );
    let mid = format!(
        " {} · {} · {} · {}",
        app.settings.model, app.settings.base_url, mode, state
    );
    let compact = format!(" {} · {} · {}", app.settings.model, mode, state);
    let gap = columns.saturating_sub(width(&left) + width(hints) + 1);
    let full = format!("{left}{}{hints} ", " ".repeat(gap));
    // Drop the least important parts first: the hints, then the server, then whatever does not fit.
    let text = if width(&full) <= columns {
        full
    } else if width(&left) <= columns {
        left
    } else if width(&mid) <= columns {
        mid
    } else {
        truncate_columns(&compact, columns)
    };
    Line::from(Span::styled(
        text,
        Style::default().add_modifier(Modifier::REVERSED),
    ))
}

fn dialog_lines(dialog: &Dialog, columns: usize) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(vec![
        Span::raw("Run "),
        Span::styled(
            dialog.pending.name.to_string(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw("?"),
    ])];
    if dialog.pending.needs_confirmation {
        lines.push(Line::from(Span::styled(
            "This tool asks for confirmation every time.",
            Style::default().fg(Color::Yellow),
        )));
    }
    lines.push(Line::default());
    let arguments = super::text::sanitise(dialog.pending.arguments.as_str());
    if arguments.trim().is_empty() || arguments.trim() == "{}" {
        lines.push(Line::from(Span::styled("(no arguments)", dim())));
    } else {
        for source in arguments.split('\n') {
            for row in hard_wrap(source, columns, Style::default()) {
                lines.push(Line::from(row));
            }
        }
    }
    lines.push(Line::default());
    match &dialog.stage {
        Stage::Choose => {
            let mut options = vec![
                Span::styled("[y]", Style::default().fg(Color::Green)),
                Span::raw(" allow once   "),
            ];
            if dialog.can_allow_session() {
                options.push(Span::styled("[a]", Style::default().fg(Color::Green)));
                options.push(Span::raw(" allow this tool for the session   "));
            }
            options.push(Span::styled("[n]", Style::default().fg(Color::Red)));
            options.push(Span::raw(" deny"));
            lines.push(Line::from(options));
        }
        Stage::Reason(reason) => {
            lines.push(Line::from(Span::styled(
                "Why? (optional; Enter denies, Esc goes back)",
                dim(),
            )));
            lines.push(Line::from(format!("> {reason}▏")));
        }
    }
    lines
}

fn render_dialog(frame: &mut Frame, dialog: &Dialog, area: Rect) {
    let box_width = area.width.saturating_sub(4).clamp(10, 76);
    let inner_columns = box_width.saturating_sub(4) as usize;
    let mut lines = dialog_lines(dialog, inner_columns);
    let max_height = area.height.saturating_sub(2).max(3) as usize;
    if lines.len() + 2 > max_height {
        // Keep the question and the options; drop the middle of long arguments.
        let keep = max_height.saturating_sub(2);
        let tail = 3.min(keep);
        let head = keep - tail;
        let total = lines.len();
        let mut shortened: Vec<Line<'static>> = lines.drain(..head).collect();
        shortened.push(Line::from(Span::styled("…", dim())));
        shortened.extend(lines.drain(total - head - tail.saturating_sub(1)..));
        lines = shortened;
    }
    let height = (lines.len() as u16 + 2).min(area.height);
    let rect = Rect {
        x: area.x + (area.width.saturating_sub(box_width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width: box_width.min(area.width),
        height,
    };
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .padding(Padding::horizontal(1))
        .title(" Tool call ");
    frame.render_widget(Paragraph::new(lines).block(block), rect);
}

/// The command completion popup, standing on the top edge of the input box.
fn render_popup(frame: &mut Frame, app: &App, input_area: Rect) {
    let Some((shown, selected)) = app.popup() else {
        return;
    };
    let name_width = shown.rows.iter().map(|r| width(r.name)).max().unwrap_or(0);
    let all: Vec<String> = shown
        .rows
        .iter()
        .map(|r| format!("/{:<name_width$}  {}", r.name, r.description))
        .collect();

    // At most `MAX_POPUP_ROWS` rows, and only as many as fit above the input box.
    let room = (input_area.y as usize).saturating_sub(2);
    let visible = all.len().min(MAX_POPUP_ROWS).min(room);
    if visible == 0 || input_area.width < 6 {
        return;
    }
    let first = (selected + 1).saturating_sub(visible);
    let longest = all.iter().map(|r| width(r)).max().unwrap_or(0);
    let popup_width = ((longest + 4) as u16).min(input_area.width);
    let height = visible as u16 + 2;
    let rect = Rect {
        x: input_area.x,
        y: input_area.y - height,
        width: popup_width,
        height,
    };
    let inner = (popup_width as usize).saturating_sub(4);
    let lines: Vec<Line<'static>> = all
        .iter()
        .enumerate()
        .skip(first)
        .take(visible)
        .map(|(index, row)| {
            let style = if index == selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            let text = truncate_columns(row, inner);
            let pad = " ".repeat(inner.saturating_sub(width(&text)));
            Line::from(Span::styled(format!("{text}{pad}"), style))
        })
        .collect();
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(dim())
        .padding(Padding::horizontal(1))
        .title(" commands ");
    frame.render_widget(Paragraph::new(lines).block(block), rect);
}

/// Draws the whole screen.
pub fn render(app: &mut App, frame: &mut Frame) {
    let area = frame.area();
    let input_columns = area.width.saturating_sub(2) as usize;
    let layout = app.input.layout(input_columns);
    let input_rows = layout.rows.len().clamp(1, MAX_INPUT_ROWS);
    let [transcript_area, input_area, status_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(input_rows as u16 + 2),
        Constraint::Length(1),
    ])
    .areas(area);

    // Transcript.
    let columns = transcript_area.width as usize;
    let height = transcript_area.height as usize;
    let lines = transcript(app, columns);
    let total = lines.len();
    app.page = height.saturating_sub(1).max(1);
    if app.last_width == transcript_area.width && app.scroll > 0 && total > app.last_total {
        let added = total - app.last_total;
        app.scroll += added;
        app.new_lines += added;
    }
    app.last_width = transcript_area.width;
    app.last_total = total;
    app.scroll = app.scroll.min(total.saturating_sub(height));
    if app.scroll == 0 {
        app.new_lines = 0;
    }
    let start = total.saturating_sub(height + app.scroll);
    let visible: Vec<Line<'static>> = lines.into_iter().skip(start).take(height).collect();
    frame.render_widget(Paragraph::new(visible), transcript_area);
    if app.scroll > 0 && transcript_area.height > 0 {
        let label = if app.new_lines > 0 {
            format!(" ↓ {} new ", app.new_lines)
        } else {
            " ↓ more below ".to_owned()
        };
        let label_width = width(&label) as u16;
        if label_width <= transcript_area.width {
            let rect = Rect {
                x: transcript_area.right() - label_width,
                y: transcript_area.bottom() - 1,
                width: label_width,
                height: 1,
            };
            frame.render_widget(
                Paragraph::new(Span::styled(
                    label,
                    Style::default().add_modifier(Modifier::REVERSED),
                )),
                rect,
            );
        }
    }

    // Input.
    let title = if app.is_working() {
        " message (sending waits for the reply) "
    } else {
        " message "
    };
    let cursor_row = layout.cursor.0;
    let first_row = (cursor_row + 1).saturating_sub(MAX_INPUT_ROWS);
    let shown: Vec<Line<'static>> = layout
        .rows
        .iter()
        .skip(first_row)
        .take(MAX_INPUT_ROWS)
        .map(|row| Line::raw(row.clone()))
        .collect();
    let block = Block::default().borders(Borders::ALL).title(title);
    frame.render_widget(Paragraph::new(shown).block(block), input_area);

    // Status.
    frame.render_widget(
        Paragraph::new(status_line(app, status_area.width as usize)),
        status_area,
    );

    render_popup(frame, app, input_area);

    // Dialog on top, or the cursor in the input box.
    match &app.dialog {
        Some(dialog) => render_dialog(frame, dialog, transcript_area),
        None => {
            let x = input_area.x + 1 + layout.cursor.1 as u16;
            let y = input_area.y + 1 + (cursor_row - first_row) as u16;
            frame.set_cursor_position(Position::new(
                x.min(input_area.right().saturating_sub(2)),
                y,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use djinn_core::values::{ModelId, ToolArguments, ToolCallId, ToolName};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use tokio::sync::oneshot;

    use super::*;
    use crate::agent::AgentEvent;
    use crate::settings::BaseUrl;
    use crate::tui::app::{Msg, Settings, ToolInfo};
    use crate::tui::approval::{PendingApproval, SessionAllow};

    fn app() -> App {
        App::new(
            Settings {
                model: ModelId::from_static("qwen2.5"),
                base_url: BaseUrl::from_static("http://localhost:8080/v1"),
                approval: Approval::Ask,
                max_steps: 8,
                tools: vec![ToolInfo {
                    name: ToolName::from_static("calculator"),
                    description: "Does math".into(),
                }],
            },
            SessionAllow::default(),
        )
    }

    /// Draws `app` on a terminal of the given size and returns the screen as text.
    fn screen(app: &mut App, columns: u16, rows: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
        terminal.draw(|frame| render(app, frame)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..rows {
            let mut line = String::new();
            for x in 0..columns {
                line.push_str(buffer[(x, y)].symbol());
            }
            out.push_str(line.trim_end());
            out.push('\n');
        }
        out
    }

    fn type_and_send(app: &mut App, text: &str) {
        for c in text.chars() {
            app.update(Msg::Key(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        app.update(Msg::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)));
    }

    fn text_event(text: &str) -> Msg {
        Msg::Agent(AgentEvent::Text(text.into()))
    }

    fn tool_ok(app: &mut App, id: &str, name: &str, arguments: &str, output: &str) {
        app.update(Msg::Agent(AgentEvent::ToolStarted {
            id: ToolCallId::new(id).unwrap(),
            name: ToolName::new(name).unwrap(),
            arguments: ToolArguments::new(arguments).unwrap(),
        }));
        app.update(Msg::Agent(AgentEvent::ToolFinished {
            id: ToolCallId::new(id).unwrap(),
            name: ToolName::new(name).unwrap(),
            ok: true,
            output: output.into(),
            elapsed: Duration::from_millis(12),
        }));
    }

    fn pending(name: &str, arguments: &str, needs_confirmation: bool) -> PendingApproval {
        let (respond, _) = oneshot::channel();
        PendingApproval {
            id: ToolCallId::from_static("c1"),
            name: ToolName::new(name).unwrap(),
            arguments: ToolArguments::new(arguments).unwrap(),
            needs_confirmation,
            respond,
        }
    }

    #[test]
    fn an_empty_session() {
        insta::assert_snapshot!(screen(&mut app(), 60, 10));
    }

    #[test]
    fn a_streaming_markdown_answer() {
        let mut app = app();
        type_and_send(&mut app, "Show me an example");
        app.update(Msg::Agent(AgentEvent::StepStarted { step: 1 }));
        app.update(text_event(
            "# Example\n\nHere is **bold**, *italic* and `code`:\n\n- one\n- two\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\nPartial sent",
        ));
        insta::assert_snapshot!(screen(&mut app, 60, 22));
    }

    #[test]
    fn a_tool_block_collapsed() {
        let mut app = app();
        type_and_send(&mut app, "2+2?");
        tool_ok(&mut app, "c1", "calculator", r#"{"expression":"2+2"}"#, "4");
        app.update(text_event("It is 4."));
        app.update(Msg::TurnEnded(Ok("It is 4.".into())));
        insta::assert_snapshot!(screen(&mut app, 60, 12));
    }

    #[test]
    fn a_tool_block_expanded_and_selected() {
        let mut app = app();
        type_and_send(&mut app, "2+2?");
        tool_ok(
            &mut app,
            "c1",
            "calculator",
            r#"{"expression":"2+2"}"#,
            "4\nsecond line of the result",
        );
        app.update(Msg::TurnEnded(Ok("".into())));
        app.update(Msg::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));
        app.update(Msg::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)));
        insta::assert_snapshot!(screen(&mut app, 60, 14));
    }

    #[test]
    fn the_approval_dialog() {
        let mut app = app();
        type_and_send(&mut app, "read it");
        app.update(Msg::Agent(AgentEvent::StepStarted { step: 1 }));
        app.update(Msg::Approval(pending(
            "read_file",
            r#"{"path":"notes/todo.txt"}"#,
            false,
        )));
        insta::assert_snapshot!(screen(&mut app, 70, 18));
    }

    #[test]
    fn the_approval_dialog_for_a_tool_that_needs_confirmation_offers_no_session_choice() {
        let mut app = app();
        type_and_send(&mut app, "go");
        app.update(Msg::Approval(pending("danger", "{}", true)));
        let shown = screen(&mut app, 70, 18);
        assert!(
            shown.contains("asks for confirmation every time"),
            "{shown}"
        );
        assert!(shown.contains("[y]") && shown.contains("[n]"), "{shown}");
        assert!(!shown.contains("[a]"), "{shown}");
        assert!(
            shown.contains("(no arguments)") && !shown.contains("{}"),
            "{shown}"
        );
    }

    #[test]
    fn the_denial_reason_prompt() {
        let mut app = app();
        type_and_send(&mut app, "go");
        app.update(Msg::Approval(pending(
            "read_file",
            r#"{"path":"a"}"#,
            false,
        )));
        for c in "n".chars().chain("too private".chars()) {
            app.update(Msg::Key(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        let shown = screen(&mut app, 70, 18);
        assert!(shown.contains("> too private"), "{shown}");
    }

    #[test]
    fn narrow_terminal() {
        let mut app = app();
        type_and_send(
            &mut app,
            "a fairly long question that has to wrap in a narrow terminal",
        );
        tool_ok(
            &mut app,
            "c1",
            "read_file",
            r#"{"path":"some/long/path/to/a/file.txt"}"#,
            "contents of the file are here",
        );
        app.update(text_event(
            "An answer with a `very_long_identifier_that_cannot_wrap_nicely` inside.",
        ));
        insta::assert_snapshot!(screen(&mut app, 40, 16));
    }

    #[test]
    fn error_notice_and_the_status_while_working() {
        let mut app = app();
        type_and_send(&mut app, "hi");
        app.update(Msg::Agent(AgentEvent::StepStarted { step: 2 }));
        let working = screen(&mut app, 80, 8);
        assert!(working.contains("working 2/8"), "{working}");
        assert!(
            working.contains("qwen2.5") && working.contains("http://localhost:8080/v1"),
            "{working}"
        );
        assert!(working.contains("approval: ask"), "{working}");
        app.update(Msg::TurnEnded(Err(crate::agent::TurnError::Backend(
            crate::backend::BackendError::StreamEndedEarly,
        ))));
        insta::assert_snapshot!(screen(&mut app, 80, 8));
    }

    #[test]
    fn a_multi_line_input_grows_the_box_and_places_the_cursor() {
        let mut app = app();
        app.update(Msg::Paste("line one\nline two\nline three".into()));
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        terminal.draw(|frame| render(&mut app, frame)).unwrap();
        terminal.backend_mut().assert_cursor_position((11, 9));
        let shown = screen(&mut app, 40, 12);
        assert!(shown.contains("line three"), "{shown}");
    }

    #[test]
    fn escape_sequences_in_model_text_do_not_reach_the_screen() {
        let mut app = app();
        type_and_send(&mut app, "x");
        app.update(text_event("before \u{1b}[2J after"));
        tool_ok(&mut app, "c1", "calculator", "{}", "\u{1b}[31mred");
        let shown = screen(&mut app, 60, 12);
        assert!(!shown.contains('\u{1b}'), "{shown:?}");
        assert!(shown.contains('\u{241b}'), "{shown}");
    }

    #[test]
    fn scrolled_up_keeps_its_place_and_counts_new_lines() {
        let mut app = app();
        type_and_send(&mut app, "go");
        let long: String = (1..=30).map(|n| format!("line {n}\n\n")).collect();
        app.update(text_event(&long));
        let following = screen(&mut app, 40, 10);
        assert!(following.contains("line 30"), "{following}");

        app.update(Msg::Key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE)));
        let reading = screen(&mut app, 40, 10);
        assert!(!reading.contains("line 30"), "{reading}");
        assert!(reading.contains("more below"), "{reading}");
        let first_visible = reading.lines().next().unwrap().to_owned();

        app.update(text_event("\n\nnewer 1\n\nnewer 2\n\n"));
        let after = screen(&mut app, 40, 10);
        assert_eq!(
            after.lines().next().unwrap(),
            first_visible,
            "the view must not move"
        );
        assert!(after.contains("4 new"), "{after}");
        assert!(app.new_lines > 0);

        app.update(Msg::Key(KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL)));
        let bottom = screen(&mut app, 40, 10);
        assert!(bottom.contains("newer 2"), "{bottom}");
        assert_eq!(app.new_lines, 0);
    }

    #[test]
    fn resizing_rewraps_without_losing_text() {
        let mut app = app();
        type_and_send(&mut app, "go");
        let words = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        app.update(text_event(words));
        for columns in [80u16, 40, 20, 12] {
            let shown = screen(&mut app, columns, 20);
            for word in words.split(' ') {
                assert!(
                    shown.contains(word),
                    "{columns}: {word} missing in\n{shown}"
                );
            }
        }
    }

    #[test]
    fn tiny_terminals_do_not_panic() {
        let mut app = app();
        type_and_send(&mut app, "hello");
        app.update(Msg::Approval(pending(
            "read_file",
            r#"{"path":"a"}"#,
            false,
        )));
        for (w, h) in [(1, 1), (2, 2), (5, 3), (10, 4), (20, 6)] {
            let _ = screen(&mut app, w, h);
        }
    }

    #[test]
    fn truncate_columns_ends_with_an_ellipsis() {
        assert_eq!(truncate_columns("abcdef", 6), "abcdef");
        assert_eq!(truncate_columns("abcdef", 4), "abc…");
        assert_eq!(truncate_columns("你好吗", 4), "你…");
        assert_eq!(truncate_columns("abc", 0), "…");
    }

    fn typing(app: &mut App, text: &str) {
        for c in text.chars() {
            app.update(Msg::Key(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
    }

    #[test]
    fn the_completion_popup_lists_every_command() {
        let mut app = app();
        typing(&mut app, "/");
        insta::assert_snapshot!(screen(&mut app, 60, 14));
    }

    #[test]
    fn the_completion_popup_narrows_to_the_matches() {
        let mut app = app();
        typing(&mut app, "/re");
        let shown = screen(&mut app, 60, 14);
        assert!(
            shown.contains("/reset") && !shown.contains("/tools"),
            "{shown}"
        );
        insta::assert_snapshot!(shown);
    }

    #[test]
    fn the_completion_popup_highlights_the_selected_row() {
        let mut app = app();
        typing(&mut app, "/");
        app.update(Msg::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)));
        app.update(Msg::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)));
        let mut terminal = Terminal::new(TestBackend::new(60, 14)).unwrap();
        terminal.draw(|frame| render(&mut app, frame)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let reversed: Vec<String> = (0..14)
            .filter(|&y| {
                (1..30).any(|x| {
                    buffer[(x, y)].modifier.contains(Modifier::REVERSED) && y < 11 // the status bar is reversed too
                })
            })
            .map(|y| (0..60).map(|x| buffer[(x, y)].symbol()).collect::<String>())
            .collect();
        assert_eq!(reversed.len(), 1, "{reversed:?}");
        assert!(reversed[0].contains("/reset"), "{reversed:?}");
        insta::assert_snapshot!(screen(&mut app, 60, 14));
    }

    #[test]
    fn the_completion_popup_in_a_narrow_terminal() {
        let mut app = app();
        typing(&mut app, "/");
        insta::assert_snapshot!(screen(&mut app, 24, 12));
    }

    #[test]
    fn the_completion_popup_is_clipped_in_a_short_terminal_and_never_panics() {
        let mut app = app();
        typing(&mut app, "/");
        let short = screen(&mut app, 40, 7);
        assert!(short.contains("/exit"), "{short}");
        for (w, h) in [(1, 1), (3, 3), (10, 5), (40, 4), (40, 6)] {
            let _ = screen(&mut app, w, h);
        }
        insta::assert_snapshot!(short);
    }

    #[test]
    fn no_popup_for_ordinary_text() {
        let mut app = app();
        typing(&mut app, "hello /re");
        let shown = screen(&mut app, 60, 14);
        assert!(!shown.contains("commands"), "{shown}");
    }
}
