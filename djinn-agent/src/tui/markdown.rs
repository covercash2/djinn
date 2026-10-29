//! Markdown to styled, wrapped lines (design D5).
//!
//! Only CommonMark without extensions is parsed, so a table or other unsupported syntax shows up as
//! its source text. A single line break in the source stays a line break (chat text is not prose
//! that should be re-flowed). The whole text is rendered again on every streamed delta;
//! `pulldown-cmark` treats an unclosed fence as code up to the end of the text, which is what a
//! half-streamed code block should look like.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::text::{Row, hard_wrap, row_width, sanitise, wrap_spans};

const QUOTE_GUTTER: &str = "│ ";

fn code_style() -> Style {
    Style::default()
        .fg(Color::Indexed(252))
        .bg(Color::Indexed(236))
}

fn inline_code_style() -> Style {
    Style::default().fg(Color::Cyan)
}

fn heading_style(level: HeadingLevel) -> Style {
    let base = Style::default().add_modifier(Modifier::BOLD);
    match level {
        HeadingLevel::H1 => base.fg(Color::Magenta).add_modifier(Modifier::UNDERLINED),
        HeadingLevel::H2 => base.fg(Color::Magenta),
        _ => base.fg(Color::Blue),
    }
}

/// What encloses the current block.
enum Container {
    Quote,
    /// A list; `next` is the number of the next item for an ordered list.
    List {
        next: Option<u64>,
    },
    /// An item of the innermost list. `marker` is shown at the item's first line only.
    Item {
        marker: String,
        used: bool,
    },
}

struct Renderer {
    width: usize,
    lines: Vec<Line<'static>>,
    containers: Vec<Container>,
    /// Source lines of the paragraph/heading being collected, each a list of styled pieces.
    segments: Vec<Row>,
    styles: Vec<Style>,
    code: Option<String>,
    link: Option<String>,
}

impl Renderer {
    fn new(width: usize) -> Self {
        Self {
            width: width.max(1),
            lines: Vec::new(),
            containers: Vec::new(),
            segments: vec![Row::new()],
            styles: vec![Style::default()],
            code: None,
            link: None,
        }
    }

    fn style(&self) -> Style {
        self.styles.last().copied().unwrap_or_default()
    }

    fn with_style(&mut self, patch: impl FnOnce(Style) -> Style) {
        let next = patch(self.style());
        self.styles.push(next);
    }

    fn push_text(&mut self, text: &str, style: Style) {
        let mut parts = text.split('\n');
        if let Some(first) = parts.next() {
            self.append(first, style);
        }
        for part in parts {
            self.segments.push(Row::new());
            self.append(part, style);
        }
    }

    fn append(&mut self, text: &str, style: Style) {
        if text.is_empty() {
            return;
        }
        self.segments
            .last_mut()
            .expect("at least one segment")
            .push(Span::styled(text.to_owned(), style));
    }

    fn has_inline(&self) -> bool {
        self.segments.iter().any(|row| !row.is_empty())
    }

    /// The prefix of the first and of the following rows of the block about to be emitted.
    fn prefixes(&mut self) -> (Row, Row) {
        let mut first = Row::new();
        let mut rest = Row::new();
        let last_item = self
            .containers
            .iter()
            .rposition(|c| matches!(c, Container::Item { .. }));
        for (index, container) in self.containers.iter_mut().enumerate() {
            match container {
                Container::Quote => {
                    for row in [&mut first, &mut rest] {
                        row.push(Span::styled(
                            QUOTE_GUTTER,
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                }
                Container::List { .. } => {}
                Container::Item { marker, used } => {
                    let pad = " ".repeat(super::text::width(marker));
                    if Some(index) == last_item && !*used {
                        first.push(Span::styled(
                            marker.clone(),
                            Style::default().fg(Color::Yellow),
                        ));
                        *used = true;
                    } else {
                        first.push(Span::raw(pad.clone()));
                    }
                    rest.push(Span::raw(pad));
                }
            }
        }
        (first, rest)
    }

    fn blank_line_before_block(&mut self) {
        let inside_item = self
            .containers
            .iter()
            .any(|c| matches!(c, Container::Item { .. }));
        let last_blank = self
            .lines
            .last()
            .is_none_or(|line| line.spans.iter().all(|s| s.content.trim().is_empty()));
        if !inside_item && !last_blank {
            self.lines.push(Line::default());
        }
    }

    /// Emits the collected paragraph or heading.
    fn flush_inline(&mut self) {
        if !self.has_inline() {
            self.segments = vec![Row::new()];
            return;
        }
        let segments = std::mem::replace(&mut self.segments, vec![Row::new()]);
        let (first, rest) = self.prefixes();
        let available = self.width.saturating_sub(row_width(&first)).max(1);
        let mut first_row = true;
        for segment in segments {
            for row in wrap_spans(&segment, available) {
                let mut line = if first_row {
                    first.clone()
                } else {
                    rest.clone()
                };
                line.extend(row);
                self.lines.push(Line::from(line));
                first_row = false;
            }
        }
    }

    fn emit_code(&mut self, code: &str) {
        self.blank_line_before_block();
        // Mark the item's marker as used by the first code line, like a paragraph would.
        let (first, rest) = self.prefixes();
        let available = self.width.saturating_sub(row_width(&rest)).max(1);
        let body = code.strip_suffix('\n').unwrap_or(code);
        let mut first_row = true;
        for source in body.split('\n') {
            for row in hard_wrap(source, available, code_style()) {
                let mut line = if first_row {
                    first.clone()
                } else {
                    rest.clone()
                };
                line.extend(row);
                self.lines.push(Line::from(line));
                first_row = false;
            }
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => self.blank_line_before_block(),
            Tag::Heading { level, .. } => {
                self.blank_line_before_block();
                self.with_style(|_| heading_style(level));
            }
            Tag::BlockQuote(_) => {
                self.flush_inline();
                self.blank_line_before_block();
                self.containers.push(Container::Quote);
            }
            Tag::CodeBlock(_) => {
                self.flush_inline();
                self.code = Some(String::new());
            }
            Tag::List(start) => {
                self.flush_inline();
                if self.containers.is_empty() {
                    self.blank_line_before_block();
                }
                self.containers.push(Container::List { next: start });
            }
            Tag::Item => {
                self.flush_inline();
                let marker = match self.containers.last_mut() {
                    Some(Container::List { next: Some(n) }) => {
                        let marker = format!("{n}. ");
                        *n += 1;
                        marker
                    }
                    _ => "• ".to_owned(),
                };
                self.containers.push(Container::Item {
                    marker,
                    used: false,
                });
            }
            Tag::Emphasis => self.with_style(|s| s.add_modifier(Modifier::ITALIC)),
            Tag::Strong => self.with_style(|s| s.add_modifier(Modifier::BOLD)),
            Tag::Strikethrough => self.with_style(|s| s.add_modifier(Modifier::CROSSED_OUT)),
            Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => {
                self.link = Some(dest_url.to_string());
                self.with_style(|s| s.add_modifier(Modifier::UNDERLINED));
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.flush_inline(),
            TagEnd::Heading(_) => {
                self.flush_inline();
                self.styles.pop();
            }
            TagEnd::BlockQuote(_) => {
                self.flush_inline();
                self.containers.pop();
            }
            TagEnd::CodeBlock => {
                if let Some(code) = self.code.take() {
                    self.emit_code(&code);
                }
            }
            TagEnd::List(_) => {
                self.flush_inline();
                self.containers.pop();
            }
            TagEnd::Item => {
                self.flush_inline();
                // An empty item still shows its marker.
                if let Some(Container::Item { used: false, .. }) = self.containers.last() {
                    self.append("", Style::default());
                    let (first, _) = self.prefixes();
                    self.lines.push(Line::from(first));
                }
                self.containers.pop();
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.styles.pop();
            }
            TagEnd::Link | TagEnd::Image => {
                self.styles.pop();
                if let Some(url) = self.link.take() {
                    let dim = Style::default().fg(Color::DarkGray);
                    self.append(&format!(" ({url})"), dim);
                }
            }
            _ => {}
        }
    }

    fn event(&mut self, event: Event<'_>) {
        if let Some(code) = &mut self.code {
            match event {
                Event::Text(text) => code.push_str(&text),
                Event::End(TagEnd::CodeBlock) => self.end(TagEnd::CodeBlock),
                _ => {}
            }
            return;
        }
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) | Event::Html(text) | Event::InlineHtml(text) => {
                let style = self.style();
                self.push_text(&text, style);
            }
            Event::Code(text) => {
                let style = self.style().patch(inline_code_style());
                self.push_text(&text, style);
            }
            Event::SoftBreak | Event::HardBreak => self.segments.push(Row::new()),
            Event::Rule => {
                self.flush_inline();
                self.blank_line_before_block();
                let (_, rest) = self.prefixes();
                let available = self.width.saturating_sub(row_width(&rest)).max(1);
                let mut line = rest;
                line.push(Span::styled(
                    "─".repeat(available.min(40)),
                    Style::default().fg(Color::DarkGray),
                ));
                self.lines.push(Line::from(line));
            }
            _ => {}
        }
    }
}

/// Renders `markdown` for a view `width` columns wide. Control characters are replaced first.
pub fn render(markdown: &str, width: usize) -> Vec<Line<'static>> {
    let text = sanitise(markdown);
    let mut renderer = Renderer::new(width);
    for event in Parser::new_ext(&text, Options::empty()) {
        renderer.event(event);
    }
    renderer.flush_inline();
    renderer.lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    fn rendered(markdown: &str, width: usize) -> Vec<String> {
        text(&render(markdown, width))
    }

    fn style_of(lines: &[Line<'_>], needle: &str) -> Style {
        lines
            .iter()
            .flat_map(|l| &l.spans)
            .find(|s| s.content.contains(needle))
            .unwrap_or_else(|| panic!("no span with {needle}"))
            .style
    }

    #[test]
    fn plain_text_is_shown_unchanged_and_line_breaks_are_kept() {
        assert_eq!(rendered("hello world", 40), vec!["hello world"]);
        assert_eq!(rendered("one\ntwo\nthree", 40), vec!["one", "two", "three"]);
        assert_eq!(
            rendered("a * b * c and snake_case_name", 40),
            vec!["a * b * c and snake_case_name"]
        );
    }

    #[test]
    fn long_plain_text_wraps_at_word_boundaries() {
        assert_eq!(
            rendered("the quick brown fox jumps", 12),
            vec!["the quick", "brown fox", "jumps"]
        );
    }

    #[test]
    fn paragraphs_are_separated_by_a_blank_line() {
        assert_eq!(rendered("one\n\ntwo", 40), vec!["one", "", "two"]);
    }

    #[test]
    fn headings_lose_their_hashes_and_are_bold() {
        let lines = render("# Title\n\nbody", 40);
        assert_eq!(text(&lines), vec!["Title", "", "body"]);
        assert!(
            style_of(&lines, "Title")
                .add_modifier
                .contains(Modifier::BOLD)
        );
    }

    #[test]
    fn emphasis_strong_and_inline_code_are_styled() {
        let lines = render("a *it* **bo** `co` z", 40);
        assert_eq!(text(&lines), vec!["a it bo co z"]);
        assert!(
            style_of(&lines, "it")
                .add_modifier
                .contains(Modifier::ITALIC)
        );
        assert!(style_of(&lines, "bo").add_modifier.contains(Modifier::BOLD));
        assert_eq!(style_of(&lines, "co").fg, Some(Color::Cyan));
    }

    #[test]
    fn bullet_and_numbered_lists_have_hanging_indents() {
        assert_eq!(
            rendered("- alpha beta gamma\n- two", 12),
            vec!["• alpha beta", "  gamma", "• two"]
        );
        assert_eq!(
            rendered("1. first\n2. second", 40),
            vec!["1. first", "2. second"]
        );
        assert_eq!(
            rendered("3. third\n4. fourth", 40),
            vec!["3. third", "4. fourth"]
        );
    }

    #[test]
    fn nested_lists_indent_under_their_parent() {
        assert_eq!(rendered("- a\n  - b\n- c", 40), vec!["• a", "  • b", "• c"]);
    }

    #[test]
    fn block_quotes_get_a_gutter_on_every_row() {
        assert_eq!(
            rendered("> quoted words here", 10),
            vec!["│ quoted", "│ words", "│ here"]
        );
    }

    #[test]
    fn horizontal_rules_draw_a_line() {
        let out = rendered("above\n\n---\n\nbelow", 20);
        assert_eq!(out[0], "above");
        assert!(out[2].starts_with('─'), "{out:?}");
        assert_eq!(out[4], "below");
    }

    #[test]
    fn links_show_their_target() {
        assert_eq!(
            rendered("see [docs](http://x.test/a)", 60),
            vec!["see docs (http://x.test/a)"]
        );
    }

    #[test]
    fn fenced_code_is_kept_verbatim_and_set_apart() {
        let out = rendered(
            "before\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\nafter",
            40,
        );
        assert_eq!(
            out,
            vec![
                "before",
                "",
                "fn main() {",
                "    println!(\"hi\");",
                "}",
                "",
                "after"
            ]
        );
        let lines = render("```\ncode\n```", 40);
        assert_eq!(style_of(&lines, "code"), code_style());
    }

    #[test]
    fn a_code_line_wider_than_the_view_continues_on_the_next_row_without_loss() {
        let out = rendered("```\nabcdefghijklmnop\n```", 5);
        assert_eq!(out, vec!["abcde", "fghij", "klmno", "p"]);
        assert_eq!(out.concat(), "abcdefghijklmnop");
    }

    #[test]
    fn code_blocks_inside_lists_keep_the_indent() {
        let out = rendered("- item\n\n  ```\n  x = 1\n  ```\n", 40);
        assert_eq!(out, vec!["• item", "  x = 1"]);
    }

    #[test]
    fn tables_and_html_fall_back_to_source_text() {
        let out = rendered("| a | b |\n|---|---|\n| 1 | 2 |", 40);
        assert_eq!(out, vec!["| a | b |", "|---|---|", "| 1 | 2 |"]);
        assert_eq!(rendered("<b>x</b>", 40), vec!["<b>x</b>"]);
    }

    #[test]
    fn escape_sequences_never_survive() {
        let out = rendered("clear \u{1b}[2J this", 40);
        assert!(out.iter().all(|line| !line.contains('\u{1b}')), "{out:?}");
        assert_eq!(out, vec!["clear \u{241b}[2J this"]);
    }

    #[test]
    fn an_unclosed_fence_renders_as_code_and_does_not_hide_later_text() {
        let lines = render("```rust\nfn main() {", 40);
        assert_eq!(text(&lines), vec!["fn main() {"]);
        assert_eq!(style_of(&lines, "fn main"), code_style());
    }

    #[test]
    fn empty_input_renders_nothing() {
        assert!(render("", 40).is_empty());
    }

    #[test]
    fn rendering_every_prefix_never_panics() {
        let documents = [
            "# Heading\n\nSome *emphasis*, **strong**, `code` and a [link](http://a.b).\n\n- one\n- two\n  - nested\n\n1. a\n2. b\n\n> quote\n\n---\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\ndone",
            "plain text with no markup at all\nand a second line",
            "你好 **世界** 🎉 `код` — ünï",
            "```\nunclosed fence\nstill code",
            "* [ ] odd *list* \\* with _under_scores_ and **unclosed bold",
        ];
        for document in documents {
            let boundaries: Vec<usize> = document
                .char_indices()
                .map(|(i, _)| i)
                .chain([document.len()])
                .collect();
            for width in [1, 7, 20, 80] {
                for &end in &boundaries {
                    let _ = render(&document[..end], width);
                }
            }
        }
    }

    #[test]
    fn no_row_exceeds_the_width() {
        let doc = "# T\n\n- a long list item that must wrap around\n\n> a long quote that must wrap around\n\n```\nabcdefghijklmnopqrstuvwxyz\n```";
        for width in 4..40 {
            for line in render(doc, width) {
                assert!(row_width(&line.spans) <= width, "{width}: {line:?}");
            }
        }
    }
}
