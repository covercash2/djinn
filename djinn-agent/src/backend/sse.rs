//! A Server-Sent Events parser that works on arbitrary chunks of bytes.
//!
//! Network reads can split an event anywhere: inside a line, between a field name and its value, or
//! in the middle of a multi-byte UTF-8 character. The parser buffers raw bytes and only decodes
//! complete lines, so none of that matters.

/// One dispatched event.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SseEvent {
    /// The `event:` field, if the server sent one.
    pub event: Option<String>,
    /// The `data:` lines, joined with `\n`.
    pub data: String,
}

/// Incremental SSE parser. Feed it bytes with [`push`](Self::push); call [`finish`](Self::finish)
/// when the connection closes.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
    has_fields: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consumes a chunk and returns the events it completed.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=newline).collect();
            line.pop(); // the '\n'
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if let Some(event) = self.line(&String::from_utf8_lossy(&line)) {
                events.push(event);
            }
        }
        events
    }

    /// Ends the stream, returning an event that was still waiting for its blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            let line = String::from_utf8_lossy(&line).into_owned();
            let line = line.strip_suffix('\r').unwrap_or(&line).to_owned();
            if let Some(event) = self.line(&line) {
                return Some(event);
            }
        }
        self.dispatch()
    }

    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return None; // comment / keep-alive
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "data" => {
                self.data.push(value.to_owned());
                self.has_fields = true;
            }
            "event" => {
                self.event = Some(value.to_owned());
                self.has_fields = true;
            }
            _ => {} // id, retry and unknown fields are not used
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        if !self.has_fields {
            return None;
        }
        self.has_fields = false;
        let event = SseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
        };
        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut parser = SseParser::new();
        let mut events: Vec<SseEvent> = chunks.iter().flat_map(|c| parser.push(c)).collect();
        events.extend(parser.finish());
        events
    }

    fn data(text: &str) -> SseEvent {
        SseEvent {
            event: None,
            data: text.to_owned(),
        }
    }

    #[test]
    fn a_single_event() {
        assert_eq!(parse_all(&[b"data: hello\n\n"]), vec![data("hello")]);
    }

    #[test]
    fn several_events_in_one_chunk() {
        assert_eq!(
            parse_all(&[b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"]),
            vec![data("one"), data("two"), data("[DONE]")]
        );
    }

    #[test]
    fn crlf_line_endings() {
        assert_eq!(
            parse_all(&[b"data: hi\r\n\r\ndata: there\r\n\r\n"]),
            vec![data("hi"), data("there")]
        );
    }

    #[test]
    fn an_event_split_across_chunks() {
        assert_eq!(
            parse_all(&[b"da", b"ta: wor", b"ld\n", b"\n"]),
            vec![data("world")]
        );
    }

    #[test]
    fn the_blank_line_in_a_separate_chunk() {
        assert_eq!(parse_all(&[b"data: x\n", b"\n"]), vec![data("x")]);
    }

    #[test]
    fn every_split_point_gives_the_same_events() {
        let text = "data: {\"a\":1}\n\n: keep-alive\n\nevent: ping\ndata: p\n\ndata: [DONE]\n\n"
            .as_bytes();
        let expected = parse_all(&[text]);
        for cut in 0..=text.len() {
            assert_eq!(
                parse_all(&[&text[..cut], &text[cut..]]),
                expected,
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn a_multibyte_character_split_across_chunks() {
        // "é" is 0xC3 0xA9, "☃" is 0xE2 0x98 0x83.
        let bytes = "data: \u{e9}t\u{e9} \u{2603}\n\n".as_bytes();
        for cut in 0..=bytes.len() {
            assert_eq!(
                parse_all(&[&bytes[..cut], &bytes[cut..]]),
                vec![data("\u{e9}t\u{e9} \u{2603}")],
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn comments_and_unknown_fields_are_ignored() {
        assert_eq!(
            parse_all(&[b": keep-alive\n\nid: 7\nretry: 100\ndata: x\n\n"]),
            vec![data("x")]
        );
    }

    #[test]
    fn event_names_are_reported() {
        assert_eq!(
            parse_all(&[b"event: error\ndata: boom\n\n"]),
            vec![SseEvent {
                event: Some("error".into()),
                data: "boom".into()
            }]
        );
    }

    #[test]
    fn multiple_data_lines_are_joined_with_newlines() {
        assert_eq!(
            parse_all(&[b"data: a\ndata: b\ndata: c\n\n"]),
            vec![data("a\nb\nc")]
        );
    }

    #[test]
    fn only_one_leading_space_is_removed() {
        assert_eq!(parse_all(&[b"data:  two\n\n"]), vec![data(" two")]);
        assert_eq!(parse_all(&[b"data:none\n\n"]), vec![data("none")]);
    }

    #[test]
    fn a_data_line_without_a_colon_is_an_empty_data_field() {
        assert_eq!(parse_all(&[b"data\n\n"]), vec![data("")]);
    }

    #[test]
    fn a_missing_trailing_blank_line_still_yields_the_last_event() {
        assert_eq!(
            parse_all(&[b"data: one\n\ndata: last"]),
            vec![data("one"), data("last")]
        );
        assert_eq!(parse_all(&[b"data: last\n"]), vec![data("last")]);
    }

    #[test]
    fn blank_lines_between_events_dispatch_nothing_extra() {
        assert_eq!(parse_all(&[b"\n\n\ndata: x\n\n\n\n"]), vec![data("x")]);
    }

    #[test]
    fn an_empty_stream_has_no_events() {
        assert!(parse_all(&[]).is_empty());
        assert!(parse_all(&[b""]).is_empty());
    }

    #[test]
    fn invalid_utf8_does_not_panic() {
        let events = parse_all(&[b"data: \xff\xfe\n\n"]);
        assert_eq!(events.len(), 1);
    }
}
