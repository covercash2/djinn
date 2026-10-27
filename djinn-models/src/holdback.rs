//! Streaming trigger detection with hold-back.
//!
//! Generation yields text in small fragments, and a trigger string (a caller-supplied stop
//! sequence, or the `<tool_call>` marker) can be split across fragments. [`Holdback`] sits between
//! the generator and the consumer: it releases text only once it can no longer be the start of a
//! trigger, and reports when a trigger is complete.
//!
//! The result must not depend on how the text happened to be chunked; the tests below feed the same
//! text in every possible chunking and check that the output is identical.

/// A trigger that was found in the stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Hit {
    /// Index of the matched trigger in the list given to [`Holdback::new`].
    pub trigger: usize,
    /// The matched trigger and everything received after its start, so callers that need the
    /// text following the trigger (a tool call body) can keep it. Callers that treat the trigger
    /// as a stop sequence simply discard it.
    pub rest: String,
}

/// What [`Holdback::push`] returns.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Pushed {
    /// Text that is safe to hand to the consumer now.
    pub emit: String,
    /// Set when a trigger has been found; the matcher is then finished.
    pub hit: Option<Hit>,
}

/// Detects triggers in a fragmented text stream. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct Holdback {
    triggers: Vec<String>,
    held: String,
    done: bool,
}

impl Holdback {
    /// Creates a matcher for the given trigger strings.
    ///
    /// Matching is by arrival: a trigger fires as soon as it is complete, so if one trigger is a
    /// prefix of another (or starts inside another), which one is reported can depend on how the
    /// text was chunked. Within a single buffer the earliest, then the longest, trigger wins. Triggers
    /// that do not overlap each other behave identically for every chunking. Empty strings are ignored (they would
    /// match everywhere), but keep their index, so `Hit::trigger` always indexes the input list.
    pub fn new<I, S>(triggers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            triggers: triggers.into_iter().map(Into::into).collect(),
            held: String::new(),
            done: false,
        }
    }

    /// Whether a trigger has already been found.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Feeds the next fragment of text.
    ///
    /// Once a trigger has been found the matcher is finished and later calls return nothing: the
    /// caller owns the rest of the stream from then on.
    pub fn push(&mut self, fragment: &str) -> Pushed {
        if self.done {
            return Pushed::default();
        }

        let mut buffer = std::mem::take(&mut self.held);
        buffer.push_str(fragment);

        if let Some((position, trigger)) = self.earliest_match(&buffer) {
            self.done = true;
            return Pushed {
                emit: buffer[..position].to_owned(),
                hit: Some(Hit {
                    trigger,
                    rest: buffer[position..].to_owned(),
                }),
            };
        }

        let keep = self.longest_pending_prefix(&buffer);
        let split = buffer.len() - keep;
        self.held = buffer[split..].to_owned();
        buffer.truncate(split);
        Pushed {
            emit: buffer,
            hit: None,
        }
    }

    /// Ends the stream without a trigger, releasing any text that was being held back.
    pub fn finish(&mut self) -> String {
        if self.done {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }

    /// The earliest trigger in `buffer`; when several start at the same place, the longest wins.
    fn earliest_match(&self, buffer: &str) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (index, trigger) in self.triggers.iter().enumerate() {
            if trigger.is_empty() {
                continue;
            }
            let Some(position) = buffer.find(trigger.as_str()) else {
                continue;
            };
            let better = match best {
                None => true,
                Some((best_position, best_index)) => {
                    position < best_position
                        || (position == best_position
                            && trigger.len() > self.triggers[best_index].len())
                }
            };
            if better {
                best = Some((position, index));
            }
        }
        best
    }

    /// Length in bytes of the longest suffix of `buffer` that is a proper prefix of a trigger.
    fn longest_pending_prefix(&self, buffer: &str) -> usize {
        self.triggers
            .iter()
            .filter(|trigger| !trigger.is_empty())
            .map(|trigger| {
                (1..trigger.len())
                    .rev()
                    .filter(|&len| trigger.is_char_boundary(len))
                    .find(|&len| buffer.ends_with(&trigger[..len]))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `chunks` through a fresh matcher the way a real caller does: after a hit the caller owns
    /// the stream and appends every remaining chunk to `rest`. Returns all emitted text (including
    /// the final flush when no trigger was found) and the hit, if any.
    fn run(triggers: &[&str], chunks: &[&str]) -> (String, Option<Hit>) {
        let mut matcher = Holdback::new(triggers.iter().copied());
        let mut emitted = String::new();
        let mut hit: Option<Hit> = None;
        for chunk in chunks {
            if let Some(hit) = hit.as_mut() {
                hit.rest.push_str(chunk);
                continue;
            }
            let pushed = matcher.push(chunk);
            emitted.push_str(&pushed.emit);
            hit = pushed.hit;
        }
        if hit.is_none() {
            emitted.push_str(&matcher.finish());
        }
        (emitted, hit)
    }

    /// All ways to cut `text` into 1, 2, or 3 pieces at character boundaries.
    fn chunkings(text: &str) -> Vec<Vec<&str>> {
        let cuts: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
            .collect();
        let mut all = vec![vec![text]];
        for &a in &cuts {
            all.push(vec![&text[..a], &text[a..]]);
            for &b in cuts.iter().filter(|&&b| b >= a) {
                all.push(vec![&text[..a], &text[a..b], &text[b..]]);
            }
        }
        all
    }

    fn assert_chunking_invariant(triggers: &[&str], text: &str) {
        let expected = run(triggers, &[text]);
        for chunks in chunkings(text) {
            assert_eq!(
                run(triggers, &chunks),
                expected,
                "chunking {chunks:?} of {text:?} differs from the unsplit result"
            );
        }
    }

    #[test]
    fn text_without_triggers_passes_through() {
        assert_eq!(
            run(&["END"], &["hello ", "world"]),
            ("hello world".into(), None)
        );
    }

    #[test]
    fn trigger_in_the_middle_splits_text_and_rest() {
        let (emitted, hit) = run(&["END"], &["hello END world"]);
        assert_eq!(emitted, "hello ");
        assert_eq!(
            hit,
            Some(Hit {
                trigger: 0,
                rest: "END world".into()
            })
        );
    }

    #[test]
    fn trigger_at_start_emits_nothing() {
        let (emitted, hit) = run(&["END"], &["END tail"]);
        assert_eq!(emitted, "");
        assert_eq!(hit.unwrap().rest, "END tail");
    }

    #[test]
    fn trigger_at_end_has_only_itself_as_rest() {
        let (emitted, hit) = run(&["END"], &["done END"]);
        assert_eq!(emitted, "done ");
        assert_eq!(hit.unwrap().rest, "END");
    }

    #[test]
    fn trigger_split_across_fragments_is_detected() {
        let (emitted, hit) = run(&["END"], &["ab", "E", "N", "D", "cd"]);
        assert_eq!(emitted, "ab");
        assert_eq!(hit.unwrap().rest, "ENDcd");
    }

    #[test]
    fn partial_trigger_is_held_back_until_resolved() {
        let mut matcher = Holdback::new(["END"]);
        let first = matcher.push("ab E");
        assert_eq!(first.emit, "ab ");
        assert!(first.hit.is_none());
        // The held "E" turns out not to start the trigger, so it is released.
        let second = matcher.push("x");
        assert_eq!(second.emit, "Ex");
        assert!(second.hit.is_none());
    }

    #[test]
    fn held_text_is_flushed_on_finish() {
        let mut matcher = Holdback::new(["END"]);
        assert_eq!(matcher.push("abEN").emit, "ab");
        assert_eq!(matcher.finish(), "EN");
        assert_eq!(matcher.finish(), "", "flushing twice yields nothing");
    }

    #[test]
    fn earliest_trigger_wins() {
        let (emitted, hit) = run(&["bb", "aa"], &["xaaybb"]);
        assert_eq!(emitted, "x");
        assert_eq!(hit.unwrap().trigger, 1);
    }

    #[test]
    fn longest_trigger_wins_at_the_same_position() {
        let (_, hit) = run(&["ab", "abc"], &["zabc"]);
        assert_eq!(hit.unwrap().trigger, 1);
    }

    #[test]
    fn multiple_triggers_report_their_index() {
        let (emitted, hit) = run(&["<tool_call>", "STOP"], &["thinking... <tool_call>{}"]);
        assert_eq!(emitted, "thinking... ");
        let hit = hit.unwrap();
        assert_eq!(hit.trigger, 0);
        assert_eq!(hit.rest, "<tool_call>{}");
    }

    #[test]
    fn empty_triggers_are_ignored_but_keep_their_index() {
        let (emitted, hit) = run(&["", "x"], &["abxc"]);
        assert_eq!(emitted, "ab");
        assert_eq!(hit.unwrap().trigger, 1);
    }

    #[test]
    fn no_triggers_at_all_is_a_passthrough() {
        let mut matcher = Holdback::new(Vec::<String>::new());
        assert_eq!(matcher.push("anything").emit, "anything");
        assert_eq!(matcher.finish(), "");
    }

    #[test]
    fn pushes_after_a_hit_are_ignored() {
        let mut matcher = Holdback::new(["END"]);
        assert!(matcher.push("END").hit.is_some());
        assert!(matcher.is_done());
        assert_eq!(matcher.push("more"), Pushed::default());
        assert_eq!(matcher.finish(), "");
    }

    #[test]
    fn multibyte_text_is_never_split_inside_a_character() {
        // The held-back prefix and the emitted text must both stay valid UTF-8.
        let (emitted, hit) = run(&["\u{2603}STOP"], &["\u{e9}t\u{e9} \u{2603}", "STOP tail"]);
        assert_eq!(emitted, "\u{e9}t\u{e9} ");
        assert_eq!(hit.unwrap().rest, "\u{2603}STOP tail");
        assert_chunking_invariant(&["\u{2603}STOP"], "\u{e9}t\u{e9} \u{2603}STOP \u{1f600}");
    }

    #[test]
    fn every_chunking_gives_the_same_result() {
        let cases: &[(&[&str], &str)] = &[
            (&["END"], "hello END world"),
            (&["END"], "hello world"),
            (&["END"], "hello EN"),
            (&["END"], "ENEND"),
            (
                &["<tool_call>"],
                "Sure.\n<tool_call>\n{\"name\": \"x\"}\n</tool_call>",
            ),
            (&["<tool_call>"], "a < b and <tool_c"),
            (&["aa"], "aaa"),
            (&["abab"], "abababab"),
            (&["bb", "aa"], "xaaybb"),
            (&["STOP", "<tool_call>"], "x<tool_cSTOP"),
        ];
        for (triggers, text) in cases {
            assert_chunking_invariant(triggers, text);
        }
    }
}
