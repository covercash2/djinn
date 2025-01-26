use std::sync::Arc;

/// A line of text meant to be shown in the TUI interface
pub struct TextLine {
    text: Arc<str>,
    /// The level of severity associated with the entry
    level: tracing::Level,
}
