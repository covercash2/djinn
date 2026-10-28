use std::fmt;

/// How many characters of a rejected value are echoed back in an error message.
const ECHO_LIMIT: usize = 64;

/// Why a value was rejected: which kind of value, which rule, and (for values that are safe to
/// show) the offending text, truncated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueError {
    kind: &'static str,
    reason: String,
    value: Option<String>,
}

impl ValueError {
    /// An error for a value of `kind` (for example `"tool name"` or `"temperature"`) that broke
    /// `reason`. Phrase `reason` as the rule, not as the failure: `"must not be empty"`.
    pub fn new(kind: &'static str, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
            value: None,
        }
    }

    /// Shows `value` (truncated) in the message. Do not use this for secrets.
    pub fn with_value(mut self, value: &str) -> Self {
        let mut shown: String = value.chars().take(ECHO_LIMIT).collect();
        if value.chars().count() > ECHO_LIMIT {
            shown.push('…');
        }
        self.value = Some(shown);
        self
    }

    /// The kind of value that was rejected.
    pub fn kind(&self) -> &'static str {
        self.kind
    }

    /// The rule that was broken.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            Some(value) => write!(f, "{} `{}`: {}", self.kind, value, self.reason),
            None => write!(f, "{}: {}", self.kind, self.reason),
        }
    }
}

impl std::error::Error for ValueError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_names_kind_value_and_rule() {
        let error = ValueError::new("tool name", "must not be empty").with_value("a b");
        assert_eq!(error.to_string(), "tool name `a b`: must not be empty");
    }

    #[test]
    fn message_without_a_value_omits_it() {
        let error = ValueError::new("API key", "must not be empty");
        assert_eq!(error.to_string(), "API key: must not be empty");
    }

    #[test]
    fn long_values_are_truncated() {
        let long = "x".repeat(200);
        let shown = ValueError::new("k", "r").with_value(&long).to_string();
        assert!(shown.contains(&"x".repeat(64)));
        assert!(!shown.contains(&"x".repeat(65)));
        assert!(shown.contains('…'));
    }

    #[test]
    fn accessors_expose_the_parts() {
        let error = ValueError::new("k", "r");
        assert_eq!((error.kind(), error.reason()), ("k", "r"));
    }
}
