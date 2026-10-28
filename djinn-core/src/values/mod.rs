//! Validated domain values.
//!
//! Identifiers, names, paths, secrets and bounded parameters are not plain strings or numbers in
//! djinn: each kind is its own type that **cannot hold an invalid value**. A value is checked when
//! it is constructed and when it is deserialized (an HTTP body, a TOML config, a command-line
//! argument, another server's response), so bad input is rejected where it enters the program and
//! the rest of the code can rely on the rules. Serialized forms are the bare string or number, so
//! nothing on the wire or on disk changes; the constraints are published in the generated JSON
//! Schema and OpenAPI document.
//!
//! | Type | Rule |
//! |---|---|
//! | [`ToolName`] | 1–64 of ASCII letters, digits, `_`, `-` |
//! | [`ToolCallId`] | 1–256 characters, no whitespace or control characters |
//! | [`ToolArguments`] | text holding a JSON object (empty text means `{}`); keeps the exact text |
//! | [`ModelId`] | 1–256 characters, no control characters, no surrounding whitespace |
//! | [`RepoId`] | `owner/name`, each 1–96 of ASCII letters, digits, `.`, `_`, `-` (not `.`/`..`) |
//! | [`Revision`] | 1–128 characters, no whitespace or control characters |
//! | [`WeightsFile`] | relative `/`-separated path, no empty, `.` or `..` segment |
//! | [`ConfigName`] | relative `/`-separated name of `[A-Za-z0-9._-]` segments, no `.toml` suffix |
//! | [`Prompt`] | any text, including empty |
//! | [`SystemPrompt`] | at least one non-whitespace character |
//! | [`StopSequence`] / [`StopSequences`] | 1–256 bytes each; at most [`MAX_STOP_SEQUENCES`] |
//! | [`Temperature`] | finite, 0 to 2 |
//! | [`TopP`] | finite, above 0 and at most 1 |
//! | [`MaxTokens`] | whole number, at least 1 |
//!
//! These stay plain `String`s on purpose: message content, tool output, generated text, error
//! messages, the ignored `model` field of a chat request, and the values inside a tool's
//! arguments (each tool validates its own parameters).
//!
//! There is deliberately no infallible `From<&str>` on a type whose conversion can fail. In
//! tests and for literals use `T::from_static("...")`, which panics with a clear message.

mod error;
#[macro_use]
mod macros;
mod model;
mod sampling;
mod text;
mod tool;

pub use error::ValueError;
pub use model::{ConfigName, ModelId, RepoId, Revision, WeightsFile};
pub use sampling::{MaxTokens, Temperature, TopP};
pub use text::{MAX_STOP_SEQUENCES, Prompt, StopSequence, StopSequences, SystemPrompt};
pub use tool::{ToolArguments, ToolCallId, ToolName};

#[cfg(test)]
mod tests {
    //! Tests of what the macros generate, using throwaway types.

    use std::borrow::Borrow;

    fn lowercase(value: &str) -> Result<(), String> {
        if value.chars().all(|c| c.is_ascii_lowercase()) && !value.is_empty() {
            Ok(())
        } else {
            Err("must be lowercase letters".to_owned())
        }
    }

    string_value! {
        /// A throwaway type.
        Word, "word", lowercase,
        min_length = Some(1), max_length = None, pattern = Some("^[a-z]+$")
    }

    #[test]
    fn construction_checks_the_rule() {
        assert_eq!(Word::new("abc").unwrap().as_str(), "abc");
        let error = Word::new("ABC").unwrap_err();
        assert_eq!(error.to_string(), "word `ABC`: must be lowercase letters");
        assert_eq!((error.kind(), Word::KIND), ("word", "word"));
    }

    #[test]
    fn every_conversion_goes_through_the_check() {
        assert!("abc".parse::<Word>().is_ok());
        assert!("ABC".parse::<Word>().is_err());
        assert!(Word::try_from("abc").is_ok());
        assert!(Word::try_from(String::from("A")).is_err());
        let word = Word::from_static("abc");
        let as_str: &str = word.as_ref();
        let borrowed: &str = word.borrow();
        assert_eq!(
            (as_str, borrowed, word.to_string()),
            ("abc", "abc", "abc".to_owned())
        );
        assert_eq!(String::from(word.clone()), "abc");
        assert_eq!(word.into_string(), "abc");
    }

    #[test]
    fn serde_round_trips_the_bare_string_and_rejects_bad_input() {
        let word = Word::from_static("abc");
        assert_eq!(serde_json::to_string(&word).unwrap(), r#""abc""#);
        assert_eq!(serde_json::from_str::<Word>(r#""abc""#).unwrap(), word);
        let error = serde_json::from_str::<Word>(r#""NO""#).unwrap_err();
        assert!(error.to_string().contains("word `NO`"), "{error}");
        assert!(serde_json::from_str::<Word>("5").is_err());
    }

    #[test]
    fn echoed_values_are_truncated() {
        let long = "A".repeat(200);
        let message = Word::new(long.clone()).unwrap_err().to_string();
        assert!(message.len() < long.len());
    }

    #[test]
    fn schema_is_inline_with_the_hints() {
        let schema = serde_json::to_value(schemars::schema_for!(Word)).unwrap();
        assert_eq!(schema["type"], "string");
        assert_eq!(schema["pattern"], "^[a-z]+$");
        assert_eq!(schema["minLength"], 1);
        assert!(schema.get("maxLength").is_none());
    }
}
