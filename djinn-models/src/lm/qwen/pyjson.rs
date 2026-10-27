//! JSON text formatted like Python's `json.dumps(..., ensure_ascii=False)`.
//!
//! Qwen's chat template renders tool specs and tool-call arguments with the Jinja `tojson` filter,
//! which Hugging Face implements as exactly that call. Its default separators put a space after
//! every `,` and `:`, unlike `serde_json::to_string`. The model was trained on that spelling, so
//! the prompt must match it.

use serde::Serialize;
use serde_json::ser::{Formatter, Serializer};

struct PyFormatter;

impl Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

/// Serializes `value` the way Python's `json.dumps` would (key order preserved, non-ASCII kept).
pub(crate) fn to_py_json<T: Serialize + ?Sized>(value: &T) -> String {
    let mut buffer = Vec::new();
    let mut serializer = Serializer::with_formatter(&mut buffer, PyFormatter);
    // Serializing a `serde_json::Value` or a plain derived struct into memory cannot fail.
    value
        .serialize(&mut serializer)
        .expect("in-memory JSON serialization");
    String::from_utf8(buffer).expect("serde_json emits UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn objects_and_arrays_use_python_separators() {
        assert_eq!(
            to_py_json(&json!({"a": 1, "b": [1, 2, {"c": null}]})),
            r#"{"a": 1, "b": [1, 2, {"c": null}]}"#
        );
    }

    #[test]
    fn empty_containers_have_no_padding() {
        assert_eq!(to_py_json(&json!({})), "{}");
        assert_eq!(to_py_json(&json!([])), "[]");
        assert_eq!(
            to_py_json(&json!({"a": {}, "b": []})),
            r#"{"a": {}, "b": []}"#
        );
    }

    #[test]
    fn key_order_is_preserved() {
        let value: serde_json::Value = serde_json::from_str(r#"{"z": 1, "a": 2, "m": 3}"#).unwrap();
        assert_eq!(to_py_json(&value), r#"{"z": 1, "a": 2, "m": 3}"#);
    }

    #[test]
    fn non_ascii_is_not_escaped_but_controls_are() {
        assert_eq!(
            to_py_json(&json!("\u{e9}t\u{e9} \u{2603}")),
            "\"\u{e9}t\u{e9} \u{2603}\""
        );
        assert_eq!(to_py_json(&json!("a\nb\"c")), r#""a\nb\"c""#);
    }

    #[test]
    fn scalars_match_python() {
        assert_eq!(to_py_json(&json!(true)), "true");
        assert_eq!(to_py_json(&json!(2.5)), "2.5");
        assert_eq!(to_py_json(&json!(3.0)), "3.0");
        assert_eq!(to_py_json(&json!("x")), "\"x\"");
    }

    #[test]
    fn plain_strings_serialize_as_json_strings() {
        assert_eq!(to_py_json("not json"), "\"not json\"");
    }
}
