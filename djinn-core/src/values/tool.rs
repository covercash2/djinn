//! Values that identify and feed tool calls.

use serde_json::{Map, Value};

use super::ValueError;

const TOOL_NAME_MAX: usize = 64;
const TOOL_CALL_ID_MAX: usize = 256;

fn validate_tool_name(value: &str) -> Result<(), String> {
    let length = value.chars().count();
    if length == 0 || length > TOOL_NAME_MAX {
        return Err(format!(
            "must be 1 to {TOOL_NAME_MAX} characters, got {length}; \
             only ASCII letters, digits, `_` and `-` are allowed"
        ));
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("only ASCII letters, digits, `_` and `-` are allowed".to_owned());
    }
    Ok(())
}

fn validate_tool_call_id(value: &str) -> Result<(), String> {
    let length = value.chars().count();
    if length == 0 || length > TOOL_CALL_ID_MAX {
        return Err(format!(
            "must be 1 to {TOOL_CALL_ID_MAX} characters, got {length}"
        ));
    }
    if value.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("must not contain whitespace or control characters".to_owned());
    }
    Ok(())
}

string_value! {
    /// The name of a tool the model can call: 1 to 64 ASCII letters, digits, `_` or `-`.
    ToolName, "tool name", validate_tool_name,
    min_length = Some(1), max_length = Some(TOOL_NAME_MAX as u32),
    pattern = Some("^[A-Za-z0-9_-]{1,64}$")
}

string_value! {
    /// Identifies one tool call so its result can be matched to it: 1 to 256 characters without
    /// whitespace or control characters.
    ToolCallId, "tool call id", validate_tool_call_id,
    min_length = Some(1), max_length = Some(TOOL_CALL_ID_MAX as u32),
    pattern = Some("^[^\\s\\x00-\\x1f\\x7f]{1,256}$")
}

/// The arguments of a tool call: text holding a JSON object, which is how the OpenAI protocol
/// carries them.
///
/// The exact text given is kept (it is what gets replayed to the model, so prompts stay
/// byte-stable) next to the parsed object. Empty or whitespace-only text is accepted and means an
/// empty object, because clients commonly send `""` for a call without arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolArguments {
    raw: String,
    object: Map<String, Value>,
}

impl ToolArguments {
    /// What this value is called in error messages.
    pub const KIND: &'static str = "tool arguments";

    /// Parses `raw`, which must be a JSON object (or empty).
    pub fn new(raw: impl Into<String>) -> Result<Self, ValueError> {
        let raw = raw.into();
        let object = if raw.trim().is_empty() {
            Map::new()
        } else {
            match serde_json::from_str::<Value>(&raw) {
                Ok(Value::Object(object)) => object,
                Ok(_) => {
                    return Err(
                        ValueError::new(Self::KIND, "must be a JSON object").with_value(&raw)
                    )
                }
                Err(error) => {
                    return Err(
                        ValueError::new(Self::KIND, format!("must be valid JSON: {error}"))
                            .with_value(&raw),
                    )
                }
            }
        };
        Ok(Self { raw, object })
    }

    /// For literals and tests only: panics if `raw` is invalid.
    #[track_caller]
    pub fn from_static(raw: &'static str) -> Self {
        match Self::new(raw) {
            Ok(arguments) => arguments,
            Err(error) => panic!("invalid ToolArguments: {error}"),
        }
    }

    /// Arguments from an already parsed object, written compactly.
    pub fn from_object(object: Map<String, Value>) -> Self {
        let raw = Value::Object(object.clone()).to_string();
        Self { raw, object }
    }

    /// Arguments with no entries (`{}`).
    pub fn empty() -> Self {
        Self::from_object(Map::new())
    }

    /// The exact text these arguments were created from.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// The parsed object.
    pub fn object(&self) -> &Map<String, Value> {
        &self.object
    }

    /// The parsed object as a JSON value, for handing to a tool.
    pub fn to_value(&self) -> Value {
        Value::Object(self.object.clone())
    }

    pub fn into_string(self) -> String {
        self.raw
    }
}

impl std::fmt::Display for ToolArguments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.raw)
    }
}

impl std::str::FromStr for ToolArguments {
    type Err = ValueError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<String> for ToolArguments {
    type Error = ValueError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ToolArguments> for String {
    fn from(value: ToolArguments) -> String {
        value.raw
    }
}

impl serde::Serialize for ToolArguments {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.raw)
    }
}

impl<'de> serde::Deserialize<'de> for ToolArguments {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = <String as serde::Deserialize>::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for ToolArguments {
    fn schema_name() -> String {
        "ToolArguments".to_owned()
    }

    fn is_referenceable() -> bool {
        false
    }

    fn json_schema(_: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        schemars::schema::SchemaObject {
            instance_type: Some(schemars::schema::InstanceType::String.into()),
            metadata: Some(Box::new(schemars::schema::Metadata {
                title: Some(Self::KIND.to_owned()),
                description: Some(
                    "A JSON object encoded as a string (empty means no arguments).".to_owned(),
                ),
                ..Default::default()
            })),
            ..Default::default()
        }
        .into()
    }
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for ToolArguments {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::ObjectBuilder::new()
            .schema_type(utoipa::openapi::schema::Type::String)
            .title(Some(Self::KIND))
            .description(Some(
                "A JSON object encoded as a string (empty means no arguments).",
            ))
            .into()
    }
}

#[cfg(feature = "openapi")]
impl utoipa::ToSchema for ToolArguments {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_tool_names_are_accepted() {
        for name in ["calculator", "read_file", "get-time", "A1", &"a".repeat(64)] {
            assert!(ToolName::new(name).is_ok(), "{name}");
        }
    }

    #[test]
    fn invalid_tool_names_are_rejected_with_the_rule() {
        for name in ["read file", "", "a.b", "naïve", &"a".repeat(65)] {
            let error = ToolName::new(name).unwrap_err();
            assert!(error.to_string().contains("tool name"), "{error}");
            assert!(error.to_string().contains("ASCII letters"), "{error}");
        }
    }

    #[test]
    fn tool_call_ids_are_non_empty_and_have_no_whitespace() {
        assert!(ToolCallId::new("call_0").is_ok());
        assert!(ToolCallId::new("toolu_01A").is_ok());
        assert!(ToolCallId::new("").is_err());
        assert!(ToolCallId::new("call 0").is_err());
        assert!(ToolCallId::new("call\n0").is_err());
        assert!(ToolCallId::new("x".repeat(257)).is_err());
    }

    #[test]
    fn arguments_must_be_a_json_object() {
        for ok in [r#"{"expression":"2+2"}"#, "{}", "", "   "] {
            assert!(ToolArguments::new(ok).is_ok(), "{ok:?}");
        }
        for bad in ["[1]", r#""x""#, r#"{"a":"#, "3", "null"] {
            assert!(ToolArguments::new(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn empty_arguments_mean_an_empty_object() {
        let arguments = ToolArguments::new("").unwrap();
        assert!(arguments.object().is_empty());
        assert_eq!(arguments.to_value(), serde_json::json!({}));
    }

    #[test]
    fn arguments_keep_their_exact_text() {
        let text = "{ \"a\" :   1,\n \"b\":[1, 2] }";
        let arguments = ToolArguments::new(text).unwrap();
        assert_eq!(arguments.as_str(), text);
        assert_eq!(
            serde_json::to_string(&arguments).unwrap(),
            format!("{text:?}")
        );
        assert_eq!(arguments.object()["a"], 1);
    }

    #[test]
    fn from_object_writes_compact_json() {
        let mut object = Map::new();
        object.insert("expression".to_owned(), Value::from("1+1"));
        let arguments = ToolArguments::from_object(object);
        assert_eq!(arguments.as_str(), r#"{"expression":"1+1"}"#);
        assert_eq!(ToolArguments::empty().as_str(), "{}");
    }

    #[test]
    fn deserialization_validates() {
        assert!(serde_json::from_str::<ToolName>(r#""read file""#).is_err());
        assert!(serde_json::from_str::<ToolArguments>(r#""[1]""#).is_err());
        let error = serde_json::from_str::<ToolName>(r#""bad name""#).unwrap_err();
        assert!(error.to_string().contains("tool name"), "{error}");
    }

    #[test]
    fn serialized_form_is_the_bare_string() {
        let name = ToolName::from_static("calculator");
        assert_eq!(serde_json::to_string(&name).unwrap(), r#""calculator""#);
        let back: ToolName = serde_json::from_str(r#""calculator""#).unwrap();
        assert_eq!(back, name);
    }

    #[test]
    #[should_panic(expected = "invalid ToolName")]
    fn from_static_panics_on_an_invalid_literal() {
        let _ = ToolName::from_static("not valid");
    }

    #[test]
    fn schema_carries_the_constraints() {
        let schema = schemars::schema_for!(ToolName);
        let value = serde_json::to_value(&schema).unwrap();
        assert_eq!(value["pattern"], "^[A-Za-z0-9_-]{1,64}$");
        assert_eq!(value["minLength"], 1);
        assert_eq!(value["maxLength"], 64);
    }
}
