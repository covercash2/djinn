//! Text that plays a particular role in a request.

use serde::{Deserialize, Serialize};

use super::ValueError;

/// The most stop sequences a request may carry.
pub const MAX_STOP_SEQUENCES: usize = 4;

const STOP_SEQUENCE_MAX_BYTES: usize = 256;

fn validate_stop_sequence(value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err("must not be empty (it would match before any text is produced)".to_owned());
    }
    if value.len() > STOP_SEQUENCE_MAX_BYTES {
        return Err(format!(
            "must be at most {STOP_SEQUENCE_MAX_BYTES} bytes, got {}",
            value.len()
        ));
    }
    Ok(())
}

fn validate_system_prompt(value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err("must contain at least one non-whitespace character".to_owned());
    }
    Ok(())
}

string_value! {
    /// A string that ends generation when the model would produce it: 1 to 256 bytes.
    StopSequence, "stop sequence", validate_stop_sequence,
    min_length = Some(1), max_length = Some(STOP_SEQUENCE_MAX_BYTES as u32),
    pattern = None
}

string_value! {
    /// Instructions that frame a conversation: at least one non-whitespace character.
    SystemPrompt, "system prompt", validate_system_prompt,
    min_length = Some(1), max_length = None,
    pattern = Some("\\S")
}

/// Text the model is asked to continue. Any text is valid, including the empty string; the type
/// exists so a prompt cannot be confused with generated output or other text.
#[derive(
    Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(transparent)]
#[schemars(transparent)]
pub struct Prompt(String);

impl Prompt {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for Prompt {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for Prompt {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<Prompt> for String {
    fn from(value: Prompt) -> String {
        value.0
    }
}

impl std::str::FromStr for Prompt {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(Self(value.to_owned()))
    }
}

impl std::fmt::Display for Prompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for Prompt {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<str> for Prompt {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for Prompt {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

/// At most [`MAX_STOP_SEQUENCES`] stop sequences.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StopSequences(Vec<StopSequence>);

impl StopSequences {
    /// What this value is called in error messages.
    pub const KIND: &'static str = "stop sequences";

    pub fn new(sequences: Vec<StopSequence>) -> Result<Self, ValueError> {
        if sequences.len() > MAX_STOP_SEQUENCES {
            return Err(ValueError::new(
                Self::KIND,
                format!(
                    "may contain at most {MAX_STOP_SEQUENCES} sequences, got {}",
                    sequences.len()
                ),
            ));
        }
        Ok(Self(sequences))
    }

    pub fn as_slice(&self) -> &[StopSequence] {
        &self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, StopSequence> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_vec(self) -> Vec<StopSequence> {
        self.0
    }
}

impl TryFrom<Vec<StopSequence>> for StopSequences {
    type Error = ValueError;

    fn try_from(value: Vec<StopSequence>) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'a> IntoIterator for &'a StopSequences {
    type Item = &'a StopSequence;
    type IntoIter = std::slice::Iter<'a, StopSequence>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Serialize for StopSequences {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StopSequences {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let sequences = Vec::<StopSequence>::deserialize(deserializer)?;
        Self::new(sequences).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for StopSequences {
    fn schema_name() -> String {
        "StopSequences".to_owned()
    }

    fn is_referenceable() -> bool {
        false
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        schemars::schema::SchemaObject {
            instance_type: Some(schemars::schema::InstanceType::Array.into()),
            array: Some(Box::new(schemars::schema::ArrayValidation {
                items: Some(generator.subschema_for::<StopSequence>().into()),
                max_items: Some(MAX_STOP_SEQUENCES as u32),
                ..Default::default()
            })),
            ..Default::default()
        }
        .into()
    }
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for StopSequences {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::schema::ArrayBuilder::new()
            .items(<StopSequence as utoipa::PartialSchema>::schema())
            .max_items(Some(MAX_STOP_SEQUENCES))
            .into()
    }
}

#[cfg(feature = "openapi")]
impl utoipa::ToSchema for StopSequences {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stop_sequence_is_rejected() {
        let error = StopSequence::new("").unwrap_err();
        assert!(error.to_string().contains("must not be empty"), "{error}");
    }

    #[test]
    fn stop_sequence_length_is_measured_in_bytes() {
        assert!(StopSequence::new("é".repeat(128)).is_ok());
        assert!(StopSequence::new("é".repeat(129)).is_err());
        assert!(StopSequence::new("END").is_ok());
    }

    #[test]
    fn four_stop_sequences_are_allowed_and_five_are_not() {
        let make = |n: usize| -> Vec<StopSequence> {
            (0..n)
                .map(|i| StopSequence::new(format!("s{i}")).unwrap())
                .collect()
        };
        assert!(StopSequences::new(make(0)).is_ok());
        assert!(StopSequences::new(make(4)).is_ok());
        let error = StopSequences::new(make(5)).unwrap_err();
        assert!(error.to_string().contains("at most 4"), "{error}");
    }

    #[test]
    fn stop_sequences_deserialize_with_validation() {
        let ok: StopSequences = serde_json::from_str(r#"["a","b"]"#).unwrap();
        assert_eq!(ok.len(), 2);
        assert!(serde_json::from_str::<StopSequences>(r#"["a",""]"#).is_err());
        assert!(serde_json::from_str::<StopSequences>(r#"["a","b","c","d","e"]"#).is_err());
        assert_eq!(serde_json::to_string(&ok).unwrap(), r#"["a","b"]"#);
    }

    #[test]
    fn blank_system_prompts_are_rejected() {
        assert!(SystemPrompt::new("  \n\t ").is_err());
        assert!(SystemPrompt::new("").is_err());
        assert!(SystemPrompt::new("You are helpful.").is_ok());
    }

    #[test]
    fn any_prompt_is_valid_including_empty() {
        let empty = Prompt::from("");
        assert_eq!(empty.as_str(), "");
        assert_eq!(serde_json::to_string(&empty).unwrap(), r#""""#);
        let back: Prompt = serde_json::from_str(r#""say hi""#).unwrap();
        assert_eq!(back, "say hi");
    }

    #[test]
    fn stop_sequences_schema_has_a_max_items_bound() {
        let schema = serde_json::to_value(schemars::schema_for!(StopSequences)).unwrap();
        assert_eq!(schema["maxItems"], 4);
        assert_eq!(schema["items"]["minLength"], 1);
    }
}
