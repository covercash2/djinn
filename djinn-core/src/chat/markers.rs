//! Fields of the protocol whose value is always one fixed string (`"object": "chat.completion"`).
//!
//! As marker types they serialize to the literal, and on deserialization accept only that literal
//! (a wrong value is an error). The containing field is `#[serde(default)]`, so a server that
//! leaves `object` out is still accepted.

macro_rules! object_marker {
    ($(#[$meta:meta])* $name:ident, $literal:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name;

        impl $name {
            /// The literal this marker stands for.
            pub const VALUE: &'static str = $literal;
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str($literal)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = <String as serde::Deserialize>::deserialize(deserializer)?;
                if value == $literal {
                    Ok(Self)
                } else {
                    Err(serde::de::Error::custom(format!(
                        "object: expected `{}`, got `{value}`",
                        $literal
                    )))
                }
            }
        }

        impl schemars::JsonSchema for $name {
            fn schema_name() -> String {
                stringify!($name).to_owned()
            }

            fn is_referenceable() -> bool {
                false
            }

            fn json_schema(_: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
                schemars::schema::SchemaObject {
                    instance_type: Some(schemars::schema::InstanceType::String.into()),
                    const_value: Some(serde_json::Value::from($literal)),
                    ..Default::default()
                }
                .into()
            }
        }

        #[cfg(feature = "openapi")]
        impl utoipa::PartialSchema for $name {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
                utoipa::openapi::ObjectBuilder::new()
                    .schema_type(utoipa::openapi::schema::Type::String)
                    .enum_values(Some([$literal]))
                    .into()
            }
        }

        #[cfg(feature = "openapi")]
        impl utoipa::ToSchema for $name {}
    };
}

object_marker!(
    /// `"chat.completion"`, the `object` of a complete response.
    ChatCompletionObject,
    "chat.completion"
);
object_marker!(
    /// `"chat.completion.chunk"`, the `object` of a streamed chunk.
    ChatCompletionChunkObject,
    "chat.completion.chunk"
);
object_marker!(
    /// `"list"`, the `object` of a list response.
    ListObject,
    "list"
);
object_marker!(
    /// `"model"`, the `object` of a model entry.
    ModelObject,
    "model"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marker_serializes_to_its_literal() {
        assert_eq!(
            serde_json::to_string(&ChatCompletionObject).unwrap(),
            r#""chat.completion""#
        );
        assert_eq!(ModelObject::VALUE, "model");
    }

    #[test]
    fn only_the_exact_literal_is_accepted() {
        assert!(serde_json::from_str::<ListObject>(r#""list""#).is_ok());
        let error = serde_json::from_str::<ListObject>(r#""chat.completion""#).unwrap_err();
        assert!(error.to_string().contains("expected `list`"), "{error}");
    }
}
