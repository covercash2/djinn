//! The two macros that generate the validated value types. Everything a type needs (construction,
//! parsing, serde, schemas) comes from here so the types cannot drift apart.

/// A validated string. `$validate` is `fn(&str) -> Result<(), String>` returning the rule that was
/// broken. `$min`/`$max`/`$pattern` only feed the published schemas (`Option`s); the real check is
/// `$validate`.
macro_rules! string_value {
    (
        $(#[$meta:meta])*
        $name:ident, $kind:literal, $validate:path,
        min_length = $min:expr_2021, max_length = $max:expr_2021, pattern = $pattern:expr_2021
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(String);

        impl $name {
            /// What this value is called in error messages.
            pub const KIND: &'static str = $kind;

            /// Checks `value` and wraps it.
            pub fn new(value: impl Into<String>) -> Result<Self, $crate::values::ValueError> {
                let value = value.into();
                match $validate(&value) {
                    Ok(()) => Ok(Self(value)),
                    Err(reason) => {
                        Err($crate::values::ValueError::new($kind, reason).with_value(&value))
                    }
                }
            }

            /// For literals and tests only: panics if `value` is invalid.
            #[track_caller]
            pub fn from_static(value: &'static str) -> Self {
                match Self::new(value) {
                    Ok(value) => value,
                    Err(error) => panic!("invalid {}: {error}", stringify!($name)),
                }
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl std::str::FromStr for $name {
            type Err = $crate::values::ValueError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<String> for $name {
            type Error = $crate::values::ValueError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = $crate::values::ValueError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> String {
                value.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl std::borrow::Borrow<str> for $name {
            fn borrow(&self) -> &str {
                &self.0
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = <String as serde::Deserialize>::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
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
                let min_length: Option<u32> = $min;
                let max_length: Option<u32> = $max;
                let pattern: Option<&str> = $pattern;
                schemars::schema::SchemaObject {
                    instance_type: Some(schemars::schema::InstanceType::String.into()),
                    metadata: Some(Box::new(schemars::schema::Metadata {
                        title: Some($kind.to_owned()),
                        ..Default::default()
                    })),
                    string: Some(Box::new(schemars::schema::StringValidation {
                        min_length,
                        max_length,
                        pattern: pattern.map(str::to_owned),
                    })),
                    ..Default::default()
                }
                .into()
            }
        }

        #[cfg(feature = "openapi")]
        impl utoipa::PartialSchema for $name {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
                let min_length: Option<usize> = $min.map(|n: u32| n as usize);
                let max_length: Option<usize> = $max.map(|n: u32| n as usize);
                let pattern: Option<&str> = $pattern;
                utoipa::openapi::ObjectBuilder::new()
                    .schema_type(utoipa::openapi::schema::Type::String)
                    .title(Some($kind))
                    .min_length(min_length)
                    .max_length(max_length)
                    .pattern(pattern)
                    .into()
            }
        }

        #[cfg(feature = "openapi")]
        impl utoipa::ToSchema for $name {}
    };
}

/// A validated number. `$valid` is `const fn(&$inner) -> bool`; `$rule` is the message shown when it
/// is false. `minimum`/`maximum`/`exclusive_minimum` only feed the published schemas.
macro_rules! number_value {
    (
        $(#[$meta:meta])*
        $name:ident($inner:ty), $kind:literal, $valid:path, $rule:literal,
        schema_type = $schema_type:ident,
        minimum = $minimum:expr_2021, maximum = $maximum:expr_2021, exclusive_minimum = $exclusive_minimum:expr_2021
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
        pub struct $name($inner);

        impl $name {
            /// What this value is called in error messages.
            pub const KIND: &'static str = $kind;

            /// Checks `value` and wraps it.
            pub fn new(value: $inner) -> Result<Self, $crate::values::ValueError> {
                if $valid(&value) {
                    Ok(Self(value))
                } else {
                    Err($crate::values::ValueError::new($kind, $rule)
                        .with_value(&value.to_string()))
                }
            }

            /// Wraps a constant, failing the build (or panicking at run time) if it is invalid.
            /// For defaults and literals.
            #[track_caller]
            pub const fn from_const(value: $inner) -> Self {
                assert!($valid(&value), "constant out of range");
                Self(value)
            }

            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }

        impl std::str::FromStr for $name {
            type Err = $crate::values::ValueError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                let parsed = value.trim().parse::<$inner>().map_err(|_| {
                    $crate::values::ValueError::new($kind, "must be a number")
                        .with_value(value)
                })?;
                Self::new(parsed)
            }
        }

        impl TryFrom<$inner> for $name {
            type Error = $crate::values::ValueError;

            fn try_from(value: $inner) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> $inner {
                value.0
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serde::Serialize::serialize(&self.0, serializer)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = <$inner as serde::Deserialize>::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
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
                let minimum: Option<f64> = $minimum;
                let maximum: Option<f64> = $maximum;
                let exclusive_minimum: Option<f64> = $exclusive_minimum;
                schemars::schema::SchemaObject {
                    instance_type: Some(schemars::schema::InstanceType::$schema_type.into()),
                    metadata: Some(Box::new(schemars::schema::Metadata {
                        title: Some($kind.to_owned()),
                        ..Default::default()
                    })),
                    number: Some(Box::new(schemars::schema::NumberValidation {
                        minimum,
                        maximum,
                        exclusive_minimum,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
                .into()
            }
        }

        #[cfg(feature = "openapi")]
        impl utoipa::PartialSchema for $name {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
                let minimum: Option<f64> = $minimum;
                let maximum: Option<f64> = $maximum;
                let exclusive_minimum: Option<f64> = $exclusive_minimum;
                utoipa::openapi::ObjectBuilder::new()
                    .schema_type(utoipa::openapi::schema::Type::$schema_type)
                    .title(Some($kind))
                    .minimum(minimum)
                    .maximum(maximum)
                    .exclusive_minimum(exclusive_minimum)
                    .into()
            }
        }

        #[cfg(feature = "openapi")]
        impl utoipa::ToSchema for $name {}
    };
}
