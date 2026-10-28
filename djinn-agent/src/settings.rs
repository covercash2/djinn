//! Validated connection settings: where the agent talks to, and the secret it may present.
//!
//! Like the values in [`djinn_core::values`], these cannot hold an invalid value: they are checked
//! when the command line or environment is read, so a typo fails before any request is made.

use std::fmt;
use std::str::FromStr;

use djinn_core::values::ValueError;
use url::Url;

/// The root of an OpenAI-compatible API, such as `http://localhost:8080/v1`.
///
/// Must be an absolute `http` or `https` URL with a host and without user information, query or
/// fragment (so no credentials can hide in a URL that gets logged). A trailing `/` is dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseUrl(Url);

impl BaseUrl {
    /// What this value is called in error messages.
    pub const KIND: &'static str = "base URL";

    pub fn new(value: &str) -> Result<Self, ValueError> {
        let error = |reason: &str| ValueError::new(Self::KIND, reason).with_value(value);

        let mut url = Url::parse(value.trim())
            .map_err(|_| error("must be an absolute URL such as http://localhost:8080/v1"))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(error("must start with http:// or https://"));
        }
        if url.host_str().is_none_or(str::is_empty) {
            return Err(error("must include a host"));
        }
        if !url.username().is_empty() || url.password().is_some() {
            // Not echoed: the value holds the credentials.
            return Err(ValueError::new(
                Self::KIND,
                "must not contain a user name or password",
            ));
        }
        if url.query().is_some() || url.fragment().is_some() {
            return Err(error("must not contain a query or fragment"));
        }
        let path = url.path().trim_end_matches('/').to_owned();
        url.set_path(&path);
        Ok(Self(url))
    }

    /// For literals and tests only: panics if `value` is invalid.
    #[track_caller]
    pub fn from_static(value: &'static str) -> Self {
        match Self::new(value) {
            Ok(url) => url,
            Err(error) => panic!("invalid BaseUrl: {error}"),
        }
    }

    /// The URL of `tail` below the base, for example `endpoint("chat/completions")`.
    pub fn endpoint(&self, tail: &str) -> Url {
        let mut url = self.0.clone();
        let path = format!(
            "{}/{}",
            url.path().trim_end_matches('/'),
            tail.trim_start_matches('/')
        );
        url.set_path(&path);
        url
    }
}

impl fmt::Display for BaseUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.as_str().trim_end_matches('/'))
    }
}

impl FromStr for BaseUrl {
    type Err = ValueError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// A bearer token for the API. Never shown: `Debug` prints a placeholder, there is no `Display`,
/// and the text is only reachable through [`ApiKey::expose`], which is called in exactly one place
/// (building the `Authorization` header).
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// What this value is called in error messages.
    pub const KIND: &'static str = "API key";

    /// The error never contains the key.
    pub fn new(value: impl Into<String>) -> Result<Self, ValueError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ValueError::new(Self::KIND, "must not be empty"));
        }
        if !value.chars().all(|c| c.is_ascii_graphic()) {
            return Err(ValueError::new(
                Self::KIND,
                "may only contain visible ASCII characters (no whitespace or control characters)",
            ));
        }
        Ok(Self(value))
    }

    /// For literals and tests only: panics if `value` is invalid.
    #[track_caller]
    pub fn from_static(value: &'static str) -> Self {
        match Self::new(value) {
            Ok(key) => key,
            Err(error) => panic!("invalid ApiKey: {error}"),
        }
    }

    /// The secret text. Do not log or display it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl FromStr for ApiKey {
    type Err = ValueError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// Parses `--api-key` for clap. A normal `FromStr` parser would make clap print the rejected value in
/// its error message, which for a secret is exactly what must not happen.
#[derive(Clone, Copy, Debug)]
pub struct ApiKeyParser;

impl clap::builder::TypedValueParser for ApiKeyParser {
    type Value = ApiKey;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<ApiKey, clap::Error> {
        let fail = |reason: String| {
            let name = arg.map_or_else(|| "API key".to_owned(), |arg| arg.to_string());
            cmd.clone().error(
                clap::error::ErrorKind::ValueValidation,
                format!("invalid value for '{name}': {reason}"),
            )
        };
        let text = value
            .to_str()
            .ok_or_else(|| fail("API key: must be valid UTF-8".to_owned()))?;
        ApiKey::new(text).map_err(|error| fail(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_base_urls_are_accepted() {
        for url in [
            "http://localhost:8080/v1",
            "https://api.example.com/v1",
            "http://127.0.0.1:11434/v1/",
            "http://[::1]:8080/v1",
            "http://host",
        ] {
            assert!(BaseUrl::new(url).is_ok(), "{url}");
        }
    }

    #[test]
    fn invalid_base_urls_are_rejected_with_the_rule() {
        for (url, reason) in [
            ("localhost:8080", "http:// or https://"),
            ("ftp://host/v1", "http:// or https://"),
            ("http://user:pass@host/v1", "user name or password"),
            ("http://host/v1?x=1", "query or fragment"),
            ("http://host/v1#top", "query or fragment"),
            ("", "absolute URL"),
            ("not a url", "absolute URL"),
        ] {
            let error = BaseUrl::new(url).unwrap_err().to_string();
            assert!(
                error.contains("base URL") && error.contains(reason),
                "{url}: {error}"
            );
        }
    }

    #[test]
    fn credentials_in_a_url_are_rejected_without_being_echoed() {
        let error = BaseUrl::new("http://user:pass-word@host/v1").unwrap_err();
        assert_eq!(error.reason(), "must not contain a user name or password");
        assert!(!error.to_string().contains("pass-word"), "{error}");
    }

    #[test]
    fn trailing_slashes_are_dropped_and_endpoints_are_joined_cleanly() {
        let base = BaseUrl::from_static("http://example.test/v1/");
        assert_eq!(base.to_string(), "http://example.test/v1");
        assert_eq!(
            base.endpoint("chat/completions").as_str(),
            "http://example.test/v1/chat/completions"
        );
        assert_eq!(
            base.endpoint("/models").as_str(),
            "http://example.test/v1/models"
        );
        let bare = BaseUrl::from_static("http://example.test");
        assert_eq!(
            bare.endpoint("models").as_str(),
            "http://example.test/models"
        );
    }

    #[test]
    fn api_keys_must_be_non_empty_without_whitespace() {
        assert!(ApiKey::new("sk-test_123").is_ok());
        assert!(ApiKey::new("").is_err());
        assert!(ApiKey::new("two words").is_err());
        assert!(ApiKey::new("line\nbreak").is_err());
    }

    #[test]
    fn the_key_never_appears_in_debug_or_errors() {
        let key = ApiKey::from_static("super-secret-key");
        assert!(!format!("{key:?}").contains("super-secret-key"));
        assert!(format!("{key:?}").contains("redacted"));
        assert_eq!(key.expose(), "super-secret-key");

        let error = ApiKey::new("super secret").unwrap_err().to_string();
        assert!(!error.contains("super"), "{error}");
    }

    #[test]
    fn a_struct_holding_a_key_prints_it_redacted() {
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Holder {
            key: Option<ApiKey>,
        }
        let shown = format!(
            "{:?}",
            Holder {
                key: Some(ApiKey::from_static("hunter2-hunter2"))
            }
        );
        assert!(!shown.contains("hunter2"), "{shown}");
    }
}
