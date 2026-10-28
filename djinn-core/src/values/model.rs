//! Values that identify models, Hugging Face Hub repositories and config files.

const MODEL_ID_MAX: usize = 256;
const REPO_PART_MAX: usize = 96;
const REVISION_MAX: usize = 128;
const WEIGHTS_FILE_MAX: usize = 256;
const CONFIG_NAME_MAX: usize = 256;

fn validate_model_id(value: &str) -> Result<(), String> {
    let length = value.chars().count();
    if length == 0 || length > MODEL_ID_MAX {
        return Err(format!(
            "must be 1 to {MODEL_ID_MAX} characters, got {length}"
        ));
    }
    if value.chars().any(char::is_control) {
        return Err("must not contain control characters".to_owned());
    }
    if value.trim() != value {
        return Err("must not start or end with whitespace".to_owned());
    }
    Ok(())
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')
}

fn validate_name_part(part: &str, what: &str) -> Result<(), String> {
    let length = part.chars().count();
    if length == 0 || length > REPO_PART_MAX {
        return Err(format!(
            "the {what} must be 1 to {REPO_PART_MAX} characters, got {length}"
        ));
    }
    if part == "." || part == ".." {
        return Err(format!("the {what} must not be `.` or `..`"));
    }
    if !part.chars().all(is_name_char) {
        return Err(format!(
            "the {what} may only contain ASCII letters, digits, `.`, `_` and `-`"
        ));
    }
    Ok(())
}

fn validate_repo_id(value: &str) -> Result<(), String> {
    let mut parts = value.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err("must have the form `owner/name`".to_owned());
    };
    validate_name_part(owner, "owner")?;
    validate_name_part(name, "name")
}

fn validate_revision(value: &str) -> Result<(), String> {
    let length = value.chars().count();
    if length == 0 || length > REVISION_MAX {
        return Err(format!(
            "must be 1 to {REVISION_MAX} characters, got {length}"
        ));
    }
    if value.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("must not contain whitespace or control characters".to_owned());
    }
    Ok(())
}

fn validate_relative_path(value: &str, max: usize) -> Result<(), String> {
    let length = value.chars().count();
    if length == 0 || length > max {
        return Err(format!("must be 1 to {max} characters, got {length}"));
    }
    if value.chars().any(char::is_control) || value.contains('\\') {
        return Err("must not contain control characters or backslashes".to_owned());
    }
    if value.starts_with('/') {
        return Err("must be a relative path, not start with `/`".to_owned());
    }
    for segment in value.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err("must not contain empty, `.` or `..` path segments".to_owned());
        }
    }
    Ok(())
}

fn validate_weights_file(value: &str) -> Result<(), String> {
    validate_relative_path(value, WEIGHTS_FILE_MAX)
}

fn validate_config_name(value: &str) -> Result<(), String> {
    validate_relative_path(value, CONFIG_NAME_MAX)?;
    if !value
        .split('/')
        .all(|segment| segment.chars().all(is_name_char))
    {
        return Err("may only contain ASCII letters, digits, `.`, `_`, `-` and `/`".to_owned());
    }
    if value.ends_with(".toml") {
        return Err("give the name without the `.toml` extension".to_owned());
    }
    Ok(())
}

string_value! {
    /// The identifier a model is reported under, for example `qwen2.5-1.5b-instruct-q4_k_m` or
    /// `llama3.2:latest`.
    ModelId, "model id", validate_model_id,
    min_length = Some(1), max_length = Some(MODEL_ID_MAX as u32),
    pattern = None
}

string_value! {
    /// A Hugging Face Hub repository, `owner/name`.
    RepoId, "repository id", validate_repo_id,
    min_length = Some(3), max_length = Some(2 * REPO_PART_MAX as u32 + 1),
    pattern = Some("^[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}$")
}

string_value! {
    /// A Hub revision: a branch, tag or commit.
    Revision, "revision", validate_revision,
    min_length = Some(1), max_length = Some(REVISION_MAX as u32),
    pattern = Some("^[^\\s\\x00-\\x1f\\x7f]{1,128}$")
}

string_value! {
    /// A file inside a repository, as a relative `/`-separated path.
    WeightsFile, "weights file", validate_weights_file,
    min_length = Some(1), max_length = Some(WEIGHTS_FILE_MAX as u32),
    pattern = Some("^[^/\\\\\\x00-\\x1f\\x7f][^\\\\\\x00-\\x1f\\x7f]*$")
}

string_value! {
    /// The name of a config under `configs/`, without the `.toml` extension, for example
    /// `model/qwen2_5_instruct`.
    ConfigName, "config name", validate_config_name,
    min_length = Some(1), max_length = Some(CONFIG_NAME_MAX as u32),
    pattern = Some("^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
}

impl RepoId {
    /// The part before the `/`.
    pub fn owner(&self) -> &str {
        self.as_str().split_once('/').map_or("", |(owner, _)| owner)
    }

    /// The part after the `/`.
    pub fn name(&self) -> &str {
        self.as_str().split_once('/').map_or("", |(_, name)| name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_identifiers_are_accepted() {
        assert!(ModelId::new("llama3.2:latest").is_ok());
        assert!(ModelId::new("qwen2.5-1.5b-instruct-q4_k_m").is_ok());
        assert!(RepoId::new("Qwen/Qwen2.5-1.5B-Instruct-GGUF").is_ok());
        assert!(Revision::new("main").is_ok());
        assert!(Revision::new("refs/pr/1").is_ok());
        assert!(Revision::new("9a3b1c").is_ok());
        assert!(WeightsFile::new("qwen2.5-1.5b-instruct-q4_k_m.gguf").is_ok());
        assert!(WeightsFile::new("quantized/model.gguf").is_ok());
        assert!(ConfigName::new("model/qwen2_5_instruct").is_ok());
        assert!(ConfigName::new("test").is_ok());
    }

    #[test]
    fn model_ids_reject_empty_control_and_padding() {
        for bad in ["", " x", "x ", "a\nb", &"m".repeat(257)] {
            assert!(ModelId::new(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn malformed_repository_ids_are_rejected() {
        for bad in [
            "no-slash", "a/b/c", "owner/", "/name", "a b/c", "../x", "x/..", "",
        ] {
            assert!(RepoId::new(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn repository_id_exposes_its_parts() {
        let repo = RepoId::from_static("Qwen/Qwen2.5-1.5B-Instruct");
        assert_eq!(
            (repo.owner(), repo.name()),
            ("Qwen", "Qwen2.5-1.5B-Instruct")
        );
    }

    #[test]
    fn revisions_reject_whitespace_and_empty() {
        for bad in ["", "ma in", "a\tb", &"r".repeat(129)] {
            assert!(Revision::new(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn path_escapes_are_rejected() {
        for bad in [
            "../secret.gguf",
            "/etc/passwd",
            "a/../b",
            "a//b",
            "./a",
            "a\\b",
            "",
        ] {
            assert!(WeightsFile::new(bad).is_err(), "{bad:?}");
        }
        assert!(ConfigName::new("../server/test").is_err());
        assert!(ConfigName::new("/abs").is_err());
        assert!(ConfigName::new("a b").is_err());
    }

    #[test]
    fn config_name_with_extension_says_to_drop_it() {
        let error = ConfigName::new("test.toml").unwrap_err();
        assert!(error.to_string().contains("without the `.toml` extension"));
    }

    #[test]
    fn schema_carries_the_constraints() {
        let schema = serde_json::to_value(schemars::schema_for!(RepoId)).unwrap();
        assert_eq!(
            schema["pattern"],
            "^[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}$"
        );
        assert_eq!(schema["type"], "string");
    }

    #[test]
    fn deserialization_validates() {
        assert!(serde_json::from_str::<RepoId>(r#""not a repo""#).is_err());
        assert!(serde_json::from_str::<WeightsFile>(r#""../../etc/passwd""#).is_err());
        let repo: RepoId = serde_json::from_str(r#""a/b""#).unwrap();
        assert_eq!(serde_json::to_string(&repo).unwrap(), r#""a/b""#);
    }
}
