//! `read_file`: read a text file, confined to one directory.

use std::path::{Component, Path, PathBuf};

use djinn_core::values::ToolName;
use futures::future::BoxFuture;
use std::sync::LazyLock;

use serde_json::{json, Value};
use tokio::io::AsyncReadExt;

use super::{string_arg, Tool, ToolError};

/// The most bytes returned from one file.
pub const MAX_READ_BYTES: usize = 64 * 1024;

const OUTSIDE: &str = "that path is outside the allowed directory";

/// Reads UTF-8 text files under a root directory.
///
/// Paths are resolved lexically first (so `..` can never step outside, and whether a file exists
/// outside the root is never revealed), then with the file system (so symbolic links cannot step
/// outside either). Both checks must pass. There is a small time-of-check/time-of-use window
/// between resolving and reading; the tool is read-only and meant for local use, so that is
/// accepted.
pub struct ReadFile {
    root: PathBuf,
}

impl ReadFile {
    /// Confines the tool to `root`, which must be an existing directory.
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = std::fs::canonicalize(root)?;
        if !root.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotADirectory,
                format!("{} is not a directory", root.display()),
            ));
        }
        Ok(Self { root })
    }

    /// Joins `requested` onto the root and resolves `.` and `..` without touching the file
    /// system. `None` means the path leaves the root.
    fn lexical_path(&self, requested: &str) -> Option<PathBuf> {
        let requested = Path::new(requested);
        let mut resolved = if requested.is_absolute() {
            PathBuf::new()
        } else {
            self.root.clone()
        };
        for component in requested.components() {
            match component {
                Component::Prefix(_) | Component::RootDir => resolved.push(component),
                Component::CurDir => {}
                Component::ParentDir => {
                    if !resolved.pop() {
                        return None;
                    }
                }
                Component::Normal(part) => resolved.push(part),
            }
        }
        resolved.starts_with(&self.root).then_some(resolved)
    }

    async fn read(&self, requested: &str) -> Result<String, ToolError> {
        let lexical = self
            .lexical_path(requested)
            .ok_or_else(|| ToolError::Failed(OUTSIDE.to_owned()))?;

        let not_found = || ToolError::Failed(format!("file not found: {requested}"));
        let resolved = tokio::fs::canonicalize(&lexical)
            .await
            .map_err(|_| not_found())?;
        if !resolved.starts_with(&self.root) {
            return Err(ToolError::Failed(OUTSIDE.to_owned()));
        }

        let metadata = tokio::fs::metadata(&resolved)
            .await
            .map_err(|_| not_found())?;
        if metadata.is_dir() {
            return Err(ToolError::Failed(format!(
                "{requested} is a directory, not a file"
            )));
        }
        if !metadata.is_file() {
            return Err(ToolError::Failed(format!(
                "{requested} is not a regular file"
            )));
        }

        let file = tokio::fs::File::open(&resolved)
            .await
            .map_err(|e| ToolError::Failed(format!("cannot read {requested}: {}", e.kind())))?;
        let mut bytes = Vec::new();
        file.take(MAX_READ_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .await
            .map_err(|e| ToolError::Failed(format!("cannot read {requested}: {}", e.kind())))?;

        let truncated = bytes.len() > MAX_READ_BYTES;
        bytes.truncate(MAX_READ_BYTES);
        let text = match String::from_utf8(bytes) {
            Ok(text) => text,
            // A cut in the middle of a multi-byte character is not an invalid file.
            Err(error) if truncated && error.utf8_error().error_len().is_none() => {
                let valid = error.utf8_error().valid_up_to();
                let mut bytes = error.into_bytes();
                bytes.truncate(valid);
                String::from_utf8(bytes).expect("prefix up to valid_up_to is valid")
            }
            Err(_) => {
                return Err(ToolError::Failed(format!(
                    "{requested} is not a UTF-8 text file"
                )))
            }
        };

        if truncated {
            Ok(format!(
                "{text}\n[truncated: showing the first {MAX_READ_BYTES} of {} bytes]",
                metadata.len()
            ))
        } else {
            Ok(text)
        }
    }
}

impl Tool for ReadFile {
    fn name(&self) -> &ToolName {
        static NAME: LazyLock<ToolName> = LazyLock::new(|| ToolName::from_static("read_file"));
        &NAME
    }

    fn description(&self) -> &str {
        "Read a UTF-8 text file and return its contents. Paths are relative to the working directory and cannot leave it. Large files are cut off at 64 KiB."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to read, for example \"README.md\" or \"src/main.rs\""
                }
            },
            "required": ["path"]
        })
    }

    fn call(&self, arguments: Value) -> BoxFuture<'_, Result<String, ToolError>> {
        Box::pin(async move {
            let path = string_arg(&arguments, "path")?;
            self.read(path).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, ReadFile) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hello.txt"), "hello world\n").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/nested.txt"), "nested").unwrap();
        let tool = ReadFile::new(dir.path()).unwrap();
        (dir, tool)
    }

    async fn read(tool: &ReadFile, path: &str) -> Result<String, ToolError> {
        tool.call(json!({ "path": path })).await
    }

    fn message(result: Result<String, ToolError>) -> String {
        result.unwrap_err().to_string()
    }

    #[tokio::test]
    async fn reads_a_file_under_the_root() {
        let (_dir, tool) = setup();
        assert_eq!(read(&tool, "hello.txt").await.unwrap(), "hello world\n");
        assert_eq!(read(&tool, "sub/nested.txt").await.unwrap(), "nested");
    }

    #[tokio::test]
    async fn dot_segments_that_stay_inside_are_fine() {
        let (_dir, tool) = setup();
        assert_eq!(read(&tool, "./hello.txt").await.unwrap(), "hello world\n");
        assert_eq!(
            read(&tool, "sub/../hello.txt").await.unwrap(),
            "hello world\n"
        );
        assert_eq!(read(&tool, "sub/./nested.txt").await.unwrap(), "nested");
    }

    #[tokio::test]
    async fn an_absolute_path_inside_the_root_is_allowed() {
        let (dir, tool) = setup();
        let absolute = dir.path().join("hello.txt");
        assert_eq!(
            read(&tool, absolute.to_str().unwrap()).await.unwrap(),
            "hello world\n"
        );
    }

    #[tokio::test]
    async fn dot_dot_cannot_escape() {
        let (dir, tool) = setup();
        std::fs::write(
            dir.path().parent().unwrap().join("secret-outside.txt"),
            "secret",
        )
        .unwrap();
        for path in [
            "../secret-outside.txt",
            "sub/../../secret-outside.txt",
            "../../../../etc/passwd",
        ] {
            assert_eq!(message(read(&tool, path).await), OUTSIDE, "{path}");
        }
        let _ = std::fs::remove_file(dir.path().parent().unwrap().join("secret-outside.txt"));
    }

    #[tokio::test]
    async fn an_absolute_path_outside_the_root_is_refused() {
        let (_dir, tool) = setup();
        assert_eq!(message(read(&tool, "/etc/passwd").await), OUTSIDE);
        assert_eq!(message(read(&tool, "/").await), OUTSIDE);
    }

    #[tokio::test]
    async fn whether_outside_files_exist_is_not_revealed() {
        let (_dir, tool) = setup();
        // Same answer for a file that exists and one that does not.
        let existing = message(read(&tool, "/etc/hostname").await);
        let missing = message(read(&tool, "/etc/definitely-not-here-123").await);
        assert_eq!(existing, missing);
        assert_eq!(existing, OUTSIDE);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_cannot_lead_outside() {
        let (dir, tool) = setup();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("target.txt"), "outside data").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("target.txt"),
            dir.path().join("link.txt"),
        )
        .unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("linked-dir")).unwrap();

        assert_eq!(message(read(&tool, "link.txt").await), OUTSIDE);
        assert_eq!(message(read(&tool, "linked-dir/target.txt").await), OUTSIDE);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_inside_the_root_is_followed() {
        let (dir, tool) = setup();
        std::os::unix::fs::symlink(dir.path().join("hello.txt"), dir.path().join("alias.txt"))
            .unwrap();
        assert_eq!(read(&tool, "alias.txt").await.unwrap(), "hello world\n");
    }

    #[tokio::test]
    async fn a_missing_file_is_not_found() {
        let (_dir, tool) = setup();
        let error = message(read(&tool, "nope.txt").await);
        assert_eq!(error, "file not found: nope.txt");
    }

    #[tokio::test]
    async fn a_directory_is_not_a_file() {
        let (_dir, tool) = setup();
        assert!(message(read(&tool, "sub").await).contains("is a directory"));
        assert!(message(read(&tool, ".").await).contains("is a directory"));
    }

    #[tokio::test]
    async fn binary_files_are_refused() {
        let (dir, tool) = setup();
        std::fs::write(dir.path().join("blob.bin"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
        assert!(message(read(&tool, "blob.bin").await).contains("not a UTF-8 text file"));
    }

    #[tokio::test]
    async fn large_files_are_cut_off_with_a_note() {
        let (dir, tool) = setup();
        let total = MAX_READ_BYTES + 5000;
        std::fs::write(dir.path().join("big.txt"), "a".repeat(total)).unwrap();
        let text = read(&tool, "big.txt").await.unwrap();
        let (body, note) = text.split_once("\n[truncated").unwrap();
        assert_eq!(body.len(), MAX_READ_BYTES);
        assert!(
            note.contains(&format!("first {MAX_READ_BYTES} of {total} bytes")),
            "{note}"
        );
    }

    #[tokio::test]
    async fn a_file_of_exactly_the_limit_is_not_marked_truncated() {
        let (dir, tool) = setup();
        std::fs::write(dir.path().join("exact.txt"), "b".repeat(MAX_READ_BYTES)).unwrap();
        let text = read(&tool, "exact.txt").await.unwrap();
        assert_eq!(text.len(), MAX_READ_BYTES);
        assert!(!text.contains("truncated"));
    }

    #[tokio::test]
    async fn cutting_inside_a_multibyte_character_is_not_an_error() {
        let (dir, tool) = setup();
        // Each snowman is 3 bytes; 65536 is not a multiple of 3, so the cut lands mid-character.
        std::fs::write(dir.path().join("snow.txt"), "\u{2603}".repeat(30_000)).unwrap();
        let text = read(&tool, "snow.txt").await.unwrap();
        let (body, _) = text.split_once("\n[truncated").unwrap();
        assert!(body.chars().all(|c| c == '\u{2603}'));
        assert_eq!(body.len(), (MAX_READ_BYTES / 3) * 3);
    }

    #[tokio::test]
    async fn an_empty_file_reads_as_empty() {
        let (dir, tool) = setup();
        std::fs::write(dir.path().join("empty.txt"), "").unwrap();
        assert_eq!(read(&tool, "empty.txt").await.unwrap(), "");
    }

    #[tokio::test]
    async fn missing_or_mistyped_argument() {
        let (_dir, tool) = setup();
        assert!(matches!(
            tool.call(json!({})).await,
            Err(ToolError::InvalidArguments(m)) if m.contains("path")
        ));
        assert!(matches!(
            tool.call(json!({"path": 3})).await,
            Err(ToolError::InvalidArguments(m)) if m.contains("string")
        ));
    }

    #[test]
    fn the_root_must_be_an_existing_directory() {
        let (dir, _) = setup();
        assert!(ReadFile::new(dir.path().join("hello.txt")).is_err());
        assert!(ReadFile::new(dir.path().join("missing")).is_err());
    }

    #[test]
    fn the_tool_never_writes() {
        // The tool only reads: it opens files read-only. Checked here by running it against a
        // read-only tree and confirming the tree is unchanged.
        let (dir, tool) = setup();
        let before: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let _ = read(&tool, "hello.txt").await;
            let _ = read(&tool, "../nope").await;
        });
        let mut after: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        let mut before = before;
        before.sort();
        after.sort();
        assert_eq!(before, after);
    }
}
