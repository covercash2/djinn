//! Command-line arguments.

use std::path::PathBuf;

use clap::Parser;

use crate::backend::{DEFAULT_BASE_URL, DEFAULT_MODEL};

/// A small agent: chat with a model that can use a few read-only tools.
///
/// Talks to any OpenAI-compatible server, such as the djinn server or Ollama
/// (`--base-url http://HOST:11434/v1 --model <name>`).
#[derive(Debug, Parser)]
#[command(name = "djinn-agent", version)]
pub struct Args {
    /// Base URL of the OpenAI-compatible API, up to and including `/v1`.
    #[arg(long, env = "DJINN_AGENT_BASE_URL", default_value = DEFAULT_BASE_URL)]
    pub base_url: String,

    /// Model name sent with every request (the djinn server ignores it).
    #[arg(long, env = "DJINN_AGENT_MODEL", default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Bearer token for the API, if the server wants one.
    #[arg(long, env = "DJINN_AGENT_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// The most model calls one message may cause before the agent gives up.
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_steps: u32,

    /// Directory the `read_file` tool is confined to.
    #[arg(long, default_value = ".")]
    pub tool_root: PathBuf,

    /// File containing the system prompt (replaces the built-in one).
    #[arg(long)]
    pub system: Option<PathBuf>,

    /// Answer this one message, print only the answer to stdout, and exit.
    #[arg(long, value_name = "MESSAGE")]
    pub once: Option<String>,
}

/// Reads the system prompt file, with an error that names it.
pub fn read_system_prompt(path: &std::path::Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read system prompt file `{}`: {error}",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Environment variables are process-wide, so every test that parses arguments takes this lock.
    static ENV: Mutex<()> = Mutex::new(());

    const VARS: [&str; 3] = [
        "DJINN_AGENT_BASE_URL",
        "DJINN_AGENT_MODEL",
        "DJINN_AGENT_API_KEY",
    ];

    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(std::iter::once("djinn-agent").chain(args.iter().copied()))
    }

    /// Runs `body` with a clean environment, restoring nothing (the vars are test-only).
    fn with_env<T>(vars: &[(&str, &str)], body: impl FnOnce() -> T) -> T {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        for var in VARS {
            std::env::remove_var(var);
        }
        for (name, value) in vars {
            std::env::set_var(name, value);
        }
        let result = body();
        for var in VARS {
            std::env::remove_var(var);
        }
        result
    }

    #[test]
    fn defaults_match_the_djinn_server() {
        let args = with_env(&[], || parse(&[]).unwrap());
        assert_eq!(args.base_url, "http://localhost:8080/v1");
        assert_eq!(args.model, "default");
        assert_eq!(args.api_key, None);
        assert_eq!(args.max_steps, 8);
        assert_eq!(args.tool_root, PathBuf::from("."));
        assert_eq!(args.system, None);
        assert_eq!(args.once, None);
    }

    #[test]
    fn environment_variables_are_used_when_no_flag_is_given() {
        let args = with_env(
            &[
                ("DJINN_AGENT_BASE_URL", "http://ollama:11434/v1"),
                ("DJINN_AGENT_MODEL", "qwen2.5"),
                ("DJINN_AGENT_API_KEY", "sk-test"),
            ],
            || parse(&[]).unwrap(),
        );
        assert_eq!(args.base_url, "http://ollama:11434/v1");
        assert_eq!(args.model, "qwen2.5");
        assert_eq!(args.api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    fn flags_override_environment_variables() {
        let args = with_env(
            &[
                ("DJINN_AGENT_MODEL", "a"),
                ("DJINN_AGENT_BASE_URL", "http://env/v1"),
            ],
            || parse(&["--model", "b", "--base-url", "http://flag/v1"]).unwrap(),
        );
        assert_eq!(args.model, "b");
        assert_eq!(args.base_url, "http://flag/v1");
    }

    #[test]
    fn all_flags_parse() {
        let args = with_env(&[], || {
            parse(&[
                "--max-steps",
                "3",
                "--tool-root",
                "/tmp",
                "--system",
                "prompt.txt",
                "--once",
                "What is 2+2?",
                "--api-key",
                "k",
            ])
            .unwrap()
        });
        assert_eq!(args.max_steps, 3);
        assert_eq!(args.tool_root, PathBuf::from("/tmp"));
        assert_eq!(args.system, Some(PathBuf::from("prompt.txt")));
        assert_eq!(args.once.as_deref(), Some("What is 2+2?"));
        assert_eq!(args.api_key.as_deref(), Some("k"));
    }

    #[test]
    fn invalid_values_are_usage_errors_with_exit_code_2() {
        for bad in [
            vec!["--max-steps", "0"],
            vec!["--max-steps", "-1"],
            vec!["--max-steps", "many"],
            vec!["--no-such-flag"],
            vec!["--model"],
        ] {
            let error = with_env(&[], || parse(&bad)).unwrap_err();
            assert_eq!(error.exit_code(), 2, "{bad:?}");
        }
    }

    #[test]
    fn help_and_version_exit_cleanly() {
        for flag in ["--help", "--version"] {
            let error = with_env(&[], || parse(&[flag])).unwrap_err();
            assert_eq!(error.exit_code(), 0, "{flag}");
        }
    }

    #[test]
    fn the_api_key_is_not_echoed_in_help() {
        let error = with_env(&[("DJINN_AGENT_API_KEY", "super-secret")], || {
            parse(&["--help"]).unwrap_err()
        });
        assert!(!error.to_string().contains("super-secret"));
    }

    #[test]
    fn a_system_prompt_file_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("system.txt");
        std::fs::write(&path, "You are terse.\n").unwrap();
        assert_eq!(read_system_prompt(&path).unwrap(), "You are terse.\n");
    }

    #[test]
    fn a_missing_system_prompt_file_is_named_in_the_error() {
        let error = read_system_prompt(std::path::Path::new("/no/such/prompt.txt")).unwrap_err();
        assert!(error.contains("/no/such/prompt.txt"), "{error}");
        assert!(error.contains("system prompt"), "{error}");
    }
}
