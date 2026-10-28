use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use djinn_agent::agent::{Agent, AgentConfig, DEFAULT_SYSTEM_PROMPT};
use djinn_agent::backend::OpenAiBackend;
use djinn_agent::cli::{Args, read_system_prompt};
use djinn_agent::repl;
use djinn_agent::tools::ToolRegistry;
use djinn_core::values::SystemPrompt;
use tokio::sync::Notify;

#[tokio::main]
async fn main() -> ExitCode {
    // clap reports usage errors itself and exits with status 2.
    let args = Args::parse();
    match run(args).await {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Returns whether the session ended successfully.
async fn run(args: Args) -> Result<bool, String> {
    // Everything that can fail locally is checked before the server is contacted.
    let system_prompt = match &args.system {
        Some(path) => read_system_prompt(path)?,
        None => SystemPrompt::from_static(DEFAULT_SYSTEM_PROMPT),
    };
    let tools = ToolRegistry::builtin(&args.tool_root).map_err(|error| {
        format!(
            "cannot use --tool-root `{}`: {error}",
            args.tool_root.display()
        )
    })?;

    let backend = OpenAiBackend::new(
        args.base_url.clone(),
        args.model.clone(),
        args.api_key.clone(),
    );
    let config = AgentConfig {
        system_prompt,
        max_steps: args.max_steps as usize,
        ..AgentConfig::default()
    };
    let io_error = |error: std::io::Error| format!("I/O error: {error}");
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();

    if let Some(message) = &args.once {
        let mut agent = Agent::new(backend, tools, config);
        return repl::run_once(&mut agent, message, &mut stdout, &mut stderr)
            .await
            .map_err(io_error);
    }

    if let Err(error) = backend.check_reachable().await {
        eprintln!(
            "warning: {error}\n(starting anyway; start the server and send your message again)"
        );
    }
    eprintln!(
        "djinn-agent: model `{}` at {}. Type /help for commands.",
        args.model, args.base_url
    );

    // Ctrl-C cancels the running turn (or is ignored at the prompt) instead of killing the program.
    let interrupt = Arc::new(Notify::new());
    let signal = interrupt.clone();
    tokio::spawn(async move {
        while tokio::signal::ctrl_c().await.is_ok() {
            signal.notify_waiters();
        }
    });

    let mut agent = Agent::new(backend, tools, config);
    let stdin = tokio::io::BufReader::new(tokio::io::stdin());
    repl::run(&mut agent, stdin, &mut stdout, &mut stderr, &interrupt)
        .await
        .map_err(io_error)?;
    Ok(true)
}
