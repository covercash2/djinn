use std::{fs::File, path::Path};

use clap::{Parser, Subcommand};
use config::Config;
use ollama::ModelHost;
use tracing_subscriber::fmt::format::FmtSpan;
use tui::AppContext;

pub mod bytes_size;
mod config;
mod cursor;
mod error;
mod fs_ext;
mod lm;
mod model_definition;
mod ollama;
mod tui;

#[derive(Parser)]
pub struct Cli {
    #[arg(long)]
    host: Option<ModelHost>,
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    OneShot {
        #[command(subcommand)]
        command: Command,
    },
    Tui,
}

#[derive(Subcommand)]
enum Command {
    Generate(ollama::generate::Request),
    Embed(ollama::generate::Request),
}

fn setup_tracing(log_file: impl AsRef<Path>) -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .json()
        .with_span_events(FmtSpan::FULL)
        .with_writer(File::create(log_file)?)
        .init();

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let config = Config::load()?;
    setup_tracing(&config.log_file)?;

    let host = args.host.as_ref().unwrap_or(&config.host);

    let client = ollama::Client::new(host.url()).await?;

    match args.mode {
        Mode::OneShot { command } => match command {
            Command::Generate(request) => {
                client.generate_stdout(request).await?;
            }
            Command::Embed(request) => {
                let embedding = client.embed(request).await?;
                tracing::info!("{embedding:?}");
            }
        },
        Mode::Tui => {
            color_eyre::install().expect("unable to install color_eyre");
            tracing::info!("starting TUI");
            let app_context = AppContext::new(client, config);
            let terminal = ratatui::init();
            app_context.run(terminal).await?;
            ratatui::restore();
        }
    }

    Ok(())
}
