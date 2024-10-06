use std::{fs::File, path::PathBuf};

use clap::{Parser, Subcommand};
use ollama::ModelHost;
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt};
use tui::AppContext;

mod error;
mod lm;
mod ollama;
mod tui;

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value_t)]
    address: ModelHost,
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

fn setup_tracing() {
    tracing_subscriber::fmt()
        .json()
        .with_span_events(FmtSpan::FULL)
        .with_writer(File::create("logs.log").unwrap())
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_tracing();
    let args = Cli::parse();

    let client = ollama::Client::new(args.address.url()).await?;

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
            let app_context = AppContext::new(client);
            let terminal = ratatui::init();
            app_context.run(terminal).await?;
            ratatui::restore();
        }
    }

    Ok(())
}
