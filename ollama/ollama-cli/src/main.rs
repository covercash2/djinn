use clap::{Parser, Subcommand};
use ollama::ModelHost;
use tui::AppContext;

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    let client = ollama::Client::new(args.address.url()).await?;

    match args.mode {
        Mode::OneShot { command } => match command {
            Command::Generate(request) => {
                client.generate(request).await?;
            }
            Command::Embed(request) => {
                let embedding = client.embed(request).await?;
                println!("{embedding:?}");
            }
        },
        Mode::Tui => {
            color_eyre::install().expect("unable to install color_eyre");
            let app_context = AppContext::new(client);
            let terminal = ratatui::init();
            app_context.run(terminal)?;
            ratatui::restore();
        }
    }

    Ok(())
}
