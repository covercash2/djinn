use clap::{Parser, Subcommand};

mod ollama;

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Generate(ollama::generate::Request),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    let client = ollama::Client::new("http://hoss:11434".parse()?).await?;

    match args.command {
        Command::Generate(request) => {
            client.generate(request).await?;
        }
    }

    Ok(())
}
