use clap::{Parser, Subcommand};
use ollama::ModelHost;

mod ollama;

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value_t)]
    address: ModelHost,
    #[command(subcommand)]
    command: Command,
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

    match args.command {
        Command::Generate(request) => {
            client.generate(request).await?;
        }
        Command::Embed(request) => {
            let embedding = client.embed(request).await?;
            println!("{embedding:?}");
        }
    }

    Ok(())
}
