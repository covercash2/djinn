#![feature(addr_parse_ascii)]

use std::{fs::File, io::Write, path::PathBuf};

use clap::{Parser, Subcommand};
use djinn_core::mistral::{config::ModelRun, run, run_model};
use server::ServerArgs;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod mistral;
mod server;

#[derive(Parser)]
struct Cli {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,
    #[command(subcommand)]
    runner: Runner,
}

#[derive(Subcommand)]
enum Runner {
    Server(ServerArgs),
    ServerConfig {
        #[arg(long)]
        path: PathBuf,
    },
    SingleRun(SingleRunArgs),
    Config(ConfigArgs),
}

#[derive(Parser)]
struct SingleRunArgs {
    /// Pass the name of the config to save
    #[arg(long)]
    save_config: Option<String>,
    /// The model architecture used
    #[command(subcommand)]
    architecture: Architecture,
}

#[derive(Parser)]
struct ConfigArgs {
    #[arg(long)]
    model_name: String,
    #[arg(long)]
    config_name: String,
    #[arg(long, default_value=PathBuf::from("./configs").into_os_string())]
    config_dir: PathBuf,
}

#[derive(Subcommand)]
enum Architecture {
    Mistral(mistral::Args),
}

async fn single_run(args: SingleRunArgs) -> anyhow::Result<()> {
    let save_config = args.save_config.clone();
    let run: ModelRun = match args.architecture {
        Architecture::Mistral(mistral_args) => run(mistral_args.try_into()?).await?,
    };

    if let Some(name) = save_config {
        let contents = toml::to_string(&run)?;
        let path = PathBuf::from(format!("./configs/mistral/{name}.toml"));
        let mut file = File::create(path)?;
        let _ = file.write_all(contents.as_bytes());
    }

    Ok(())
}

impl TryFrom<ConfigArgs> for ModelRun {
    type Error = anyhow::Error;

    fn try_from(value: ConfigArgs) -> anyhow::Result<ModelRun> {
        let path = value
            .config_dir
            .join(value.model_name)
            .join(format!("{}.toml", value.config_name));
        if !path.exists() {
            return Err(anyhow::Error::msg(format!(
                "config does not exist at {path:?}"
            )));
        }

        let contents = std::fs::read_to_string(path)?;
        let data: ModelRun = toml::from_str(&contents)?;

        Ok(data)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    match args.runner {
        Runner::Server(args) => server::run(args).await,
        Runner::ServerConfig { path } => {
            let config = server::load_config(path).await?;
            djinn_server::run_server(config).await
        }
        Runner::SingleRun(args) => single_run(args).await,
        Runner::Config(args) => {
            let config: ModelRun = args.try_into()?;
            //TODO only Mistral is supported for now
            run_model(config).await
        }
    }
}
