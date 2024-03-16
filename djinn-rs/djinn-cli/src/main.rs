#![feature(addr_parse_ascii)]

use std::{
    fmt::{Debug, Display},
    fs::File,
    io::Write,
    path::PathBuf,
};

use clap::{Parser, Subcommand, ValueEnum};
use djinn_core::{
    config::DEFAULT_CONFIG_DIR,
    mistral::{config::ModelRun, run, run_model},
};
use server::ServerArgs;
use tracing::{Instrument, Level};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Registry};

mod mistral;
mod server;

const DEFAULT_LOG_ENV: &str = "djinn_server=debug,djinn_core=debug,axum=info";

#[derive(Parser)]
struct Cli {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long, default_value_t)]
    pub tracing: TracingArgs,
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
    #[arg(long, default_value=PathBuf::from(DEFAULT_CONFIG_DIR).into_os_string())]
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

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum TracingArgs {
    Chrome,
    #[default]
    Stdout,
    None,
}

impl Display for TracingArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TracingArgs::Chrome => write!(f, "chrome"),
            TracingArgs::Stdout => write!(f, "stdout"),
            TracingArgs::None => write!(f, "none"),
        }
    }
}

fn setup_tracing(tracing_args: TracingArgs) -> anyhow::Result<Option<Box<dyn Drop>>> {
    match tracing_args {
        TracingArgs::Chrome => {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Ok(Some(Box::new(guard)))
        }
        TracingArgs::Stdout => {
            tracing_subscriber::registry()
                .with(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| DEFAULT_LOG_ENV.into()),
                )
                .with(tracing_subscriber::fmt::layer().pretty())
                .init();

            tracing::info!("tracing started");

            Ok(None)
        }
        TracingArgs::None => Ok(None),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let _guard = setup_tracing(args.tracing)?;
    match args.runner {
        Runner::Server(args) => server::run(args).await,
        Runner::ServerConfig { path } => {
            let config = server::load_config(path).await?;
            let span = tracing::info_span!("run_server span");
            djinn_server::run_server(config).instrument(span).await
        }
        Runner::SingleRun(args) => single_run(args).await,
        Runner::Config(args) => {
            let config: ModelRun = args.try_into()?;
            //TODO only Mistral is supported for now
            run_model(config).await
        }
    }
}
