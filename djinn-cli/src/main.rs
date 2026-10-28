use std::{
    fmt::{Debug, Display},
    fs::File,
    io::Write,
    path::PathBuf,
};

use clap::{Parser, Subcommand, ValueEnum};
use djinn_core::values::ConfigName;
use djinn_core::{config::DEFAULT_CONFIG_DIR, lm::config::ModelRun};
use djinn_models::lm::loader::{run, run_model};
use server::ServerArgs;
use tracing::Instrument;
use tracing_chrome::{ChromeLayerBuilder, FlushGuard};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod server;
mod single_run;

const DEFAULT_LOG_ENV: &str = "warn,djinn_server=debug,djinn_core=debug,axum=debug,axum::rejection=trace,candle_core=info,tower_http=debug";

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
    /// Run the server with the given CLI args
    Server(ServerArgs),
    /// Run the server from a given path
    ServerConfig {
        #[arg(long)]
        name: ConfigName,
        #[arg(long, default_value = DEFAULT_CONFIG_DIR)]
        config_dir: PathBuf,
    },
    SingleRun(SingleRunArgs),
    Config(ConfigArgs),
}

#[derive(Parser)]
struct SingleRunArgs {
    /// Pass the name of the config to save
    #[arg(long)]
    save_config: Option<ConfigName>,
    #[command(flatten)]
    run: single_run::Args,
}

#[derive(Parser)]
struct ConfigArgs {
    #[arg(long)]
    model_name: ConfigName,
    #[arg(long)]
    config_name: ConfigName,
    #[arg(long, default_value=PathBuf::from(DEFAULT_CONFIG_DIR).into_os_string())]
    config_dir: PathBuf,
}

async fn single_run(args: SingleRunArgs) -> anyhow::Result<()> {
    let save_config = args.save_config.clone();
    let run: ModelRun = run(args.run.try_into()?).await?;

    if let Some(name) = save_config {
        let contents = toml::to_string(&run)?;
        // `djinn-cli config --model-name <variant> --config-name <name>` loads it back.
        let dir = PathBuf::from(DEFAULT_CONFIG_DIR).join(run.model_config.variant.config_name());
        std::fs::create_dir_all(&dir)?;
        let mut file = File::create(dir.join(format!("{name}.toml")))?;
        file.write_all(contents.as_bytes())?;
    }

    Ok(())
}

impl TryFrom<ConfigArgs> for ModelRun {
    type Error = anyhow::Error;

    fn try_from(value: ConfigArgs) -> anyhow::Result<ModelRun> {
        let path = value
            .config_dir
            .join(value.model_name.as_str())
            .join(format!("{}.toml", value.config_name));
        if !path.exists() {
            return Err(anyhow::Error::msg(format!(
                "config does not exist at {path:?}"
            )));
        }

        let contents = std::fs::read_to_string(&path)?;
        let data: ModelRun = djinn_core::config::validate_and_load::<ModelRun>(&contents, &path)?;

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

fn setup_tracing(tracing_args: TracingArgs) -> anyhow::Result<Option<FlushGuard>> {
    match tracing_args {
        TracingArgs::Chrome => {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Ok(Some(guard))
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
        Runner::ServerConfig { name, config_dir } => {
            let filename = format!("server/{name}.toml");
            let path = config_dir.join(filename);
            let config = server::load_config(path).await?;
            let span = tracing::info_span!("run_server span");
            djinn_server::run_server(config).instrument(span).await
        }
        Runner::SingleRun(args) => single_run(args).await,
        Runner::Config(args) => {
            let config: ModelRun = args.try_into()?;
            run_model(config).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("djinn-cli").chain(args.iter().copied()))
    }

    #[test]
    fn server_config_takes_a_name_without_an_extension() {
        assert!(parse(&["server-config", "--name", "test"]).is_ok());
        let error = parse(&["server-config", "--name", "test.toml"])
            .err()
            .expect("an extension is rejected");
        let shown = error.to_string();
        assert!(
            shown.contains("--name") && shown.contains("without the `.toml` extension"),
            "{shown}"
        );
    }

    #[test]
    fn config_names_cannot_escape_the_config_directory() {
        assert!(parse(&["server", "--model-config", "../elsewhere"]).is_err());
        assert!(parse(&["server-config", "--name", "/etc/passwd"]).is_err());
        assert!(parse(&["config", "--model-name", "..", "--config-name", "x"]).is_err());
    }

    #[test]
    fn the_documented_invocations_still_parse() {
        for args in [
            &["server-config", "--name", "test"][..],
            &["server", "--ip", "127.0.0.1", "--port", "9000"],
            &["server", "--model-config", "model/qwen2_5_instruct_0_5b"],
            &[
                "config",
                "--model-name",
                "qwen2_5_instruct",
                "--config-name",
                "run",
            ],
        ] {
            assert!(parse(args).is_ok(), "{args:?}");
        }
    }
}
