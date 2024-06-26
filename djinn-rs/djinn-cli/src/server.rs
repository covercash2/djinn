use std::{
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
};

use djinn_core::mistral::{config::ModelRun, create_new_context, model::ModelContext};
use djinn_server::Config;
use tracing::instrument;

use clap::Parser;

const DEFAULT_HOST_ADDR: &str = "::1";
const DEFAULT_HOST_PORT: u16 = 8080;
const DEFAULT_CONFIG_DIR: &str = "./configs/server/";
const DEFAULT_MODEL_CONFIG: &str = "mistral/fib";

#[derive(Parser, Clone, Debug, PartialEq)]
pub struct ServerArgs {
    #[arg(long, default_value = DEFAULT_HOST_ADDR)]
    ip: String,
    #[arg(long, default_value_t = DEFAULT_HOST_PORT)]
    port: u16,
    #[arg(long, default_value = DEFAULT_CONFIG_DIR)]
    config_dir: PathBuf,
    #[arg(long)]
    name: Option<String>,
    #[arg(long, default_value = DEFAULT_MODEL_CONFIG)]
    model_config: String,
}

impl Default for ServerArgs {
    fn default() -> Self {
        Self {
            ip: DEFAULT_HOST_ADDR.to_string(),
            port: DEFAULT_HOST_PORT,
            config_dir: PathBuf::from(DEFAULT_CONFIG_DIR),
            name: None,
            model_config: DEFAULT_MODEL_CONFIG.to_string(),
        }
    }
}

impl TryFrom<ServerArgs> for Config {
    type Error = anyhow::Error;

    fn try_from(value: ServerArgs) -> anyhow::Result<Self> {
        let ServerArgs {
            ip,
            port,
            model_config,
            ..
        } = value;

        let filename = format!("{model_config}.toml");
        let path = PathBuf::from("./configs/").join(filename);

        let address = IpAddr::parse_ascii(ip.as_bytes())?;
        let full_address = SocketAddr::new(address, port);
        Ok(Config::new(full_address, path))
    }
}

#[instrument]
async fn load_model(config_path: &PathBuf) -> anyhow::Result<ModelContext> {
    let contents = tokio::fs::read_to_string(config_path).await?;
    let run: ModelRun = toml::from_str(&contents)?;
    let context: ModelContext = create_new_context(&run.model_config).await?;
    Ok(context)
}

pub async fn run(args: ServerArgs) -> anyhow::Result<()> {
    let config = if let Some(ref name) = args.name {
        tracing::info!("loading config {name}");
        let filename = format!("{name}.toml");
        let path = args.config_dir.join(filename);
        load_config(path).await?
    } else {
        args.try_into()?
    };

    tracing::info!("starting server: {config:?}");
    djinn_server::run_server(config).await?;

    Ok(())
}

#[instrument]
async fn save_config(name: &str, args: ServerArgs) -> anyhow::Result<()> {
    if !args.config_dir.exists() {
        std::fs::create_dir_all(&args.config_dir)?;
    }
    let filename = format!("{name}.toml");
    let path = args.config_dir.join(filename);

    let config: Config = args.clone().try_into()?;

    let contents = toml::to_string(&config)?;
    tokio::fs::write(path, contents).await?;

    Ok(())
}

pub async fn load_config(path: impl AsRef<Path>) -> anyhow::Result<Config> {
    let contents = tokio::fs::read_to_string(path).await?;
    Ok(toml::from_str(&contents)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_parser_is_valid() {
        ServerArgs::command().debug_assert();
    }

    #[test]
    fn default_config_works() {
        let default_args = ServerArgs::default();
        let _config: Config = default_args
            .try_into()
            .expect("server Config should work with default args");
    }
}
