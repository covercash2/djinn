use std::{
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
};

use djinn_core::{
    config::DEFAULT_CONFIG_DIR,
    lm::config::ModelRun,
    lm::{mistral::create_new_context, model::ModelContext},
};
use djinn_server::Config;
use tracing::instrument;

use clap::Parser;

const DEFAULT_HOST_ADDR: &str = "::1";
const DEFAULT_HOST_PORT: u16 = 8080;
const DEFAULT_MODEL_CONFIG: &str = "mistral/fib";

#[derive(Parser, Clone, Debug, PartialEq)]
pub struct ServerArgs {
    #[arg(long, default_value = DEFAULT_HOST_ADDR)]
    ip: String,
    #[arg(long, default_value_t = DEFAULT_HOST_PORT)]
    port: u16,
    /// Where server configs are stored
    #[arg(long, default_value = DEFAULT_CONFIG_DIR)]
    config_dir: PathBuf,
    /// An optional name of this config to save to [`ServerArgs::config_dir`]
    #[arg(long)]
    save_config: Option<String>,
    #[arg(long, default_value = DEFAULT_MODEL_CONFIG)]
    model_config: String,
}

impl Default for ServerArgs {
    fn default() -> Self {
        Self {
            ip: DEFAULT_HOST_ADDR.to_string(),
            port: DEFAULT_HOST_PORT,
            config_dir: PathBuf::from(DEFAULT_CONFIG_DIR),
            save_config: None,
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
    let config = args.clone().try_into()?;

    tracing::info!(?config, "starting server");
    djinn_server::run_server(config).await?;

    if let Some(ref name) = args.save_config {
        tracing::info!(name, "saving config");
        save_config(args).await?;
    }

    Ok(())
}

#[instrument]
async fn save_config(args: ServerArgs) -> anyhow::Result<()> {
    if !args.config_dir.exists() {
        std::fs::create_dir_all(&args.config_dir)?;
    }
    let name = args.save_config.clone().expect("no config name given!");
    let filename = format!("server/{name}.toml");
    let path = args.config_dir.join(filename);

    let config: Config = args.clone().try_into()?;

    let contents = toml::to_string(&config)?;
    tokio::fs::write(path, contents).await?;

    Ok(())
}

pub async fn load_config(path: impl AsRef<Path>) -> anyhow::Result<Config> {
    let path = path.as_ref();
    tracing::info!(?path, "loading config");
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
