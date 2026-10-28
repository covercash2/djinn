use std::{
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
};

use djinn_core::config::DEFAULT_CONFIG_DIR;
use djinn_core::values::ConfigName;
use djinn_server::Config;
use tracing::instrument;

use clap::Parser;

const DEFAULT_HOST_ADDR: &str = "::1";
const DEFAULT_HOST_PORT: u16 = 8080;
const DEFAULT_MODEL_CONFIG: &str = "model/qwen2_5_instruct";

#[derive(Parser, Clone, Debug, PartialEq)]
pub struct ServerArgs {
    #[arg(long, default_value = DEFAULT_HOST_ADDR)]
    ip: IpAddr,
    #[arg(long, default_value_t = DEFAULT_HOST_PORT)]
    port: u16,
    /// Where server configs are stored
    #[arg(long, default_value = DEFAULT_CONFIG_DIR)]
    config_dir: PathBuf,
    /// An optional name of this config to save to [`ServerArgs::config_dir`]
    #[arg(long)]
    save_config: Option<ConfigName>,
    /// Name of the model config under the config directory, without `.toml`.
    #[arg(long, default_value = DEFAULT_MODEL_CONFIG)]
    model_config: ConfigName,
}

impl Default for ServerArgs {
    fn default() -> Self {
        Self {
            ip: DEFAULT_HOST_ADDR
                .parse()
                .expect("the default address is valid"),
            port: DEFAULT_HOST_PORT,
            config_dir: PathBuf::from(DEFAULT_CONFIG_DIR),
            save_config: None,
            model_config: ConfigName::from_static(DEFAULT_MODEL_CONFIG),
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

        let full_address = SocketAddr::new(ip, port);
        Ok(Config::new(full_address, path))
    }
}

pub async fn run(args: ServerArgs) -> anyhow::Result<()> {
    let config = args.clone().try_into()?;

    tracing::info!(?config, "starting server");
    djinn_server::run_server(config).await?;

    if let Some(ref name) = args.save_config {
        tracing::info!(%name, "saving config");
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
    Ok(djinn_core::config::validate_and_load::<Config>(
        &contents, path,
    )?)
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
    fn the_default_model_config_is_a_real_file() {
        let config: Config = ServerArgs::default().try_into().unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(&config.model_config);
        assert!(path.exists(), "{} does not exist", path.display());
    }

    #[test]
    fn a_bad_ip_is_a_usage_error_naming_the_flag() {
        let error = ServerArgs::try_parse_from(["server", "--ip", "not-an-ip"]).unwrap_err();
        assert!(error.to_string().contains("--ip"), "{error}");
        assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn ipv4_and_ipv6_addresses_parse() {
        for ip in ["127.0.0.1", "::1", "0.0.0.0"] {
            let args = ServerArgs::try_parse_from(["server", "--ip", ip]).unwrap();
            assert_eq!(args.ip, ip.parse::<IpAddr>().unwrap());
        }
    }

    #[test]
    fn default_config_works() {
        let default_args = ServerArgs::default();
        let _config: Config = default_args
            .try_into()
            .expect("server Config should work with default args");
    }
}
