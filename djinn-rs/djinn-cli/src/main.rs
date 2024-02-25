use clap::{Parser, Subcommand};
use djinn_core::mistral::run_model;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod mistral;

#[derive(Parser)]
struct Cli {
    /// Pass the name of the config to save
    #[arg(long)]
    save_config: Option<String>,
    #[command(subcommand)]
    runner: Runner,
}

#[derive(Subcommand)]
enum Runner {
    Server,
    SingleRun(Args),
}

#[derive(Parser)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,
    /// The model architecture used
    #[command(subcommand)]
    architecture: Architecture,
}

#[derive(Subcommand)]
enum Architecture {
    Mistral(mistral::Args),
}

async fn single_run(args: Args) -> anyhow::Result<()> {
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    match args.architecture {
        Architecture::Mistral(mistral_args) => run_model(mistral_args.try_into()?).await?,
    }
    Ok(())
}

//async fn run_server() -> anyhow::Result<()> {
//    let addr: SocketAddrV4 = "127.0.0.1:8090".parse()?;
//    let context = Context::new(Config::new(SocketAddr::from(addr)));
//    let server = HttpServer::new(Arc::from(context));
//
//    server.start().await
//}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    match args.runner {
        Runner::Server => todo!(),
        Runner::SingleRun(args) => single_run(args).await,
    }
}
