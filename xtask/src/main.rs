//! Build utilities for the djinn workspace.
//!
//! Invoked via the `cargo xtask` alias (see `.cargo/config.toml`):
//!   cargo xtask schema

use std::{env, fs, path::PathBuf};

use djinn_core::{
    image::config::GenConfig,
    lm::config::{ModelConfig, ModelRun, RunConfig},
};
use djinn_server::Config;
use schemars::JsonSchema;

fn write_schema<T: JsonSchema>(path: PathBuf) {
    let schema = schemars::schema_for!(T);
    let json =
        serde_json::to_string_pretty(&schema).expect("schema serialization should not fail");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("failed to create schema output directory");
    }
    fs::write(&path, json).expect("failed to write schema file");
    println!("wrote {}", path.display());
}

fn schema() {
    let root = project_root::get_project_root().expect("failed to locate workspace root");

    write_schema::<Config>(root.join("configs/server/server.schema.json"));
    write_schema::<ModelConfig>(root.join("configs/model/model.schema.json"));
    write_schema::<RunConfig>(root.join("configs/lm/run-config.schema.json"));
    write_schema::<ModelRun>(root.join("configs/lm/model-run.schema.json"));
    write_schema::<GenConfig>(root.join("configs/image-gen.schema.json"));
}

fn main() {
    let task = env::args().nth(1);
    match task.as_deref() {
        Some("schema") => schema(),
        _ => {
            eprintln!("usage: cargo xtask <task>");
            eprintln!("tasks:");
            eprintln!("  schema   generate JSON Schema files for all config types");
            std::process::exit(1);
        }
    }
}
