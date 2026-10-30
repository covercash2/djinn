//! Runs the real `djinn-cli` binary for the parts of `server` that fail before a model is loaded.

use std::process::Command;

#[test]
fn a_missing_model_config_is_reported_with_its_resolved_path() {
    let output = Command::new(env!("CARGO_BIN_EXE_djinn-cli"))
        .args([
            "server",
            "--config-dir",
            "/nonexistent-djinn-dir",
            "--model-config",
            "model/custom",
        ])
        .output()
        .expect("djinn-cli runs");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("/nonexistent-djinn-dir/model/custom.toml"),
        "{stderr}"
    );
}
