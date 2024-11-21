tui:
	cargo run --bin ollama-cli -- tui

check:
	typos
	cargo clippy
	cargo test --all
