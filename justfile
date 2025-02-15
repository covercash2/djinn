tui:
	cargo run --bin ollama-cli -- tui

check:
	typos
	cargo clippy
	cargo test --all

open_log:
	cat  ~/.local/state/ollama_tui/tui.log

# list modelfiles on this machine
list_modelfiles:
  ls ~/.local/share/ollama_tui/modelfile/
