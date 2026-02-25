# list recipes by default
default:
  just --list

# run the ollama-cli
tui:
	cargo run --bin ollama-cli -- tui

# generate JSON Schema files for all config types
schema:
	cargo run -p djinn-cli --bin schema-gen

# check for typos, lint, and test
check:
	typos
	cargo clippy -- -D warnings
	cargo test --all

# get the contents of the ollama TUI log
open_log:
	cat  ~/.local/state/ollama_tui/tui.log

# list modelfiles on this machine
list_modelfiles:
  ls ~/.local/share/ollama_tui/modelfile/
