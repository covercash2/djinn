# pass recipe arguments to the shell exactly as typed (see the `agent` recipe)
set positional-arguments

# Code that needs real model weights, the Hugging Face Hub or a GPU cannot run in unit tests, so it
# is left out of the coverage numbers. Keep this in sync with `.github/workflows/rust.yml`.
coverage_ignore := '(djinn-models/src/lm/(model\.rs|loader/)|djinn-cli/src/main\.rs|djinn-server/src/lib\.rs)'

# list recipes by default
default:
	just --list

# run the djinn server (loads configs/server/<name>.toml)
server name="test":
	#!/usr/bin/env sh
	case "$(uname -s)" in
		Darwin) feature=djinn-models/mac ;;
		Linux) feature=djinn-models/cuda ;;
		*) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;;
	esac
	cargo run --release --features "$feature" -- server-config --name {{name}}

# run the ollama-cli
tui:
	cargo run --bin ollama-cli -- tui

# generate JSON Schema files for all config types
schema:
	cargo xtask schema

# djinn-core must stay free of the inference stack (model execution lives in djinn-models)
check-core-deps:
	#!/usr/bin/env sh
	if cargo tree -p djinn-core --prefix none | grep -E "^(candle-|tokenizers |hf-hub )"; then
		echo "error: djinn-core must not depend on candle, tokenizers, or hf-hub" >&2
		exit 1
	fi

# djinn-agent must not depend on the inference stack or the server (it talks to any backend over HTTP)
check-agent-deps:
	#!/usr/bin/env sh
	if cargo tree -p djinn-agent --prefix none | grep -E "^(candle-|djinn-models |djinn-server )"; then
		echo "error: djinn-agent must not depend on candle, djinn-models, or djinn-server" >&2
		exit 1
	fi

# check for typos, lint, test, and enforce coverage threshold
check:
	typos
	just check-core-deps
	just check-agent-deps
	cargo fmt --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo llvm-cov nextest --all --ignore-filename-regex '{{coverage_ignore}}' --fail-under-lines 30

# generate HTML coverage report (opens at target/llvm-cov/html/index.html)
coverage:
	cargo llvm-cov nextest --all --ignore-filename-regex '{{coverage_ignore}}' --html
	@echo "Report: target/llvm-cov/html/index.html"

# run the agent REPL (flags after the recipe name are passed through, e.g. `just agent --once "What is 2+2?"`)
agent *args:
	@cargo run -q -p djinn-agent -- "$@"

# get the contents of the ollama TUI log
open_log:
	cat ~/.local/state/ollama_tui/tui.log

# list modelfiles on this machine
list_modelfiles:
	ls ~/.local/share/ollama_tui/modelfile/
