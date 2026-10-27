# `djinn-rs`

[![CI](https://github.com/covercash2/djinn/actions/workflows/rust.yml/badge.svg)](https://github.com/covercash2/djinn/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/covercash2/djinn/graph/badge.svg)](https://codecov.io/gh/covercash2/djinn)

a Rust playground
for running language models locally.

## `ollama-cli`

a CLI/TUI for interacting with an [Ollama] server.

## `djinn-cli`
the main entrypoint
for running either
a server (with `djinn-server`)
or a one shot inference.

## `djinn-core`

shared config loading and config types.
Deliberately free of the inference stack (candle, tokenizers, hf-hub).

## `djinn-models`

model loading and token generation, built on [candle].
Hardware acceleration is selected here: `--features djinn-models/mac` (Metal) or `--features djinn-models/cuda`.

## `djinn-server`

an HTTP server
with an API
and SSE streaming
for running models

### Routes

| Path | Description |
|------|-------------|
| `POST /complete` | Run a completion, JSON response |
| `POST /complete/stream` | Run a completion, streamed as Server-Sent Events |
| `POST /v1/chat/completions` | OpenAI-compatible chat completion, with tool calls; `"stream": true` streams SSE |
| `GET /v1/models` | The loaded model |
| `/swagger-ui` | Interactive OpenAPI docs |
| `/health-check` | Liveness probe |

## `djinn-agent`

a small agent: a chat REPL where the model can use a few read-only tools
(`current_time`, `calculator`, `read_file` confined to a directory).
It talks to any OpenAI-compatible server and does not depend on the model code or `djinn-server`.

against the djinn server (start it first, see below):

```sh
just agent                                   # interactive
just agent --once "What is 17 * 23? Use the calculator."
```

against Ollama (any model that supports tools):

```sh
just agent --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
```

flags can also be set with `DJINN_AGENT_BASE_URL`, `DJINN_AGENT_MODEL` and `DJINN_AGENT_API_KEY`.
in the REPL, `/help` lists the commands; tool activity is printed to stderr and the answer to stdout.
the djinn server needs an instruct model for chat; `configs/model/qwen2_5_instruct.toml` (1.5B) and
`configs/model/qwen2_5_instruct_0_5b.toml` (0.5B) are provided.

# development

```sh
just check       # typos, fmt, clippy, tests, core-deps guard
just coverage    # HTML coverage report → target/llvm-cov/html/index.html
just schema      # regenerate JSON Schema files after config struct changes
```

# examples

run the server on a Macbook M-series:

```sh
cargo run --release --features djinn-models/mac -- server-config --name test
```

breakdown:
    - `cargo run --release` to build in release mode for best performance
    - `--features djinn-models/mac` to enable Metal acceleration
    - `--` everything before this are `cargo` args and everything after are `djinn` args
    - `server-config` command to run the server from a config file
    - `--name test` to run the config named `test`, in `./configs/server/test.toml`

[Ollama]: https://github.com/ollama/ollama/
[candle]: https://github.com/huggingface/candle
