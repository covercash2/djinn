# `djinn-rs`

[![CI](https://github.com/covercash2/djinn/actions/workflows/rust.yml/badge.svg)](https://github.com/covercash2/djinn/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/covercash2/djinn/graph/badge.svg)](https://codecov.io/gh/covercash2/djinn)

a Rust playground
for running language models
and some other machine learning models.

## `ollama-cli`

a CLI/TUI for interacting with an [Ollama] server.

## `djinn-cli`
the main entrypoint
for running either
a server (with `djinn-server`)
or a one shot inference.

## `djinn-core`

core ML functionality and model implementations.

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
| `POST /clip` | CLIP text/image similarity |
| `/swagger-ui` | Interactive OpenAPI docs |
| `/health-check` | Liveness probe |

# development

```sh
just check       # typos, clippy, tests
just coverage    # HTML coverage report → target/llvm-cov/html/index.html
just schema      # regenerate JSON Schema files after config struct changes
```

# examples

run the server on a Macbook M-series:

```sh
cargo run --release --features djinn-core/mac -- server-config --name test
```

breakdown:
    - `cargo run --release` to build in release mode for best performance
    - `--features djinn-core/mac` to enable CoreML acceleration
    - `--` everything before this are `cargo` args and everything after are `djinn` args
    - `server-config` command to run the server from a config file
    - `--name test` to run the config named `test`, in `./configs/server/test.toml`

[Ollama]: https://github.com/ollama/ollama/
