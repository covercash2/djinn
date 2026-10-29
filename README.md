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

a small agent: a terminal chat UI (or, in pipes and with `--plain`, a line-based REPL) where the model can use a few read-only tools
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
### terminal UI

in a terminal `djinn-agent` opens a full-screen UI: the conversation (assistant text is rendered as Markdown), each tool call as a block that you can expand, a multi-line input box and a status bar. when stdin or stdout is a pipe, or with `--plain`, it runs the line-based REPL instead (`--once` is unchanged).

| key | action |
| --- | --- |
| `Enter` | send the message |
| `Alt-Enter` (`Shift-Enter`, `Ctrl-J`) | new line in the message |
| `Up` / `Down` | recall earlier messages (on the first / last line of the input) |
| `PageUp` / `PageDown`, mouse wheel | scroll the conversation; `Ctrl-Home` / `Ctrl-End` jump to the start / end |
| `Tab` / `Shift-Tab` | select a tool call; `Enter` or `Space` opens or closes it, `Esc` deselects |
| `Ctrl-C` | cancel the running turn (it never quits); `/exit` or `Ctrl-D` quits |

tool calls need your approval before they run: `y` allows once, `a` allows that tool for the rest of the session, `n` denies (type an optional reason; the model is told). `--approval auto` only asks about tools that declare they need confirmation (the built-in tools do not); the default `--approval ask` asks about every call. 

the UI captures the mouse so the wheel can scroll; hold Shift while dragging to select text in most terminals. build without the UI with `--no-default-features`.

the djinn server needs an instruct model for chat; `configs/model/qwen2_5_instruct.toml` (1.5B) and
`configs/model/qwen2_5_instruct_0_5b.toml` (0.5B) are provided.

# development

```sh
just check       # typos, fmt, clippy, tests, core-deps guard
just coverage    # HTML coverage report → target/llvm-cov/html/index.html
just schema      # regenerate JSON Schema files after config struct changes
```

# examples

run the server with NVIDIA GPU acceleration (use `djinn-models/mac` instead on Apple Silicon; the dev shell sets `CUDA_COMPUTE_CAP`, see `flake.nix`):

```sh
cargo run --release --features djinn-models/cuda -- server-config --name test
```

breakdown:
    - `cargo run --release` to build in release mode for best performance
    - `--features djinn-models/cuda` to enable CUDA acceleration
    - `--` everything before this are `cargo` args and everything after are `djinn` args
    - `server-config` command to run the server from a config file
    - `--name test` to run the config named `test`, in `./configs/server/test.toml`

Arguments are validated when they are parsed: for example `djinn-cli server --ip not-an-ip` is a usage error that names `--ip`, before anything starts.

## Validated settings

Identifiers, names, paths, addresses, URLs, secrets and tuning parameters are checked when they are read, not when they are used, so a mistake fails early and says which value and which rule:

```sh
$ djinn-agent --base-url localhost:8080 --once hi
error: invalid value 'localhost:8080' for '--base-url <BASE_URL>': base URL `localhost:8080`: must start with http:// or https://

$ curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"temperature":5}'
{"error":{"message":"Failed to deserialize the JSON body ... temperature `5`: must be a number from 0 to 2", ...}}
```

The same rules apply to TOML configs (`repo_id`, `weights_file`, `temperature`, ...) and are published in the generated JSON Schemas under `configs/` and in the OpenAPI document at `/swagger-ui`, so editors and clients see the constraints (patterns, lengths, numeric bounds). The rules for each value are listed in `djinn-core/src/values/mod.rs`.

[Ollama]: https://github.com/ollama/ollama/
[candle]: https://github.com/huggingface/candle
