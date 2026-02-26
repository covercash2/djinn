# `djinn-rs`

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
and HTMX front-end
for running models

### Frontend

`djinn-server` ships a small HTMX-powered UI served from `./djinn-server/assets/`.

| Path | Description |
|------|-------------|
| `/` | Main page — prompt input + live completion via HTMX |
| `/swagger-ui` | Interactive OpenAPI docs |
| `/health-check` | Liveness probe |

**How it works**

1. `index.html` is served as a static file by `ServeDir`.
2. The form POSTs to `/ui/complete` (form-encoded).
3. The server runs inference and returns an HTML fragment.
4. HTMX swaps the fragment into `#response` without a page reload.

**HTMX version**: loaded from the [unpkg CDN](https://unpkg.com/htmx.org@2.0.4).

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
