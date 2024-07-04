# `djinn-rs`

a Rust playground
for running language models
and some other machine learning models.

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

# examples

run the server on a Macbook M-series:

```sh
cargo run --release --features djinn-core/mac -- server --name test
```

breakdown:
    - `cargo run --release` to build in release mode for best performance
    - `--features djinn-core/mac` to enable CoreML acceleration
    - `--` everything before this are `cargo` args and everything after are `djinn` args
    - `server` command to run the server
    - `--name test` to run the config named `test`, in `./configs/server/test.toml`
