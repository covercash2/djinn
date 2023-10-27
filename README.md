# djinn

Right now the Rust implementation is most complete.

Run with command line arguments:

```sh
$ cd djinn-rs
$ cargo run -- --help
# example prompt
$ cargo run --release -- llama --disable-flash-attention --model-version v1 --dtype f32
 "In a world" --sample-len 256 --repeat-penalty 50
```
