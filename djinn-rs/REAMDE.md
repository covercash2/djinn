# `djinn-rs`

a Rust implementation 
of several language and vision models
for my experimentation and use.

# troubleshooting

## `candle-kernels` not compiling

after a recent upgrade,
the CUDA kernels were not compiling.
there is a [note in the `candle` README](https://github.com/huggingface/candle?tab=readme-ov-file#compiling-with-flash-attention-fails)
about changing the C compiler.
this command allowed the project to build
on my CUDA machine:

```nu
$env.NVCC_CCBIN = /usr/bin/gcc-13
$env.CC = /usr/bin/gcc-13
```
