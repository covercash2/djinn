# Proposal

> **Sequence:** this is change **2 of 2**. **Prerequisite:** change
> `restructure-workspace` must be implemented (and ideally archived) first.
> This change assumes: `djinn-core` is candle-free, the `djinn-models` crate
> exists and owns model execution, `/clip` and the vision/detection models are
> gone, and `ModelArchitecture` currently contains only `QMistral`. If you
> start here and `djinn-models` does not exist, stop and do change 1 first.

## Why

We want an agent that can answer questions and use a few simple tools,
running against our own inference infrastructure but not welded to it. Today
that is impossible for three separate reasons:

1. **The server has no chat or tool interface.** It exposes raw
   `prompt -> tokens` completion only: no message roles, no chat template, no
   stop sequences, no way to describe tools or receive a tool call.
2. **The only model is a base model.** `lmz/candle-mistral` is a quantized
   Mistral 7B v0.1 *base* model: it was never trained to follow a chat format
   or to emit tool calls, so an agent on top of it would be unreliable no matter
   how good the harness is.
3. **There is no agent code at all**, and it must not be embedded in the
   server: the harness has to run against *any* backend (our server, Ollama,
   llama.cpp, vLLM).

## What Changes

- **New wire protocol on the server:** an OpenAI-compatible subset,
  `POST /v1/chat/completions` (JSON and SSE streaming, with `tools` and
  `tool_calls`) and `GET /v1/models`. This is the one protocol the agent speaks,
  and it is also what Ollama, llama.cpp, and vLLM serve, so the agent works with
  all of them. The existing `/complete` and `/complete/stream` routes are
  unchanged.
- **Candle upgrade:** bump `candle-core` / `candle-nn` / `candle-transformers` from
  0.8.4 to the current release (0.11.0 when this was planned). Only
  `djinn-models` depends on candle after change 1, so this is a one-crate change. The
  version itself is not a requirement; the point is to write the new loader against
  the current API (e.g. explicit `clear_kv_cache()`).
- **New small instruct model with native tool calling:** Qwen2.5-Instruct (GGUF,
  quantized, small sizes first: 0.5B for fast checks, 1.5B as the default) via
  candle's `quantized_qwen2`, with a Qwen chat template (ChatML),
  tool descriptions in the prompt, tool-call parsing from the model output, and
  stop-sequence support.
- **BREAKING** remove the base `QMistral` model, its `ModelArchitecture`
  variant, and `configs/model/q_mistral.toml`; the default server config now
  loads Qwen2.5-Instruct. Raw `/complete` keeps working but now runs the
  instruct model without a chat template.
- **New shared wire types in `djinn-core`:** serde types for chat requests,
  responses, streaming chunks, tools and tool calls, used by both the server and
  the agent so they cannot drift. `djinn-core` stays candle-free.
- **New crate `djinn-agent`** (library + `djinn-agent` binary), independent of
  `djinn-models` and of the server. It contains: a `Backend` abstraction with an
  OpenAI-compatible HTTP implementation; a `Tool` abstraction with three
  read-only built-in tools (`current_time`, `read_file`, `calculator`); the
  agent loop with a step limit; and a plain stdin/stdout REPL.
- Documentation: `CLAUDE.md`, `README.md`, and an example config.

Explicitly **out of scope** (see `design.md` Non-Goals): a `ratatui` TUI
(planned later), tools that write files, run commands, or use the network,
multi-user sessions, authentication, context-window management, persistent
conversation history, embeddings, and Qwen3 (dense Qwen3 is available after the candle upgrade; it needs a second chat template and `<think>` handling, so it is a follow-up).

## Capabilities

### New Capabilities

- `chat-completions-api`: the server's OpenAI-compatible chat endpoints:
  request/response shapes, streaming, tool calls, stop sequences, finish
  reasons, error behavior.
- `chat-model-runtime`: what the model layer guarantees for chat: supported
  model, chat templating, tool-call extraction, stop and length handling.
- `agent-backend`: how the agent talks to an inference backend: configuration,
  streaming, tool calls, error reporting; works with any OpenAI-compatible
  server.
- `agent-loop`: the conversation/tool-use cycle: when to call tools, how
  results and errors are fed back, step limits, history within a session.
- `agent-tools`: the tool contract and the three built-in read-only tools,
  including their safety limits.
- `agent-repl`: the interactive command-line experience: flags, prompt,
  streaming output, slash commands, exit behavior, one-shot mode.

### Modified Capabilities

<!-- None. `inference-server-api` and `workspace-layout` (introduced by change 1)
     stay valid: /complete is unchanged and the model-variant requirement is
     phrased generically ("text-generation language model"). -->

## Impact

- **New crates/dirs:** `djinn-agent/`; new modules in `djinn-core` (chat wire
  types), `djinn-models` (chat model, template, tool-call parser, stop
  handling, Qwen loading), `djinn-server` (chat routes).
- **New dependencies:** `reqwest` (already in the lockfile through hf-hub; the
  agent adds it directly with `json` + `stream`), a minimal time-formatting
  dependency for `current_time`, `tokio` for the agent. No new TLS stack
  (verify with `cargo tree`). `djinn-agent` must NOT depend on candle,
  `djinn-models`, or `djinn-server`.
- **Configs:** `configs/model/q_mistral.toml` removed; new
  `configs/model/qwen2_5_instruct.toml`; `configs/server/test.toml` points at it;
  schema files regenerated (`just schema`).
- **Docs/CI:** `CLAUDE.md`, `README.md`, `justfile` (`agent` recipe), CI picks the
  new crate up automatically via `--workspace`.
- **Runtime:** the server holds one model behind a mutex; a chat request holds
  it for the whole generation. Prompt prefill is recomputed per request (no
  cross-request KV reuse), so very long conversations get slower.
- **Risk:** medium. Biggest unknowns are model quality for tool use on small
  quantized models, GGUF file layout for the chosen model, and exact chat-template
  fidelity. `design.md` lists mitigations and what to verify first.
