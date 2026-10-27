# Qwen2.5-Instruct: Hub facts and chat template

Verified against the Hugging Face Hub on 2026-10-26 (task 3.1 of `add-agent-harness`).
Re-verify before relying on file names: Hub repositories can change.

## Weights (GGUF) and tokenizer

| Size | GGUF repo | File | Size on disk | Split? |
|------|-----------|------|--------------|--------|
| **1.5B (default)** | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1065 MiB | no, single file |
| 0.5B (quick checks) | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 468 MiB | no, single file |

- Both small sizes are single-file GGUFs, so candle's single-file loader works. (Larger sizes in
  the official repos may be split into several files; that is not a concern for the small
  default.)
- The GGUF repos do not carry a usable `tokenizer.json`; take it from the original model repo:
  `Qwen/Qwen2.5-1.5B-Instruct` / `Qwen/Qwen2.5-0.5B-Instruct` (`tokenizer.json`, ~7 MB).
- `tokenizer_config.json`: `eos_token` = `<|im_end|>`, `pad_token` = `<|endoftext|>`,
  `bos_token` = null, `model_max_length` = 131072.
- `generation_config.json`: `eos_token_id` = `[151645, 151643]` (`<|im_end|>` and `<|endoftext|>`),
  suggested `temperature` 0.7, `top_p` 0.8, `top_k` 20, `repetition_penalty` 1.1.
  Both end-of-turn tokens must stop generation.

## Chat template (authoritative: `tokenizer_config.json` -> `chat_template`)

This Jinja template is the source of truth for prompt rendering. Golden tests must be derived
from it, not from prose descriptions. Notable details:

- Default system message when none is given:
  `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`
- With `tools`, the system turn is the system message, then
  `\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>`,
  then one `\n` + JSON per tool, then
  `\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n`.
- An assistant message with tool calls renders `<|im_start|>assistant`, then `\n` + content if
  any, then for each call `\n<tool_call>\n{"name": "<name>", "arguments": <arguments as JSON>}\n</tool_call>`,
  then `<|im_end|>\n`. Note: `arguments` is emitted as a JSON **object** (the template applies
  `tojson` to the arguments value), whereas the OpenAI wire format carries it as a JSON-encoded
  string. Parse the wire string to a value before rendering.
- Consecutive `tool` messages are merged into a single `<|im_start|>user` turn; each is wrapped as
  `\n<tool_response>\n<content>\n</tool_response>`, and the turn closes with `<|im_end|>\n` after the
  last one.
- The generation prompt is `<|im_start|>assistant\n`.
- **JSON formatting:** the template's `tojson` is Hugging Face's variant of Python's `json.dumps` with default separators: `", "` between items and `": "` after keys, non-ASCII left unescaped, key order preserved. `serde_json::to_string` emits compact JSON (`","` and `":"`), so the renderer needs a small custom serializer (a custom serde_json `Formatter`) to match, and golden tests must pin this.

```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}

```

## Golden fixtures (`fixtures.json`)

`fixtures.json` holds conversations in OpenAI wire format together with the prompt the **official
Jinja template** renders for them. `Qwen25Template::render` is tested against every entry, so a
change to the Rust renderer that drifts from the real template fails the test.

The fixtures were generated once, outside the repository (the project is Rust-only), with Python
and Jinja2 3.1.6, configured like Hugging Face `transformers` (`ImmutableSandboxedEnvironment`,
`trim_blocks`, `lstrip_blocks`, `tojson` = `json.dumps(ensure_ascii=False)`), from the template
above saved to a file. To regenerate (for example if the template changes):

```sh
nix shell --impure --expr 'let p = import <nixpkgs> {}; in p.python3.withPackages (ps: [ ps.jinja2 ])' \
  -c python3 gen_fixtures.py chat_template.jinja djinn-models/src/lm/qwen/fixtures.json
```

where `chat_template.jinja` is the template text from `tokenizer_config.json` and `gen_fixtures.py` is:

```python
"""One-off generator for djinn-models Qwen2.5 chat-template golden fixtures.

Renders the official Qwen2.5-Instruct `chat_template` (tokenizer_config.json) with Jinja2 the same
way Hugging Face transformers does (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks,
`tojson` = json.dumps(ensure_ascii=False)), for a set of OpenAI-wire-format conversations.

Not part of the repository: the repo is Rust-only. The produced fixtures.json is committed and
NOTES.md records how it was made.
"""
import copy
import json
import sys

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception

template_path, out_path = sys.argv[1], sys.argv[2]
template = env.from_string(open(template_path).read())


def hf_messages(messages):
    """Wire format -> what transformers hands the template: arguments as dicts."""
    out = copy.deepcopy(messages)
    for m in out:
        for call in m.get("tool_calls") or []:
            args = call["function"]["arguments"]
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
    return out


def tool(name, description, parameters):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


CALC = tool(
    "calculator",
    "Evaluate an arithmetic expression",
    {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 2+2"}}, "required": ["expression"]},
)
TIME = tool("current_time", "Get the current UTC time", {"type": "object", "properties": {}})


def call(id_, name, args):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


cases = [
    ("plain_user_only", [{"role": "user", "content": "hi"}], []),
    ("custom_system", [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}], []),
    (
        "multi_turn",
        [
            {"role": "user", "content": "What is Rust?"},
            {"role": "assistant", "content": "A systems language."},
            {"role": "user", "content": "Who made it?"},
        ],
        [],
    ),
    ("system_later_in_conversation", [{"role": "user", "content": "a"}, {"role": "system", "content": "mid"}, {"role": "user", "content": "b"}], []),
    ("tools_default_system", [{"role": "user", "content": "2+2?"}], [CALC]),
    ("tools_custom_system_two_tools", [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "time?"}], [CALC, TIME]),
    (
        "tool_round_trip",
        [
            {"role": "user", "content": "What is 17*23?"},
            {"role": "assistant", "content": None, "tool_calls": [call("call_1", "calculator", {"expression": "17*23"})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "391"},
        ],
        [CALC],
    ),
    (
        "assistant_text_and_tool_call",
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me check.", "tool_calls": [call("call_1", "calculator", {"expression": "2+2"})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
            {"role": "assistant", "content": "It is 4."},
            {"role": "user", "content": "thanks"},
        ],
        [CALC],
    ),
    (
        "two_tool_calls_two_results_merged",
        [
            {"role": "user", "content": "both please"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call("call_1", "calculator", {"expression": "1+1"}), call("call_2", "current_time", {})],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
            {"role": "tool", "tool_call_id": "call_2", "content": "2026-10-26T00:00:00Z"},
        ],
        [CALC, TIME],
    ),
    (
        "unicode_and_nested_args",
        [
            {"role": "user", "content": "café ☃"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call("call_1", "calculator", {"expression": "1.5*2", "meta": {"tags": ["a", "b"], "n": 1, "x": 2.5, "ok": True, "none": None, "s": "café"}})],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "3.0"},
        ],
        [CALC],
    ),
    ("empty_user_content", [{"role": "user", "content": ""}], []),
]

fixtures = []
for name, messages, tools in cases:
    rendered = template.render(messages=hf_messages(messages), tools=tools or None, add_generation_prompt=True)
    fixtures.append({"name": name, "messages": messages, "tools": tools, "expected": rendered})

with open(out_path, "w") as f:
    json.dump(fixtures, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(f"wrote {len(fixtures)} fixtures to {out_path}")
```
