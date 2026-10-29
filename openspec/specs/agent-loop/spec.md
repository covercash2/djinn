# agent-loop Specification

## Purpose
Defines the agent's core cycle: ask the model, run any tools it requests, feed
results back, and finish with an answer. It must be bounded, observable, and
robust to bad tool calls.

## Requirements

### Requirement: Answer without tools
When the model replies with text and no tool calls, the agent SHALL return that
text as the answer for the turn without invoking any tool.

#### Scenario: Direct answer
- **WHEN** the user asks "What is Rust?" and the model replies with text only
- **THEN** the turn ends after one model call and the text is the answer

### Requirement: Tool-use cycle
When the model replies with one or more tool calls, the agent SHALL run each
requested tool in order, append the assistant message (with its calls) and one tool
result per call to the conversation, and call the model again, repeating until the
model replies without tool calls.

#### Scenario: One tool then answer
- **WHEN** the model first requests `calculator` with `2+2` and then, given the result `4`, replies "The answer is 4"
- **THEN** the turn makes two model calls, runs the tool once, and returns "The answer is 4"

#### Scenario: Several calls in one reply
- **WHEN** the model requests two tools in one reply
- **THEN** both run in the requested order and both results are appended before the next model call

#### Scenario: Results are matched to calls
- **WHEN** a tool result is appended
- **THEN** it carries the id of the call that requested it

### Requirement: Step limit
Each user turn SHALL allow at most a configured number of model calls (default 8).
When the limit is reached without a final answer the agent SHALL stop, tell the user
that the step limit was reached, and keep the conversation usable for the next turn.

#### Scenario: Model never stops calling tools
- **WHEN** the model requests a tool on every reply
- **THEN** after the configured number of model calls the turn ends with a step-limit message and no further model call is made

#### Scenario: Limit is configurable
- **WHEN** the step limit is set to 2
- **THEN** a turn that needs three model calls ends with the step-limit message after two

### Requirement: Tool failures are reported to the model
If a requested tool does not exist, its arguments do not match its declared
parameters, it returns an error, or it exceeds its time limit, the agent SHALL NOT
abort the turn; it SHALL append a tool result describing the failure and let the
model continue.

#### Scenario: Unknown tool
- **WHEN** the model requests a tool named `delete_everything`
- **THEN** the tool result states the tool does not exist and lists the available tool names, and the turn continues

#### Scenario: Bad arguments
- **WHEN** the arguments are not valid JSON or miss a required parameter
- **THEN** the tool result names the problem and the turn continues

#### Scenario: Tool timeout
- **WHEN** a tool runs longer than its time limit (default 10 seconds)
- **THEN** it is cancelled and the tool result states that it timed out

### Requirement: Tool output limits
Tool results appended to the conversation SHALL be truncated to a maximum size
(default 8 KiB) with a visible marker stating how much was cut.

#### Scenario: Oversized result
- **WHEN** a tool returns 100 KiB of text
- **THEN** the appended result is at most the limit plus the marker, and the marker states the number of bytes omitted

### Requirement: Conversation memory within a session
Within one session the conversation (system message, user turns, assistant replies,
tool calls and results) SHALL be kept and sent to the model on later turns, until
the user resets it.

#### Scenario: Follow-up refers to earlier turn
- **WHEN** the user asks a follow-up question that depends on the previous answer
- **THEN** the model request includes the earlier turns

#### Scenario: Reset
- **WHEN** the user resets the conversation
- **THEN** the next model request contains only the system message and the new user message

### Requirement: Backend errors end the turn cleanly
If the backend reports an error, the turn SHALL end with that error shown, any
partial assistant text kept out of the conversation history, and the session left
usable.

#### Scenario: Server error mid-turn
- **WHEN** the backend fails during the second model call of a turn
- **THEN** the error is shown, the turn's messages up to the failure are discarded from history, and the next user message is accepted

### Requirement: Observable events
The agent loop SHALL emit events for streamed text, tool call started (name and
arguments), tool call finished (result or error and duration), and turn finished,
so that a user interface can render them without parsing text.

#### Scenario: Interface receives tool events
- **WHEN** a turn uses one tool
- **THEN** a started event with the tool name and arguments precedes a finished event for the same call

### Requirement: Tool approval
Before running a requested tool the agent SHALL consult an approval policy supplied
by the front end, which allows the call or denies it. The agent SHALL wait for the
decision without a time limit, and cancelling the turn while a decision is pending
SHALL discard the turn like any other cancellation. When no policy is supplied every
call SHALL be allowed (the plain REPL, `--once` and library users are unchanged).

#### Scenario: Approval comes before the tool starts
- **WHEN** a policy is consulted for a call
- **THEN** the decision is made before the call is reported as started; an allowed call then reports started and finished as usual, and a denied call reports a single "denied" event with the reason instead

#### Scenario: No policy
- **WHEN** the agent runs without an approval policy
- **THEN** behaviour is identical to the existing tool-use cycle

#### Scenario: Unknown tool
- **WHEN** the model requests a tool that does not exist
- **THEN** the policy is not consulted and the model is told the tool does not exist

### Requirement: Denied calls are reported to the model
A denied call SHALL NOT run the tool. The agent SHALL append a tool result stating
that the user denied the call (including the user's reason when one was given) and let
the model continue, exactly like other tool failures.

#### Scenario: Denied call
- **WHEN** the policy denies a `read_file` call with the reason "private"
- **THEN** the file is not read, the tool result tells the model the user denied the call because "private", and the turn continues

#### Scenario: Several calls in one reply
- **WHEN** the model requests two tools and the first is denied
- **THEN** the second is still presented for a decision, and both results are appended in order before the next model call

### Requirement: Model calls are announced
The agent loop SHALL emit an event before each model call of a turn that carries the
call's number within the turn (starting at 1), so that an interface can show progress
against the step limit.

#### Scenario: Two model calls
- **WHEN** a turn needs a tool call and then a final answer
- **THEN** events announce call 1 and call 2, each before the corresponding model request
