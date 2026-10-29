# agent-loop Specification

## ADDED Requirements

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
