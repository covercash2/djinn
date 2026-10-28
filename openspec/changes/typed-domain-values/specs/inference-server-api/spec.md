## ADDED Requirements

### Requirement: Completion parameters are validated when parsed
`POST /complete` and `POST /complete/stream` SHALL reject, with a JSON error
that names the parameter and the rule it broke, a request whose `temperature` is
outside 0 to 2, whose `top_p` is not above 0 and at most 1, or whose
`sample_len` is 0. Rejection SHALL happen before the model is used, and the
status SHALL be the same one the server already uses for an unparsable body.

#### Scenario: Out-of-range temperature
- **WHEN** a client posts `{"prompt":"hi","temperature":3}` to `/complete`
- **THEN** the response is a client error naming `temperature`, and the model is not run

#### Scenario: Zero sample length
- **WHEN** a client posts `{"prompt":"hi","sample_len":0}`
- **THEN** the response is a client error naming `sample_len`

#### Scenario: Defaults still work
- **WHEN** a client posts `{"prompt":"hi"}` with no parameters
- **THEN** the request succeeds as before
