# inference-server-api Specification

## Purpose
Describes the HTTP surface of `djinn-server` after the restructure: a small JSON
and Server-Sent-Events API for raw text completion, with no image endpoints and
no bundled web frontend.

## Requirements

### Requirement: Health check
The server SHALL answer `GET /health-check` with status 200 and body `OK` whenever
it is running.

#### Scenario: Health check succeeds
- **WHEN** a client sends `GET /health-check`
- **THEN** the response status is 200 and the body is `OK`

### Requirement: Raw completion returns JSON
`POST /complete` SHALL accept a JSON body with a `prompt` string and optional
sampling parameters, run the loaded model to completion, and respond with a JSON
object containing the original `prompt` and the generated `output`. The request
SHALL be served only when `Content-Type: application/json` is present.

#### Scenario: Successful completion
- **WHEN** a client posts `{"prompt": "say hi"}` with `Content-Type: application/json`
- **THEN** the response is 200 with JSON `{"prompt": "say hi", "output": "<generated text>"}`

#### Scenario: Empty prompt is accepted
- **WHEN** a client posts `{"prompt": ""}`
- **THEN** the response is 200

#### Scenario: Missing content type
- **WHEN** a client posts a body without a `Content-Type` header
- **THEN** the response is 415

#### Scenario: Model failure
- **WHEN** the model returns an error while generating
- **THEN** the response is 500 with a JSON `message` that does not expose internal error details

### Requirement: Streaming completion over SSE
`POST /complete/stream` SHALL accept the same request body as `/complete` and
respond with `Content-Type: text/event-stream`. Each generated token SHALL be sent
as an unnamed event whose data is the token text. On normal completion the server
SHALL send a final event named `done` with empty data. If generation fails the
server SHALL send an event named `error` whose data is the error message and then
end the stream without sending `done`.

#### Scenario: Tokens then done
- **WHEN** the model yields tokens `hello` and ` world`
- **THEN** the stream contains data events `hello` and ` world`, followed by `event: done`

#### Scenario: Error mid-stream
- **WHEN** the model returns an error while generating
- **THEN** the stream contains `event: error` and no `event: done`

#### Scenario: Client disconnects
- **WHEN** the client closes the connection during generation
- **THEN** the server stops sending and releases the model for the next request

### Requirement: Inference requests are serialised
The server SHALL run at most one generation at a time against the loaded model;
a second inference request SHALL wait for the first rather than fail. Inference
routes SHALL NOT have a request timeout. Non-inference routes SHALL time out
after 1 second with status 408.

#### Scenario: Concurrent requests
- **WHEN** two clients post to `/complete` at the same time
- **THEN** both eventually receive 200 responses and their outputs are not interleaved

### Requirement: OpenAPI documentation
The server SHALL serve interactive API documentation at `/swagger-ui` and the
machine-readable schema at `/api-doc/openapi.json`, covering every public route.

#### Scenario: Schema lists the routes
- **WHEN** a client fetches `/api-doc/openapi.json`
- **THEN** the document lists `/health-check`, `/complete`, and `/complete/stream` and does not list `/clip`

### Requirement: No image endpoint and no bundled frontend
The server SHALL NOT expose `/clip` or any other image-processing route, SHALL NOT
load a vision model at start-up, and SHALL NOT serve static files. Any unknown
path SHALL return 404.

#### Scenario: Clip route removed
- **WHEN** a client sends `POST /clip`
- **THEN** the response status is 404

#### Scenario: Root path is not a page
- **WHEN** a client sends `GET /`
- **THEN** the response status is 404 with body `Not found`

#### Scenario: Start-up loads only the language model
- **WHEN** the server starts with a valid model config
- **THEN** only the configured language model is loaded and no CLIP weights are fetched

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
