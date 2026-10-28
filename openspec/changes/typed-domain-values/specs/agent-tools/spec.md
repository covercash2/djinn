## MODIFIED Requirements

### Requirement: Tool contract
Every tool SHALL have a unique, valid tool name (see `domain-values`), a
description for the model, and a JSON-schema description of its parameters, and
SHALL run asynchronously taking JSON arguments and returning text or an error
message. The agent SHALL advertise exactly the registered tools to the model on
every request.

#### Scenario: Tools are advertised
- **WHEN** three tools are registered
- **THEN** every model request lists those three with their names, descriptions, and parameter schemas

#### Scenario: Duplicate names rejected
- **WHEN** two tools with the same name are registered
- **THEN** registration fails with an error naming the duplicate

#### Scenario: Invalid name cannot be used
- **WHEN** a tool is defined with the name `read file`
- **THEN** the name is rejected when it is created, so the tool can never be registered or advertised

#### Scenario: Built-in names are valid
- **WHEN** the built-in tools `current_time`, `calculator`, and `read_file` are registered
- **THEN** registration succeeds
