## ADDED Requirements

### Requirement: Domain values are validated types
Public interfaces in the workspace SHALL represent identifiers, names, paths,
addresses, URLs, secrets, and bounded tuning parameters with validated types,
not bare strings or numbers, except for the free-text values listed in
`domain-values`. The shared types SHALL live in `djinn-core` and SHALL NOT add
the inference stack or other heavy dependencies to it. The convention SHALL be
written down in `CLAUDE.md` and in the OpenSpec project context.

#### Scenario: Core stays light
- **WHEN** `just check-core-deps` is run after the shared types are added
- **THEN** it passes

#### Scenario: Convention is documented
- **WHEN** `CLAUDE.md` is read
- **THEN** it states the rule, names where the shared types live, and lists the free-text exceptions

#### Scenario: Agent independence preserved
- **WHEN** `just check-agent-deps` is run
- **THEN** it passes, and `djinn-agent` still does not depend on candle, `djinn-models`, or `djinn-server`
