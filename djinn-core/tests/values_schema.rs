//! Snapshots of the schemas the validated values publish (JSON Schema and, with the `openapi`
//! feature, OpenAPI), so a change to a rule shows up as a reviewable diff.

use djinn_core::values::*;

macro_rules! json_schema_snapshots {
    ($($test:ident: $ty:ty),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                let schema = serde_json::to_value(schemars::schema_for!($ty)).unwrap();
                insta::assert_json_snapshot!(stringify!($test), schema);
            }
        )*
    };
}

json_schema_snapshots! {
    json_schema_tool_name: ToolName,
    json_schema_tool_call_id: ToolCallId,
    json_schema_tool_arguments: ToolArguments,
    json_schema_model_id: ModelId,
    json_schema_repo_id: RepoId,
    json_schema_revision: Revision,
    json_schema_weights_file: WeightsFile,
    json_schema_config_name: ConfigName,
    json_schema_system_prompt: SystemPrompt,
    json_schema_stop_sequence: StopSequence,
    json_schema_stop_sequences: StopSequences,
    json_schema_temperature: Temperature,
    json_schema_top_p: TopP,
    json_schema_max_tokens: MaxTokens,
}

#[cfg(feature = "openapi")]
mod openapi {
    use super::*;
    use utoipa::PartialSchema;

    macro_rules! openapi_snapshots {
        ($($test:ident: $ty:ty),* $(,)?) => {
            $(
                #[test]
                fn $test() {
                    let schema = serde_json::to_value(<$ty as PartialSchema>::schema()).unwrap();
                    insta::assert_json_snapshot!(stringify!($test), schema);
                }
            )*
        };
    }

    openapi_snapshots! {
        openapi_tool_name: ToolName,
        openapi_repo_id: RepoId,
        openapi_stop_sequences: StopSequences,
        openapi_temperature: Temperature,
        openapi_top_p: TopP,
        openapi_max_tokens: MaxTokens,
        openapi_prompt: Prompt,
    }
}
