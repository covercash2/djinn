//! Ollama is developed by genius snowflakes
//! who know more about serialized configuration file formats
//! than everyone else who has come before them.
//! Thus, it was necessary for them to create a bespoke file format
//! to define a derived model.
//! This is an attempt to parse that genius file format in Rust.
//!
//! TODO:
//! - [ ] gather samples

mod parser;

type Parameter = (String, String);

pub struct ModelFile {
    from: String,
    parameters: Vec<Parameter>,
    template: Option<String>,
    system: Option<String>,
    adapters: Option<String>,
    license: Option<String>,
    messages: Vec<String>,
}


