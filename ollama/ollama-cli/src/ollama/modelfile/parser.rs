use std::path::PathBuf;

use nom::{
    branch::alt,
    bytes::streaming::{tag, take_while1},
    character::{complete::not_line_ending, streaming::{multispace1, newline, space1}},
    combinator::{all_consuming, eof, not},
    error::context,
    sequence::{pair, preceded, separated_pair, terminated},
    IResult, Parser as _,
};

/// Takes an input string and returns a `ModelName`
pub fn parse_from<'a>(input: &'a str) -> IResult<&'a str, &'a str> {
    let from_tag = tag("FROM");
    let space = take_while1(|c| c == ' ');

    context("FROM", preceded(pair(from_tag, space), model_id)).parse(input)
}

fn model_id(input: &str) -> IResult<&str, &str> {
    nom::character::complete::not_line_ending(input)
}

#[derive(Debug, Clone)]
pub enum ModelId {
    VersionedModel { name: String, version: String },
    Path { path: PathBuf },
    Gguf { path: PathBuf },
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    const TEST_DATA_DIR: &str = "./test";

    fn load_modelfiles(test_dir: impl AsRef<Path>) -> Vec<String> {
        test_dir
            .as_ref()
            .read_dir()
            .expect("could not open test data dir")
            .map(|file| {
                let file = file.expect("error reading dir entry");
                file.path()
            })
            .filter(|path| path.ends_with("Modelfile"))
            .map(|path| std::fs::read_to_string(path).expect("could not read file contents"))
            .collect()
    }

    #[test]
    fn test_data_loads() {
        let _modelfiles: Vec<String> = load_modelfiles(TEST_DATA_DIR);
    }

    const TEST_FROM: &str = "FROM /mnt/space/ollama/models/blobs/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373 ";

    const TEST_MODEL_IDS: &[&str] = &[
        "/mnt/space/ollama/models/blobs/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373",
        "llama3.1:latest",
    ];

    #[test]
    fn from_field_is_parsed() {
        parse_from(TEST_FROM).expect("should be able to parse single example");
    }

    #[test]
    fn model_name_is_parsed() {
        fn parser(input: &str) -> IResult<&str, &str> {
            not_line_ending(input)
        }
        assert_eq!(parser("abc"), Ok(("", "abc")));
    }
}
