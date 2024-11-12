//! Parse the [`Modelfile`] according to the [Modelfile spec].
//!
//!
//! TODO:
//! - [x] comments [`comment`]
//! - [x] FROM [`from`]
//! - [x] PARAMETER
//! - [x] TEMPLATE [`template`]
//! - [x] SYSTEM
//! - [x] ADAPTER
//! - [x] LICENSE
//! - [ ] MESSAGE
//! - [x] case insensitivity
//!
//! [Modelfile spec]: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
use std::path::PathBuf;

use nom::{
    branch::alt,
    bytes::{
        complete::tag_no_case,
        streaming::{tag, take_until, take_while, take_while1},
    },
    character::complete::{self, multispace1, newline},
    combinator::{eof, value},
    error::context,
    multi::many0,
    sequence::{delimited, pair, preceded, terminated},
    IResult, Parser as _,
};

use crate::ollama::chat::{Message, MessageRole};

use super::{Parameter, ParameterName, TensorFile};

const TRIPLE_QUOTES: &str = r#"""""#;
const SINGLE_QUOTE: &str = r#"""#;
const SINGLE_QUOTE_NEWLINE: &str = r#""\n"#;

/// Takes an input string and returns a `ModelName`.
/// Parses a line that starts with `FROM`
/// that specifies the [`ModelId`]
pub fn from(input: &str) -> IResult<&str, &str> {
    let from_tag = tag_no_case("FROM");
    let space = take_while1(|c| c == ' ');

    context("FROM", preceded(pair(from_tag, space), model_id)).parse(input)
}

pub fn model_id(input: &str) -> IResult<&str, &str> {
    complete::not_line_ending(input)
}

#[derive(Debug, Clone)]
pub enum ModelId {
    VersionedModel { name: String, version: String },
    Path { path: PathBuf },
    Gguf { path: PathBuf },
}

/// Parse a comment line.
/// Comments start with a `#` and take a single line.
pub fn comment(input: &str) -> IResult<&str, ()> {
    let comment_delimeter = tag("#");

    context(
        "comment",
        value((), pair(comment_delimeter, complete::not_line_ending)),
    )
    .parse(input)
}

/// Consume empty lines and comments
pub fn skip_lines(input: &str) -> IResult<&str, ()> {
    context("skip_lines", value((), many0(comment))).parse(input)
}

/// Parse a model template.
/// The template is preceded by the `TEMPLATE` keyword
/// and is formatted like a go template`*`
///
/// `*` the docs are actually super weird about this.
/// They say in [the spec]:
/// > Note: syntax may be model specific
/// Good thing the authors are geniuses,
/// otherwise this would be totally unhinged.
///
/// We're not going to worry about parsing the template
/// in this project.
/// Leave that to the geniuses at Ollama.
///
/// [the spec]: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template
fn template(input: &str) -> IResult<&str, &str> {
    let template = tag_no_case("TEMPLATE");
    let space = take_while(|c| c == ' ');
    preceded(
        pair(template, space),
        alt((triple_quote_string, single_quoted_multiline_string)),
    )
    .parse(input)
}

/// A string surrounded by trippled quotes.
/// """Like this!
/// And they can be on multiple lines."""
pub fn triple_quote_string(input: &str) -> IResult<&str, &str> {
    delimited(
        tag(TRIPLE_QUOTES),
        take_until(TRIPLE_QUOTES),
        tag(TRIPLE_QUOTES),
    )
    .parse(input)
}

/// A multiline string with single quotes.
/// Why is this allowed?
/// The inmates are running the asylum.
pub fn single_quoted_multiline_string(input: &str) -> IResult<&str, &str> {
    delimited(
        tag(SINGLE_QUOTE),
        take_until(SINGLE_QUOTE),
        tag(SINGLE_QUOTE),
    )
    .parse(input)
}

pub fn parameter_name(input: &str) -> IResult<&str, ParameterName> {
    dbg!(input);
    alt((
        terminated(
            tag::<&str, &str, _>(ParameterName::Mirostat.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::MirostatEta.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::MirostatTau.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::NumCtx.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::RepeatLastN.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::RepeatPenalty.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::Seed.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::Temperature.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::Stop.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::TfsZ.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::NumPredict.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::TopK.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::TopP.into()),
            multispace1,
        ),
        terminated(
            tag::<&str, &str, _>(ParameterName::MinP.into()),
            multispace1,
        ),
    ))(input)
    .map(|(rest, name)| {
        (
            rest,
            name.parse::<ParameterName>()
                .expect("should be able to parse parameter name from parse parameter name"),
        )
    })
}

pub fn float_parameter_value(input: &str) -> IResult<&str, f32> {
    nom::number::complete::float(input)
}

pub fn int_parameter_value(input: &str) -> IResult<&str, usize> {
    nom::character::complete::digit1(input).map(|(s, int)| {
        (
            s,
            int.parse().expect("should be able to parse int from input"),
        )
    })
}

pub fn string_parameter_value(input: &str) -> IResult<&str, &str> {
    complete::not_line_ending(input)
}

pub fn parameter(input: &str) -> IResult<&str, Parameter> {
    dbg!(&input);
    let (input, name) = parameter_name(input)?;
    dbg!(name);
    match name {
        ParameterName::Mirostat => {
            int_parameter_value(input).map(|(rest, int)| (rest, Parameter::Mirostat(int)))
        }
        ParameterName::MirostatEta => {
            float_parameter_value(input).map(|(rest, value)| (rest, Parameter::MirostatEta(value)))
        }
        ParameterName::MirostatTau => {
            float_parameter_value(input).map(|(rest, value)| (rest, Parameter::MirostatTau(value)))
        }
        ParameterName::NumCtx => {
            int_parameter_value(input).map(|(rest, value)| (rest, Parameter::NumCtx(value)))
        }
        ParameterName::RepeatLastN => {
            int_parameter_value(input).map(|(rest, value)| (rest, Parameter::RepeatLastN(value)))
        }
        ParameterName::RepeatPenalty => float_parameter_value(input)
            .map(|(rest, value)| (rest, Parameter::RepeatPenalty(value))),
        ParameterName::Temperature => {
            float_parameter_value(input).map(|(rest, value)| (rest, Parameter::Temperature(value)))
        }
        ParameterName::Seed => {
            int_parameter_value(input).map(|(rest, value)| (rest, Parameter::Seed(value)))
        }
        ParameterName::Stop => string_parameter_value(input)
            .map(|(rest, value)| (rest, Parameter::Stop(value.to_string()))),
        ParameterName::TfsZ => {
            float_parameter_value(input).map(|(rest, value)| (rest, Parameter::TfsZ(value)))
        }
        ParameterName::NumPredict => {
            int_parameter_value(input).map(|(rest, value)| (rest, Parameter::NumPredict(value)))
        }
        ParameterName::TopK => {
            int_parameter_value(input).map(|(rest, value)| (rest, Parameter::TopK(value)))
        }
        ParameterName::TopP => {
            float_parameter_value(input).map(|(rest, value)| (rest, Parameter::TopP(value)))
        }
        ParameterName::MinP => {
            float_parameter_value(input).map(|(rest, value)| (rest, Parameter::MinP(value)))
        }
    }
}

/// Parameters are key value pairs
/// of a name from a set of predefined keys
/// to an arbitrary string value
///
/// https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
pub fn parameter_line(input: &str) -> IResult<&str, Parameter> {
    let parameter_tag = tag_no_case("PARAMETER");

    dbg!(&input);

    preceded(pair(parameter_tag, multispace1), parameter).parse(input)
}

pub fn multiline(input: &str) -> IResult<&str, &str> {
    alt((
        triple_quote_string,
        single_quoted_multiline_string,
        complete::not_line_ending,
    ))
    .parse(input)
}

/// The system message to be used in the template, if applicable
///
/// https://github.com/ollama/ollama/blob/main/docs/modelfile.md#system
pub fn system(input: &str) -> IResult<&str, &str> {
    let template = tag_no_case("system");
    preceded(pair(template, multispace1), multiline).parse(input)
}

/// Takes an input string and returns a `ModelName`.
/// Parses a line that starts with `FROM`
/// that specifies the [`ModelId`]
pub fn adapter(input: &str) -> IResult<&str, TensorFile> {
    let adapter_tag = tag_no_case("adapter");

    context(
        "ADAPTER",
        preceded(pair(adapter_tag, multispace1), tensor_file),
    )
    .parse(input)
}

pub fn tensor_file(input: &str) -> IResult<&str, TensorFile> {
    filename
        .map(|filename| {
            if filename.ends_with(".gguf") {
                TensorFile::Gguf(PathBuf::from(filename))
            } else if filename.ends_with(".safetensors") {
                TensorFile::Safetensor(PathBuf::from(filename))
            } else {
                panic!("bad file extensions");
            }
        })
        .parse(input)
}

pub fn filename(input: &str) -> IResult<&str, &str> {
    nom::bytes::complete::take_while1(is_file_char).parse(input)
}

pub fn is_file_char(c: char) -> bool {
    c.is_alphanumeric() || c == '/' || c == '.' || c == '-' || c == '_'
}

pub fn license(input: &str) -> IResult<&str, &str> {
    let license_tag = tag_no_case("license");
    preceded(pair(license_tag, multispace1), multiline)(input)
}

pub fn message(input: &str) -> IResult<&str, Message> {
    let message_tag = tag_no_case("message");
    let user_tag = tag("user");
    let assistant_tag = tag("assistant");
    let system_tag = tag("system");
    let role = alt((user_tag, assistant_tag, system_tag));

    preceded(pair(message_tag, multispace1), pair(role, multiline))
        .map(|(role, message)| {
            let role: MessageRole = role.parse().expect("should be able to parse role from tag");
            (role, message).into()
        })
        .parse(input)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use strum::IntoEnumIterator;

    use crate::ollama::modelfile::tests::{TEST_COMMENTS, TEST_FROM, TEST_MODEL_IDS, TEST_SINGLE_QUOTE_MULTILINE, TEST_TRIPLE_QUOTES};

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

    #[test]
    fn from_field_is_parsed() {
        for case in TEST_FROM {
            dbg!(case);
            from(case).expect("should be able to parse single example");
        }
    }

    #[test]
    fn model_name_is_parsed() {
        for model in TEST_MODEL_IDS {
            model_id(model).expect("could not parse model ID");
        }
    }

    #[test]
    fn comments_are_parsed() {
        for comment_string in TEST_COMMENTS {
            comment(comment_string).expect("could not parse comments");
        }

        let long_comment = TEST_COMMENTS.iter().fold(String::new(), |mut acc, line| {
            acc.push_str(line);
            acc.push('\n');
            acc.push('\n');
            acc
        });

        dbg!(&long_comment);

        skip_lines(long_comment.as_str()).expect("should skip all comment lines");
    }

    #[test]
    fn triple_quotes_are_parsed() {
        for quote in TEST_TRIPLE_QUOTES {
            triple_quote_string(quote).expect("should be able to parse triple quoted strings");
        }
    }

    #[test]
    fn single_quote_multiline_string_is_parsed() {
        for quote in TEST_SINGLE_QUOTE_MULTILINE {
            single_quoted_multiline_string(quote)
                .expect("should be able to parse triple quoted strings");
        }
    }

    const TEMPLATE_PREFIX: &str = "TEMPLATE ";

    #[test]
    fn templates_are_parsed() {
        let test_cases: Vec<String> = TEST_TRIPLE_QUOTES
            .iter()
            .chain(TEST_SINGLE_QUOTE_MULTILINE)
            .map(|test_case| format!("{TEMPLATE_PREFIX}{test_case}"))
            .collect();

        for test_case in test_cases {
            dbg!(&test_case);
            template(test_case.as_str()).expect("should be able to parse template");
        }
    }

    #[test]
    fn parameter_names_are_parsed() {
        let names: Vec<String> = ParameterName::iter()
            .map(|s| {
                let s: &str = s.into();
                format!("{s} ")
            })
            .collect();

        for name in names {
            dbg!(&name);
            parameter_name(&name).expect("should be able to parse parameter names");
        }
    }

    #[test]
    fn parameters_are_parsed() {
        let test_data = include_str!("./testdata/parameters.txt");
        for line in test_data.lines() {
            dbg!(&line);
            parameter_line(line).expect("should be able to parse parameter line");
        }
    }

    #[test]
    fn system_messages_are_parsed() {
        let test_data = include_str!("./testdata/systems.txt");
        for message in test_data.split(":endcase\n").filter(|s| !s.is_empty()) {
            dbg!(&message);
            system(message).expect("should be able to parse system message");
        }
    }

    #[test]
    fn adapaters_are_parsed() {
        let test_data = include_str!("./testdata/adapters.txt");
        for test_case in test_data.lines() {
            dbg!(test_case);
            adapter(test_case).expect("should be able to parse adapter");
        }
    }

    #[test]
    fn licenses_are_parsed() {
        let test_data = include_str!("./testdata/licenses.txt");
        for test_case in test_data.split(":endcase\n").filter(|s| !s.is_empty()) {
            dbg!(&test_case);
            license(test_case).expect("should be able to parse license");
        }
    }

    #[test]
    fn messages_are_parsed() {
        let test_data = include_str!("./testdata/messages.txt");
        for case in test_data.lines() {
            dbg!(&case);
            message(case).expect("should be able to parse a single message");
        }
    }
}
