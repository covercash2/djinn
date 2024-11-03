//! Parse the [`Modelfile`] according to the [Modelfile spec].
//!
//!
//! TODO:
//! - [x] comments [`comment`]
//! - [x] FROM [`from`]
//! - [ ] PARAMETER
//! - [x] TEMPLATE [`template`]
//! - [ ] SYSTEM
//! - [ ] ADAPTER
//! - [ ] LICENSE
//! - [ ] MESSAGE
//! - [ ] case insensitivity
//!
//! [Modelfile spec]: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
use std::{path::PathBuf, str::FromStr};

use nom::{
    branch::alt,
    bytes::{
        self,
        streaming::{tag, take_until, take_while, take_while1},
    },
    character::{
        complete::{self, multispace1, space1},
        streaming,
    },
    combinator::{all_consuming, value},
    error::{context, Error, ErrorKind, ParseError},
    multi::many0,
    sequence::{delimited, pair, preceded, separated_pair, terminated},
    IResult, Parser as _,
};
use strum::{
    EnumDiscriminants, EnumIter, EnumString, IntoEnumIterator as _, IntoStaticStr, VariantArray,
};

const TRIPLE_QUOTES: &str = r#"""""#;
const SINGLE_QUOTE: &str = r#"""#;

/// Takes an input string and returns a `ModelName`.
/// Parses a line that starts with `FROM`
/// that specifies the [`ModelId`]
pub fn from(input: &str) -> IResult<&str, &str> {
    let from_tag = tag("FROM");
    let space = take_while1(|c| c == ' ');

    context("FROM", preceded(pair(from_tag, space), model_id)).parse(input)
}

fn model_id(input: &str) -> IResult<&str, &str> {
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
fn comment(input: &str) -> IResult<&str, ()> {
    let comment_delimeter = tag("#");

    context(
        "comment",
        value((), pair(comment_delimeter, complete::not_line_ending)),
    )
    .parse(input)
}

/// Consume empty lines and comments
fn skip_lines(input: &str) -> IResult<&str, ()> {
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
    let template = tag("TEMPLATE");
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
fn triple_quote_string(input: &str) -> IResult<&str, &str> {
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
fn single_quoted_multiline_string(input: &str) -> IResult<&str, &str> {
    delimited(
        tag(SINGLE_QUOTE),
        take_until(SINGLE_QUOTE),
        tag(SINGLE_QUOTE),
    )
    .parse(input)
}

/// https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
#[derive(EnumDiscriminants)]
#[strum_discriminants(name(ParameterName))]
#[strum_discriminants(derive(EnumIter, IntoStaticStr, EnumString, VariantArray))]
#[strum_discriminants(strum(serialize_all = "snake_case"))]
pub enum Parameter {
    /// Enable Mirostat sampling for controlling perplexity.
    /// (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
    Mirostat(usize),
    /// Influences how quickly the algorithm responds
    /// to feedback from the generated text.
    /// A lower learning rate will result in slower adjustments,
    /// while a higher learning rate will make the algorithm more responsive.
    /// (Default: 0.1)
    MirostatEta(f32),
    /// Controls the balance between coherence and diversity of the output.
    /// A lower value will result in more focused and coherent text.
    /// (Default: 5.0)
    MirostatTau(f32),
    /// Sets the size of the context window
    /// used to generate the next token.
    /// (Default: 2048)
    NumCtx(usize),
    /// Sets how far back for the model
    /// to look back to prevent repetition.
    /// (Default: 64, 0 = disabled, -1 = num_ctx)
    RepeatLastN(usize),
    /// Sets how strongly to penalize repetitions.
    /// A higher value (e.g., 1.5) will penalize repetitions more strongly,
    /// while a lower value (e.g., 0.9) will be more lenient.
    /// (Default: 1.1)
    RepeatPenalty(f32),
    /// The temperature of the model.
    /// Increasing the temperature will make the model answer more creatively.
    /// (Default: 0.8)
    Temperature(f32),
    /// Sets the random number seed to use for generation.
    /// Setting this to a specific number will make the model generate the same text
    /// for the same prompt.
    /// (Default: 0)
    Seed(usize),
    /// Sets the stop sequences to use.
    /// When this pattern is encountered the LLM will stop generating text and return.
    /// Multiple stop patterns may be set by specifying multiple separate stop parameters
    /// in a modelfile.
    Stop(String),
    /// Tail free sampling is used to reduce the impact
    /// of less probable tokens from the output.
    /// A higher value (e.g., 2.0) will reduce the impact more,
    /// while a value of 1.0 disables this setting.
    /// (default: 1)
    TfsZ(f32),
    /// Maximum number of tokens to predict when generating text.
    /// (Default: 128, -1 = infinite generation, -2 = fill context)
    NumPredict(usize),
    /// Reduces the probability of generating nonsense.
    /// A higher value (e.g. 100) will give more diverse answers,
    /// while a lower value (e.g. 10) will be more conservative.
    /// (Default: 40)
    TopK(usize),
    /// Works together with top-k.
    /// A higher value (e.g., 0.95) will lead to more diverse text,
    /// while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// (Default: 0.9)
    TopP(f32),
    /// Alternative to the top_p,
    /// and aims to ensure a balance of quality and variety.
    /// The parameter p represents the minimum probability for a token to be considered,
    /// relative to the probability of the most likely token.
    /// For example, with p=0.05 and the most likely token having a probability of 0.9,
    /// logits with a value less than 0.045 are filtered out.
    /// (Default: 0.0)
    MinP(f32),
}

fn parameter_name(input: &str) -> IResult<&str, ParameterName> {
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

fn float_parameter_value(input: &str) -> IResult<&str, f32> {
    nom::number::complete::float(input)
}

fn int_parameter_value(input: &str) -> IResult<&str, usize> {
    nom::character::complete::digit1(input).map(|(s, int)| {
        (
            s,
            int.parse().expect("should be able to parse int from input"),
        )
    })
}

fn string_parameter_value(input: &str) -> IResult<&str, &str> {
    complete::not_line_ending(input)
}

fn parameter(input: &str) -> IResult<&str, Parameter> {
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
fn parameter_line(input: &str) -> IResult<&str, Parameter> {
    let parameter_tag = tag("PARAMETER");

    dbg!(&input);

    preceded(pair(parameter_tag, multispace1), parameter).parse(input)
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

    const TEST_FROM: &str = "FROM /mnt/space/ollama/models/blobs/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373";

    const TEST_MODEL_IDS: &[&str] = &[
        "/mnt/space/ollama/models/blobs/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373",
        "llama3.1:latest",
    ];

    #[test]
    fn from_field_is_parsed() {
        from(TEST_FROM).expect("should be able to parse single example");
    }

    #[test]
    fn model_name_is_parsed() {
        for model in TEST_MODEL_IDS {
            model_id(model).expect("could not parse model ID");
        }
    }

    const TEST_COMMENTS: &[&str] = &[
        r#"# Modelfile generated by "ollama show""#,
        r#"# To build a new Modelfile based on this, replace FROM with:"#,
        r#"# FROM llama3.1:latest"#,
    ];

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

    const TEST_TRIPLE_QUOTES: &[&str] = &[r#""""here's some text
            in triple quotes
            we just need to consume it all
            and return it back
            """"#];

    #[test]
    fn triple_quotes_are_parsed() {
        for quote in TEST_TRIPLE_QUOTES {
            triple_quote_string(quote).expect("should be able to parse triple quoted strings");
        }
    }

    const TEST_SINGLE_QUOTE_MULTILINE: &[&str] = &[r#""here's some text
            in triple quotes
            we just need to consume it all
            and return it back
            ""#];

    #[test]
    fn single_quote_multiline_string_is_parsed() {
        for quote in TEST_SINGLE_QUOTE_MULTILINE {
            single_quoted_multiline_string(quote)
                .expect("should be able to parse triple quoted strings");
        }
    }

    const TEMPLATE_PREFIX: &str = "TEMPLATE ";

    #[test]
    fn test_template() {
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
}
