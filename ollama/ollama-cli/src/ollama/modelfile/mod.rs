//! Ollama is developed by genius snowflakes
//! who know more about serialized configuration file formats
//! than everyone else who has come before them.
//! Thus, it was necessary for them to create a bespoke file format
//! to define a derived model.
//! This is an attempt to parse that genius file format in Rust.

use std::{path::PathBuf, str::FromStr};

use nom::instructions;
use strum::{EnumDiscriminants, EnumIter, EnumString, IntoStaticStr, VariantArray};

use super::chat::Message;

mod nom;

#[cfg(test)]
pub mod test_data;

#[derive(Clone, Debug)]
pub struct Modelfile {
    from: String,
    parameters: Vec<Parameter>,
    template: Option<String>,
    system: Option<String>,
    adapter: Option<TensorFile>,
    license: Option<String>,
    messages: Vec<Message>,
}

impl FromStr for Modelfile {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let instructions: Vec<Instruction> = instructions(input)
            .map_err(|error| error.to_string())
            .and_then(|(rest, instructions)| {
                if rest.is_empty() {
                    Ok(instructions)
                } else {
                    Err("parser did not consume all input".into())
                }
            })?;

        let mut modelfile = ModelfileBuilder::default();

        for instruction in instructions {
            let _ = match instruction {
                Instruction::From(model) => modelfile.from(model)?,
                Instruction::Parameter(parameter) => modelfile.parameter(parameter),
                Instruction::Template(template) => modelfile.template(template)?,
                Instruction::System(system) => modelfile.system(system)?,
                Instruction::Adapter(tensor_file) => modelfile.adapter(tensor_file)?,
                Instruction::License(license) => modelfile.license(license),
                Instruction::Message(message) => modelfile.message(message),
                Instruction::Skip => &mut modelfile,
            };
        }

        modelfile.build()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModelfileBuilder {
    from: Option<String>,
    parameters: Vec<Parameter>,
    template: Option<String>,
    system: Option<String>,
    adapter: Option<TensorFile>,
    license: Option<String>,
    messages: Vec<Message>,
}

impl ModelfileBuilder {
    pub fn build(self) -> Result<Modelfile, String> {
        let ModelfileBuilder {
            from,
            parameters,
            template,
            system,
            adapter,
            license,
            messages,
        } = self;
        if let Some(from) = from {
            Ok(Modelfile {
                from,
                parameters,
                template,
                system,
                adapter,
                license,
                messages,
            })
        } else {
            Err("Modelfile requires a FROM instruction".into())
        }
    }

    pub fn from(&mut self, input: String) -> Result<&mut Self, String> {
        if self.from.is_some() {
            Err(format!(
                "Modelfile can only have one FROM instruction: {}",
                input
            ))
        } else {
            self.from = Some(input);
            Ok(self)
        }
    }

    pub fn parameter(&mut self, parameter: Parameter) -> &mut Self {
        self.parameters.push(parameter);
        self
    }

    pub fn template(&mut self, template: String) -> Result<&mut Self, String> {
        if self.template.is_some() {
            Err(format!(
                "Modelfile can only have one TEMPLATE instruction: {}",
                template
            ))
        } else {
            self.template = Some(template);
            Ok(self)
        }
    }

    pub fn system(&mut self, system: String) -> Result<&mut Self, String> {
        if self.system.is_some() {
            Err(format!(
                "Modelfile can only have one SYSTEM instruction: {}",
                system,
            ))
        } else {
            self.system = Some(system);
            Ok(self)
        }
    }

    pub fn adapter(&mut self, adapter: TensorFile) -> Result<&mut Self, String> {
        if self.adapter.is_some() {
            Err(format!(
                "Modelfile can only have one ADAPTER instruction: {:?}",
                adapter,
            ))
        } else {
            self.adapter = Some(adapter);
            Ok(self)
        }
    }

    pub fn license(&mut self, license: String) -> &mut Self {
        if let Some(existing) = &self.license {
            self.license = Some(format!("{existing}\n{license}"));
        } else {
            self.license = Some(license);
        }

        self
    }

    pub fn message(&mut self, message: Message) -> &mut Self {
        self.messages.push(message);
        self
    }
}

pub enum Instruction {
    Skip,
    From(String),
    Parameter(Parameter),
    Template(String),
    System(String),
    Adapter(TensorFile),
    License(String),
    Message(Message),
}

impl From<Parameter> for Instruction {
    fn from(value: Parameter) -> Self {
        Instruction::Parameter(value)
    }
}

impl From<TensorFile> for Instruction {
    fn from(value: TensorFile) -> Self {
        Instruction::Adapter(value)
    }
}

impl From<Message> for Instruction {
    fn from(value: Message) -> Self {
        Instruction::Message(value)
    }
}

#[derive(Debug, Clone)]
pub enum TensorFile {
    Gguf(PathBuf),
    Safetensor(PathBuf),
}

/// https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
#[derive(Debug, Clone, EnumDiscriminants)]
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

#[cfg(test)]
mod tests {
    use test_data::{load_modelfiles, TEST_DATA_DIR};

    use super::*;

    #[test]
    fn modelfiles_are_parsed() {
        let modelfiles: Vec<(PathBuf, String)> = load_modelfiles(TEST_DATA_DIR);

        for (path, case) in modelfiles {
            dbg!(&path);
            let modelfile: Modelfile = case
                .parse::<Modelfile>()
                .expect("should be able to parse Modelfile");

            dbg!(modelfile);
        }
    }
}
