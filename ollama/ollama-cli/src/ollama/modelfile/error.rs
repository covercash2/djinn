use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum ModelfileError {
    #[error("error building Modelfile from parts")]
    Builder(String),

    #[error("unable to parse Modelfile")]
    Parse(String),
}
