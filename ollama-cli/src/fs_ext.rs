use std::path::Path;

use crate::error::Result;

pub fn read_file_to_string(path: impl AsRef<Path>) -> Result<String> {
    std::fs::read_to_string(&path).map_err(|source| crate::error::Error::ReadFile {
        source,
        path: path.as_ref().into(),
    })
}
