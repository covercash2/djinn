use std::{
    fs::{DirEntry, ReadDir},
    io::Write as _,
    path::{Path, PathBuf},
};

use modelfile::Modelfile;

use crate::error::Result;

#[derive(Clone, Debug)]
pub enum AppFileData {
    Modelfile(Modelfile),
}

impl AppFileData {
    pub fn file_extension(&self) -> &'static str {
        match self {
            AppFileData::Modelfile(_) => ".Modelfile",
        }
    }
}

impl From<Modelfile> for AppFileData {
    fn from(value: Modelfile) -> Self {
        AppFileData::Modelfile(value)
    }
}

pub fn read_file_to_string(path: impl AsRef<Path>) -> Result<String> {
    std::fs::read_to_string(&path).map_err(|source| crate::error::Error::ReadFile {
        source,
        path: path.as_ref().into(),
    })
}

pub fn save_file(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::options()
        .write(true)
        .create(true)
        .open(path)
        .map_err(|source| crate::error::Error::OpenFile {
            source,
            path: path.to_owned(),
        })?;

    file.write_all(contents.as_ref())
        .map_err(|source| crate::error::Error::WriteFile {
            source,
            path: path.to_owned(),
        })?;

    Ok(())
}

pub trait PathExt {
    fn create_dir_all(&self) -> Result<()>;
}

impl<T> PathExt for T
where
    T: AsRef<Path>,
{
    fn create_dir_all(&self) -> Result<()> {
        std::fs::create_dir_all(self.as_ref()).map_err(|source: std::io::Error| {
            crate::error::Error::CreateDir {
                source,
                path: self.as_ref().to_owned(),
            }
        })
    }
}

pub struct DirIter {
    iter: ReadDir,
    path: PathBuf,
}

impl Iterator for DirIter {
    type Item = Result<DirEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|r| {
            r.map_err(|source| crate::error::Error::ReadDir {
                source,
                path: self.path.clone(),
            })
        })
    }
}

pub trait FilesystemExt {
    fn dir(&self) -> Result<DirIter>;
}

impl<P: AsRef<Path>> FilesystemExt for P {
    fn dir(&self) -> Result<DirIter> {
        let iter = self
            .as_ref()
            .read_dir()
            .map_err(|source| crate::error::Error::ReadDir {
                source,
                path: self.as_ref().to_path_buf(),
            })?;

        Ok(DirIter {
            iter,
            path: self.as_ref().to_path_buf(),
        })
    }
}
