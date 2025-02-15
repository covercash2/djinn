/// A [`String`] builder with a [`Cursor`].
/// Useful for creating user input widgets.
#[derive(Debug, Clone)]
pub struct StringCursor {
    data: String,
    cursor: CursorInner,
}

impl StringCursor {
    pub fn new(initial_data: impl ToString) -> Self {
        let data = initial_data.to_string();
        let cursor: CursorInner = data.as_str().into();
        Self { data, cursor }
    }

    pub fn push(&self, c: char) -> Self {
        let StringCursor { mut data, cursor } = self.clone();
        data.push(c);
        let cursor = cursor.next();

        StringCursor { data, cursor }
    }

    pub fn pop(&self) -> Self {
        let StringCursor { mut data, cursor } = self.clone();
        let cursor = if let Some(c) = data.pop() {
            tracing::debug!(%c, "popped from the cursor");
            cursor.prev()
        } else {
            tracing::debug!("tried to pop from an empty cursor");
            cursor
        };

        StringCursor { data, cursor }
    }

    pub fn next(&self) -> Self {
        StringCursor {
            cursor: self.cursor.next(),
            ..self.clone()
        }
    }

    pub fn end(&self) -> Self {
        StringCursor {
            cursor: self.cursor.end(),
            ..self.clone()
        }
    }

    pub fn prev(&self) -> Self {
        StringCursor {
            cursor: self.cursor.prev(),
            ..self.clone()
        }
    }

    pub fn reset(&self) -> Self {
        StringCursor {
            cursor: self.cursor.reset(),
            ..self.clone()
        }
    }
}

impl Default for StringCursor {
    fn default() -> Self {
        Self::new(String::default())
    }
}

impl<T> From<T> for StringCursor
where
    T: ToString,
{
    fn from(value: T) -> Self {
        Self::new(value.to_string())
    }
}

impl AsRef<str> for StringCursor {
    fn as_ref(&self) -> &str {
        self.data.as_str()
    }
}

/// A simple integer cursor for keeping track
/// of the positional state of a list.
#[derive(Debug, Clone)]
struct CursorInner {
    /// Index into a list.
    index: usize,
    /// List size.
    size: usize,
    /// Max index
    max: usize,
}

impl CursorInner {
    fn seek(&self, index: usize) -> CursorInner {
        CursorInner {
            index,
            ..self.clone()
        }
    }

    /// Derive a cursor from this one at index 0.
    pub fn reset(&self) -> CursorInner {
        CursorInner {
            index: 0,
            ..self.clone()
        }
    }

    pub fn end(&self) -> CursorInner {
        CursorInner {
            index: self.max,
            ..self.clone()
        }
    }

    pub fn next(&self) -> CursorInner {
        let new_index = if self.index >= self.max {
            self.max
        } else {
            self.index.saturating_add(1)
        };
        self.seek(new_index)
    }

    pub fn prev(&self) -> CursorInner {
        self.seek(self.index.saturating_sub(1))
    }
}

impl<T> From<T> for CursorInner
where
    T: CollectionExt,
{
    fn from(value: T) -> Self {
        let size = value.size();
        let max = size.saturating_sub(1);

        CursorInner {
            index: 0,
            size,
            max,
        }
    }
}

pub trait CollectionExt {
    fn size(&self) -> usize;
}

impl<T> CollectionExt for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl CollectionExt for &str {
    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_cursor_from_vec() {
        let vec = vec![1, 2, 3];

        let cursor: CursorInner = vec.into();

        assert_eq!(cursor.max, 2);
        assert_eq!(cursor.index, 0);
    }

    #[test]
    fn maintain_max_index_on_max_increment() {
        let vec = vec!["one"];

        let cursor: CursorInner = vec.into();
        let cursor = cursor.next();
        let cursor = cursor.next();

        assert_eq!(cursor.index, 0);
        assert_eq!(cursor.max, 0);
    }
}
