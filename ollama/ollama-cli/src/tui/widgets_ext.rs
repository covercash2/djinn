use std::borrow::Cow;

use ratatui::layout::Rect;

#[extend::ext]
pub impl Rect {
    fn wrap_inside<'a>(&self, text: &'a str) -> Vec<Cow<'a, str>> {
        let max_width = self.width - 2;
        let wrap = textwrap::Options::new(max_width.into());

        textwrap::wrap(text, wrap)
    }
}
