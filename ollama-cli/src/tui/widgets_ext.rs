use std::borrow::Cow;

use ratatui::{layout::Rect, style::Style, Frame};

#[extend::ext]
pub impl Rect {
    fn wrap_inside<'a>(&self, text: &'a str) -> Vec<Cow<'a, str>> {
        let max_width = self.width - 2;
        let wrap = textwrap::Options::new(max_width.into());

        textwrap::wrap(text, wrap)
    }
}

pub trait DrawViewModel {
    fn draw_view_model(&mut self, frame: &mut Frame<'_>, parent: Rect, style: Style);
}
