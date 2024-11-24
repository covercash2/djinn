use std::path::Path;

use itertools::Itertools;
use ratatui::{
    layout::{Constraint, Flex, Layout, Rect},
    style::Style,
    widgets::{Block, Clear, Paragraph, Wrap},
    Frame,
};

use crate::{
    error::{Error, Result},
    fs_ext::read_file_to_string,
};

#[derive(Debug, Clone)]
pub struct PopupViewModel {
    title: String,
    content: String,
}

impl PopupViewModel {
    pub fn new(title: impl ToString, content: impl ToString) -> Self {
        PopupViewModel {
            title: title.to_string(),
            content: content.to_string(),
        }
    }

    pub fn log_popup(log_file: impl AsRef<Path>) -> Result<Self> {
        let logs = read_file_to_string(log_file)?;

        let content = logs.lines().rev().take(10).join("\n");

        Ok(PopupViewModel {
            title: "logs".to_string(),
            content,
        })
    }
}

fn popup_area(parent: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vertical = Layout::vertical([Constraint::Percentage(percent_y)]).flex(Flex::Center);
    let horizontal = Layout::horizontal([Constraint::Percentage(percent_x)]).flex(Flex::Center);

    let [area] = vertical.areas(parent);
    let [area] = horizontal.areas(area);

    area
}

#[extend::ext(name = PopupView)]
pub impl<'a> Frame<'a> {
    fn popup(&mut self, parent: Rect, style: Style, view_model: &mut PopupViewModel) {
        let block = Block::bordered().title("popup");
        let content = Paragraph::new(view_model.content.as_str())
            .wrap(Wrap { trim: true })
            .block(block)
            .style(style);

        let area = popup_area(parent, 60, 60);
        self.render_widget(Clear, area);
        self.render_widget(content, area);
    }
}
