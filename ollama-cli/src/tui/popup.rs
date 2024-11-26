use std::{path::Path, sync::Arc};

use itertools::Itertools;
use ratatui::{
    layout::{Constraint, Flex, Layout, Rect},
    style::Style,
    widgets::{Block, Clear, Padding, Paragraph, Wrap},
    Frame,
};
use strum::IntoEnumIterator as _;

use crate::{
    error::{Error, Result},
    fs_ext::read_file_to_string,
    tui::event::InputMode,
};

use super::{
    event::{Action, EventProcessor},
    AppEvent,
};

#[derive(Debug, Clone)]
pub struct PopupViewModel {
    title: String,
    content: PopupContent,
    scroll_offset: u16,
}

#[derive(Debug, Clone)]
pub struct PopupContent {
    columns: Arc<[String]>,
}

impl<S: AsRef<str>> From<S> for PopupContent {
    fn from(value: S) -> Self {
        PopupContent {
            columns: [value.as_ref().to_string()].into(),
        }
    }
}

impl<S: AsRef<str>> FromIterator<S> for PopupContent {
    fn from_iter<T: IntoIterator<Item = S>>(iter: T) -> Self {
        let columns: Arc<[String]> = iter.into_iter().map(|s| s.as_ref().into()).collect();

        Self { columns }
    }
}

impl PopupContent {
    pub fn line_count(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.lines().count())
            .max()
            .expect("should have a non-zero number of columns")
    }
}

impl PopupViewModel {
    pub fn new(title: impl ToString, content: impl Into<PopupContent>) -> Self {
        PopupViewModel {
            title: title.to_string(),
            content: content.into(),
            scroll_offset: 0,
        }
    }

    pub fn log_popup(log_file: impl AsRef<Path>) -> Result<Self> {
        let logs = read_file_to_string(log_file)?;

        let content = logs.lines().rev().take(10).join("\n");

        Ok(PopupViewModel::new("logs".to_string(), content))
    }

    pub fn keymap_popup(event_processor: &EventProcessor) -> Self {
        let keymaps = &event_processor.definitions.0;
        let keymap_help: PopupContent = InputMode::iter()
            .map(|mode| {
                let keymap = keymaps
                    .get(&mode)
                    .ok_or(Error::MissingKeymap(mode))
                    .expect("should be able to get the keymap");

                std::iter::once(mode.to_string())
                    .chain(
                        keymap
                            .0
                            .iter()
                            .map(|(key, action)| format!("{key}: {action}")),
                    )
                    .join("\n")
            })
            .collect();

        PopupViewModel::new("help", keymap_help)
    }

    fn max_scroll(&self) -> u16 {
        (self.content.line_count())
            .try_into()
            .expect("should be able to fit popup content into u16")
    }

    pub fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
        match action {
            Action::Up => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
                Ok(None)
            }
            Action::Down => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
                if self.scroll_offset > self.max_scroll() {
                    self.scroll_offset = self.max_scroll();
                }
                Ok(None)
            }
            Action::Beginning => {
                self.scroll_offset = 0;
                Ok(None)
            }
            Action::End => {
                self.scroll_offset = self.max_scroll();
                Ok(None)
            }
            Action::Popup | Action::Help | Action::Quit | Action::Enter | Action::Escape => {
                Ok(Some(AppEvent::Deactivate))
            }
            _ => Ok(None),
        }
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
        let area = popup_area(parent, 60, 60);
        self.render_widget(Clear, area);

        let block = Block::bordered().title(view_model.title.as_str());

        let layout = Layout::horizontal(
            view_model
                .content
                .columns
                .iter()
                .map(|_| Constraint::Fill(1)),
        );

        for (i, column_area) in layout.split(area).iter().enumerate() {
            let content = Paragraph::new(view_model.content.columns[i].as_str())
                .wrap(Wrap { trim: true })
                .scroll((view_model.scroll_offset, 0))
                .block(Block::new().padding(Padding::proportional(2)))
                .style(style);
            self.render_widget(content, *column_area);
        }

        self.render_widget(block, area);
    }
}
