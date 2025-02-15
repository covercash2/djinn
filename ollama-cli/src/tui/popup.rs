use std::{path::Path, sync::Arc};

use form_enter::{FormAction, FormEnter};
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
    fs_ext::{read_file_to_string, AppFileData},
    tui::event::InputMode,
};

use super::{
    event::{Action, ActionHandler, EventProcessor},
    widgets_ext::DrawViewModel,
    AppEvent,
};

pub mod form_enter;

#[derive(Debug, Clone)]
pub struct PopupViewModel {
    title: String,
    content: PopupContent,
}

#[derive(Clone, Debug)]
pub enum PopupContent {
    List(ListContent),
    Form(FormEnter),
}

impl PopupContent {
    pub fn line_count(&self) -> usize {
        match self {
            PopupContent::List(list_content) => list_content.line_count(),
            PopupContent::Form(_) => 2,
        }
    }
}

impl From<ListContent> for PopupContent {
    fn from(value: ListContent) -> Self {
        PopupContent::List(value)
    }
}

impl From<FormEnter> for PopupContent {
    fn from(value: FormEnter) -> Self {
        PopupContent::Form(value)
    }
}

impl ActionHandler for PopupContent {
    type Event = AppEvent;

    fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
        match self {
            PopupContent::List(list_content) => list_content.handle_action(action),
            PopupContent::Form(form_enter) => form_enter.handle_action(action),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ListContent {
    columns: Arc<[String]>,
    scroll_offset: u16,
}

impl<S: AsRef<str>> From<S> for ListContent {
    fn from(value: S) -> Self {
        ListContent {
            columns: [value.as_ref().to_string()].into(),
            scroll_offset: 0,
        }
    }
}

impl<S: AsRef<str>> FromIterator<S> for ListContent {
    fn from_iter<T: IntoIterator<Item = S>>(iter: T) -> Self {
        let columns: Arc<[String]> = iter.into_iter().map(|s| s.as_ref().into()).collect();

        Self {
            columns,
            scroll_offset: 0,
        }
    }
}

impl ListContent {
    pub fn line_count(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.lines().count())
            .max()
            .expect("should have a non-zero number of columns")
    }

    fn max_scroll(&self) -> u16 {
        (self.line_count())
            .try_into()
            .expect("should be able to fit popup content into u16")
    }
}

#[bon::bon]
impl PopupViewModel {
    pub fn new(title: impl ToString, content: impl Into<ListContent>) -> Self {
        PopupViewModel {
            title: title.to_string(),
            content: content.into().into(),
        }
    }

    #[builder]
    pub fn file_save(data: AppFileData, prefix: Option<String>) -> Self {
        PopupViewModel {
            title: "save file".to_string(),
            content: PopupContent::Form(FormEnter {
                title: "enter filename".into(),
                prefix: prefix.unwrap_or_default().into(),
                suffix: data.file_extension().into(),
                entry: "".into(),
                action: FormAction::SaveFile { data },
            }),
        }
    }

    pub fn log_popup(log_file: impl AsRef<Path>) -> Result<Self> {
        let logs = read_file_to_string(log_file)?;

        let content = logs.lines().rev().take(10).join("\n");

        Ok(PopupViewModel::new("logs".to_string(), content))
    }

    pub fn keymap_popup(event_processor: &EventProcessor) -> Self {
        let keymaps = &event_processor.definitions.0;
        let keymap_help: ListContent = InputMode::iter()
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
}

impl ActionHandler for ListContent {
    type Event = AppEvent;

    fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
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

impl ActionHandler for PopupViewModel {
    type Event = AppEvent;

    fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
        self.content.handle_action(action)
    }
}

fn popup_area(parent: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vertical = Layout::vertical([Constraint::Percentage(percent_y)]).flex(Flex::Center);
    let horizontal = Layout::horizontal([Constraint::Percentage(percent_x)]).flex(Flex::Center);

    let [area] = vertical.areas(parent);
    let [area] = horizontal.areas(area);

    area
}

impl DrawViewModel for ListContent {
    fn draw_view_model(&mut self, frame: &mut Frame<'_>, parent: Rect, style: Style) {
        let layout = Layout::horizontal(self.columns.iter().map(|_| Constraint::Fill(1)));

        for (i, column_area) in layout.split(parent).iter().enumerate() {
            let content = Paragraph::new(self.columns[i].as_str())
                .wrap(Wrap { trim: true })
                .scroll((self.scroll_offset, 0))
                .block(Block::new().padding(Padding::proportional(2)))
                .style(style);
            frame.render_widget(content, *column_area);
        }
    }
}

impl DrawViewModel for PopupViewModel {
    fn draw_view_model(&mut self, frame: &mut Frame<'_>, parent: Rect, style: Style) {
        let area = popup_area(parent, 60, 60);
        frame.render_widget(Clear, area);

        let block = Block::bordered().title(self.title.as_str());

        match &mut self.content {
            PopupContent::List(list_content) => list_content.draw_view_model(frame, area, style),
            PopupContent::Form(form_enter) => form_enter.draw_view_model(frame, area, style),
        }

        frame.render_widget(block, area);
    }
}

#[extend::ext(name = PopupView)]
pub impl<'a> Frame<'a> {
    fn popup(&mut self, parent: Rect, style: Style, view_model: &mut PopupViewModel) {
        view_model.draw_view_model(self, parent, style);
    }
}
