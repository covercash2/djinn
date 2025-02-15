use std::sync::Arc;

use ratatui::{
    layout::{Alignment, Constraint, Layout},
    widgets::{Block, Paragraph},
};

use crate::{
    cursor::StringCursor,
    error::Result,
    fs_ext::AppFileData,
    tui::{
        event::{Action, ActionHandler, InputMode},
        widgets_ext::DrawViewModel,
        AppEvent, BatchEvents,
    },
};

/// For stuff like entering filenames.
/// Just a single entry form for now.
#[derive(Clone, Debug)]
pub struct FormEnter {
    pub title: Arc<str>,
    pub prefix: Arc<str>,
    pub suffix: Arc<str>,
    pub action: FormAction,
    pub entry: StringCursor,
}

#[derive(Clone, Debug)]
pub enum FormAction {
    SaveFile { data: AppFileData },
}

impl ActionHandler for FormEnter {
    type Event = AppEvent;

    fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
        match &action {
            Action::Unhandled(c) => {
                self.entry = self.entry.push(*c);
            }
            Action::Beginning => self.entry = self.entry.reset(),
            Action::End => self.entry = self.entry.end(),
            Action::Left => self.entry = self.entry.prev(),
            Action::Right => {
                self.entry = self.entry.next();
            }
            Action::Backspace => {
                self.entry = self.entry.pop();
            }
            Action::Enter => match &self.action {
                FormAction::SaveFile { data } => {
                    if self.entry.as_ref().is_empty() {
                        return Ok(None);
                    }

                    let save_file_event = AppEvent::SaveFile {
                        data: data.clone(),
                        name: self.entry.as_ref().into(),
                    };

                    let events = BatchEvents::from_iter([
                        save_file_event,
                        AppEvent::Submit(crate::lm::Prompt::LocalModels),
                        AppEvent::InputMode(InputMode::Normal),
                        AppEvent::Deactivate,
                    ]);

                    return Ok(Some(AppEvent::Batch(events)));
                }
            },
            Action::Escape => {
                return Ok(Some(AppEvent::Deactivate));
            }
            Action::LeftWord => todo!(),
            Action::RightWord => todo!(),
            Action::Quit => todo!(),
            _ => {}
        }

        Ok(None)
    }
}

impl DrawViewModel for FormEnter {
    fn draw_view_model(
        &mut self,
        frame: &mut ratatui::Frame<'_>,
        parent: ratatui::prelude::Rect,
        style: ratatui::prelude::Style,
    ) {
        let layout = Layout::horizontal([Constraint::Max(16), Constraint::Percentage(80)]);
        let [prefix_area, entry_area] = layout.areas(parent);

        // let prefix = Text::from(self.prefix.as_ref());
        let prefix = Paragraph::new(self.prefix.as_ref())
            .block(Block::new())
            .style(style)
            .alignment(Alignment::Center);
        let mut entry = String::from(self.entry.as_ref());
        entry.push_str(&self.suffix);
        let entry = Paragraph::new(entry)
            .block(Block::new())
            .style(style)
            .alignment(Alignment::Center);

        frame.render_widget(prefix, prefix_area);
        frame.render_widget(entry, entry_area);
    }
}
