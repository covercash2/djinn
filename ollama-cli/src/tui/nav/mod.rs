use std::sync::Arc;

use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Style, Stylize as _},
    text::Text,
    widgets::{Block, List, ListState, Padding},
    Frame,
};
use strum::VariantNames;

use super::{event::Action, AppEvent, ViewName};
use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct NavViewModel {
    views: Arc<[&'static str]>,
    list_state: ListState,
}

impl Default for NavViewModel {
    fn default() -> Self {
        let selected = 0;
        let list_state = ListState::default().with_selected(Some(selected));
        let views: Arc<[&'static str]> = ViewName::VARIANTS
            .iter()
            .filter(|view_name| **view_name != "nav")
            .copied()
            .collect();

        Self { views, list_state }
    }
}

impl NavViewModel {
    pub fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
        match action {
            Action::Up => {
                self.list_state.select_previous();
                Ok(None)
            }
            Action::Down => {
                self.list_state.select_next();
                Ok(None)
            }
            Action::Enter => {
                let Some(selected) = self.list_state.selected() else {
                    return Ok(None);
                };
                self.views
                    .get(selected)
                    .ok_or(Error::BadIndex {
                        index: selected,
                        msg: "unable to index view",
                    })
                    .and_then(|view_str| view_str.parse().map_err(|_| Error::ViewParse(view_str)))
                    .map(|view| Some(AppEvent::Activate(view)))
            }
            Action::Quit => Ok(Some(AppEvent::Quit)),
            _ => Ok(None),
        }
    }
}

#[extend::ext(name = NavView)]
pub impl<'a> Frame<'a> {
    fn nav_view(&mut self, parent: Rect, style: Style, view_model: &mut NavViewModel) {
        let list = List::from_iter(
            view_model
                .views
                .iter()
                .copied()
                .map(|name| Text::from(name).alignment(Alignment::Center)),
        )
        .style(style)
        .highlight_style(
            style
                .fg(style.bg.unwrap_or(Color::Black))
                .bg(style.fg.unwrap_or(Color::White)),
        )
        .block(
            Block::bordered()
                .padding(Padding::proportional(5))
                .title("Ollama control panel")
                .title_style(Style::default().bold().underlined().italic())
                .title_alignment(Alignment::Center),
        );

        self.render_stateful_widget_ref(list, parent, &mut view_model.list_state);
    }
}
