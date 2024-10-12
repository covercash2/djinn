use chrono::{DateTime, Utc};
use crossterm::{event::Event, style::Stylize as _};
use ollama_rs::models::{LocalModel, ModelInfo};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Row, Table, TableState},
    Frame,
};

use crate::{
    error::{Error, Result},
    lm::{Prompt, Response},
};

use super::{
    event::Action,
    messages::{widget::List, widget_item::ListItem, widget_state::ListState},
    AppEvent,
};

#[derive(Clone, Debug, Default)]
pub struct ModelsViewModel {
    models: Vec<LocalModel>,
    active_pane: Option<Pane>,
    widget_state: TableState,
}

impl ModelsViewModel {
    pub fn new(models: Vec<LocalModel>) -> Self {
        ModelsViewModel {
            models,
            active_pane: None,
            widget_state: Default::default(),
        }
    }

    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        let Response::LocalModels(local_models) = response else {
            return Err(Error::UnexpectedResponse(response));
        };

        self.models = local_models;
        Ok(())
    }

    pub async fn handle_event(&mut self, action: Action) -> Result<Option<AppEvent>> {
        match action {
            Action::Refresh => Ok(Some(AppEvent::Submit(Prompt::LocalModels))),
            Action::Quit => Ok(Some(AppEvent::Quit)),
            Action::Enter => {
                self.widget_state.select(Some(0));
                Ok(None)
            }
            Action::Down => {
                self.widget_state.select_next();
                Ok(None)
            }
            Action::Up => {
                self.widget_state.select_previous();
                Ok(None)
            }
            _ => Ok(None),
        }
    }
}

#[derive(Clone, Debug)]
enum Pane {
    ModelInfo,
}

#[extend::ext(name = ModelsView)]
pub impl<'a> Frame<'a> {
    fn models_view(&mut self, parent: Rect, style: Style, view_model: &mut ModelsViewModel) {
        let vertical = Layout::vertical([Constraint::Min(1)]);

        let [model_info_area] = vertical.areas(parent);

        let models: Table = view_model
            .models
            .iter()
            .map(|info| {
                let name = Span::from(info.name.as_str());
                let size = Span::from(info.size.fit_to_bytesize());
                let last_modified: DateTime<Utc> = info
                    .modified_at
                    .parse()
                    .expect("could not parse datetime from ollama");
                let last_modified = Span::from(last_modified.format("%a %v %T").to_string());
                Row::from_iter([name, size, last_modified])
            })
            .collect();

        let list = models
            .block(Block::bordered())
            .style(style)
            .highlight_style(
                style
                    .fg(style.bg.unwrap_or(Color::Black))
                    .bg(style.fg.unwrap_or(Color::White)),
            )
            .highlight_symbol(">>");

        self.render_stateful_widget(list, model_info_area, &mut view_model.widget_state);
    }
}

const BYTE_UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB"];

#[extend::ext]
pub impl u64 {
    fn to_kib(&self) -> String {
        let value = self.div_ceil(1024);
        format!("{value}KiB")
    }

    fn to_mib(&self) -> String {
        let value = self.div_ceil(1024 ^ 2);
        format!("{value}MiB")
    }

    fn to_gib(&self) -> String {
        let value = self.div_ceil(1024 ^ 3);
        format!("{value}GiB")
    }

    fn fit_to_bytesize(&self) -> String {
        let (value, order) = std::iter::repeat(())
            .enumerate()
            .map(|(i, ())| i)
            .scan(*self, |state, i| {
                if *state >= 1024 {
                    *state /= 1024;
                    Some((*state, i + 1))
                } else {
                    None
                }
            })
            .last()
            .unwrap_or((*self, 0));

        let unit = BYTE_UNITS[order];

        format!("{value}{unit}")
    }
}
