use std::sync::Arc;

use chrono::{DateTime, Utc};
use ollama_rs::models::LocalModel;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Row, Table, TableState},
    Frame,
};

use crate::{
    error::{Error, Result},
    lm::Response,
    ollama::ModelName,
    tui::event::Action,
};

use super::{u64Ext as _, ModelEvent};

#[derive(Clone, Debug, Default)]
pub struct ModelListViewModel {
    models: Vec<LocalModel>,
    widget_state: TableState,
}

impl ModelListViewModel {
    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        let Response::LocalModels(local_models) = response else {
            return Err(Error::UnexpectedResponse(response));
        };

        self.models = local_models;
        Ok(())
    }

    pub async fn handle_event(&mut self, action: Action) -> Result<Option<ModelEvent>> {
        match action {
            Action::Refresh => Ok(Some(ModelEvent::Refresh)),
            Action::Quit => Ok(Some(ModelEvent::Deactivate)),
            Action::Enter => {
                if let Some(model) = self
                    .widget_state
                    .selected()
                    .and_then(|index| self.models.get(index))
                {
                    let name: Arc<str> = model.name.clone().into();
                    Ok(Some(ModelEvent::GetInfo(ModelName(name))))
                } else {
                    Ok(None)
                }
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

#[extend::ext(name = ModelListView)]
pub impl<'a> Frame<'a> {
    fn model_list(&mut self, parent: Rect, style: Style, view_model: &mut ModelListViewModel) {
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
