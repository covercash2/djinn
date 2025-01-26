use std::sync::Arc;

use chrono::{DateTime, Utc};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::Span,
    widgets::{Block, Row, Table, TableState},
    Frame,
};

use crate::{
    error::Result,
    model_definition::ModelDefinition,
    ollama::ModelName,
    tui::{event::Action, ResponseEvent},
};

use super::{u64Ext as _, ModelEvent};

const DEFAULT_TIME_FORMAT: &str = "%a %v %T";

#[derive(Clone, Debug, Default)]
pub struct ModelListViewModel {
    models: Vec<ModelDefinition>,
    widget_state: TableState,
}

impl ModelListViewModel {
    pub fn handle_response_event(&mut self, event: ResponseEvent) -> Result<()> {
        match event {
            ResponseEvent::UpdatedModels(definitions) => {
                self.models = definitions;
                Ok(())
            }
            event => {
                tracing::warn!(?event, "unexpected event",);
                Ok(())
            }
        }
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
                    let name: Arc<str> = model.name().clone().into();
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

impl ModelDefinition {
    pub fn name(&self) -> String {
        match self {
            ModelDefinition::OllamaRemote(local_model) => local_model.name.as_str().into(),
            ModelDefinition::LocalCache(local_modelfile) => local_modelfile
                .path
                .file_name()
                .map(std::ffi::OsStr::to_string_lossy)
                .unwrap_or("???".into())
                .into(),
            ModelDefinition::Synced { remote, local: _ } => remote.name.as_str().into(),
        }
    }

    pub fn size(&self) -> String {
        match self {
            ModelDefinition::OllamaRemote(local_model) => local_model.size.fit_to_bytesize().into(),
            ModelDefinition::LocalCache(_) => "NA".into(),
            ModelDefinition::Synced { remote, local: _ } => remote.size.fit_to_bytesize().into(),
        }
    }

    pub fn modified_at(&self) -> String {
        match self {
            ModelDefinition::OllamaRemote(local_model) => {
                let last_modified: DateTime<Utc> = local_model
                    .modified_at
                    .parse()
                    .expect("could not parse datetime from ollama");
                last_modified.format(DEFAULT_TIME_FORMAT).to_string().into()
            }
            ModelDefinition::LocalCache(local_modelfile) => {
                let last_modified: DateTime<Utc> = local_modelfile.modified.into();
                last_modified.format(DEFAULT_TIME_FORMAT).to_string().into()
            }
            ModelDefinition::Synced { remote, local: _ } => {
                let last_modified: DateTime<Utc> = remote
                    .modified_at
                    .parse()
                    .expect("could not parse datetime from ollama");
                last_modified.format(DEFAULT_TIME_FORMAT).to_string().into()
            }
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
                let name = Span::from(info.name());
                let size = Span::from(info.size());
                let last_modified: String = info.modified_at();
                let last_modified = Span::from(last_modified);
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
