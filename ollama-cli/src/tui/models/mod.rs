use model_info::{ModelInfoView, ModelInfoViewModel};
use model_list::{ModelListView, ModelListViewModel};
use modelfile::{ModelfileView, ModelfileViewModel};
use ollama_rs::models::ModelInfo;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Style,
    Frame,
};

use crate::{
    error::{Error, Result},
    lm::{Prompt, Response},
    ollama::ModelName,
};

use super::{event::Action, AppEvent, StyleExt};

mod model_info;
mod model_list;
mod modelfile;

#[derive(Clone, Debug, Default)]
pub struct ModelsViewModel {
    model_list: ModelListViewModel,
    model_info: ModelInfoViewModel,
    modelfile: ModelfileViewModel,
    active_pane: Option<Pane>,
    focused_pane: Pane,
}

impl ModelsViewModel {
    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        match response {
            Response::LocalModels(_) => self.model_list.handle_response(response),
            Response::ModelInfo(_) => self
                .model_info
                .handle_response(response.clone())
                .and_then(|_| self.modelfile.handle_response(response)),
            _ => Err(Error::UnexpectedResponse(response)),
        }
    }

    pub async fn handle_event(&mut self, action: Action) -> Result<Option<AppEvent>> {
        if let Some(pane) = &self.active_pane {
            let model_event = match pane {
                Pane::ModelList => self.model_list.handle_event(action).await?,
                Pane::ModelInfo => self.model_info.handle_action(action)?,
                Pane::Modelfile => self.modelfile.handle_action(action)?,
            };

            if let Some(model_event) = model_event {
                match model_event {
                    ModelEvent::Deactivate => {
                        self.active_pane = None;
                        Ok(None)
                    }
                    ModelEvent::Refresh => Ok(Some(AppEvent::Submit(Prompt::LocalModels))),
                    ModelEvent::GetInfo(model_name) => {
                        Ok(Some(AppEvent::Submit(Prompt::ModelInfo(model_name))))
                    }
                    ModelEvent::EditInfo(model_info) => {
                        Ok(Some(AppEvent::EditSystemPrompt(model_info)))
                    }
                }
            } else {
                Ok(None)
            }
        } else {
            match action {
                Action::Up | Action::Left => {
                    self.focused_pane = self.focused_pane.previous();
                    Ok(None)
                }
                Action::Down | Action::Right => {
                    self.focused_pane = self.focused_pane.next();
                    Ok(None)
                }
                Action::Refresh => Ok(Some(AppEvent::Submit(Prompt::LocalModels))),
                Action::Enter => {
                    self.active_pane = Some(self.focused_pane);
                    Ok(None)
                }
                Action::Quit => Ok(Some(AppEvent::Deactivate)),
                Action::LeftWord | Action::RightWord | Action::Unhandled => Ok(None),
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum ModelEvent {
    Deactivate,
    EditInfo(ModelInfo),
    GetInfo(ModelName),
    Refresh,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Pane {
    #[default]
    ModelList,
    ModelInfo,
    Modelfile,
}

impl Pane {
    fn next(&self) -> Pane {
        match self {
            Pane::ModelList => Pane::ModelInfo,
            Pane::ModelInfo => Pane::Modelfile,
            Pane::Modelfile => Pane::ModelList,
        }
    }

    fn previous(&self) -> Pane {
        match self {
            Pane::ModelList => Pane::Modelfile,
            Pane::ModelInfo => Pane::ModelList,
            Pane::Modelfile => Pane::ModelInfo,
        }
    }
}

#[extend::ext(name = ModelsView)]
pub impl<'a> Frame<'a> {
    fn models_view(&mut self, parent: Rect, style: Style, view_model: &mut ModelsViewModel) {
        let vertical = Layout::vertical([
            Constraint::Percentage(20),
            Constraint::Min(1),
            Constraint::Min(1),
        ]);

        let [model_list_area, model_info_area, modelfile_area] = vertical.areas(parent);

        let model_list_style = if let Some(Pane::ModelList) = view_model.active_pane {
            Style::active()
        } else if view_model.focused_pane == Pane::ModelList {
            Style::focused()
        } else {
            style
        };
        self.model_list(
            model_list_area,
            model_list_style,
            &mut view_model.model_list,
        );

        let model_info_style = if let Some(Pane::ModelInfo) = view_model.active_pane {
            Style::active()
        } else if view_model.focused_pane == Pane::ModelInfo {
            Style::focused()
        } else {
            style
        };

        self.model_info(
            model_info_area,
            model_info_style,
            &mut view_model.model_info,
        );

        let modelfile_style = if let Some(Pane::Modelfile) = view_model.active_pane {
            Style::active()
        } else if view_model.focused_pane == Pane::Modelfile {
            Style::focused()
        } else {
            style
        };

        self.modelfile(modelfile_area, modelfile_style, &mut view_model.modelfile);
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
