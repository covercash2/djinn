use model_info::{ModelInfoView, ModelInfoViewModel};
use model_list::{ModelListView, ModelListViewModel};
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

#[derive(Clone, Debug, Default)]
pub struct ModelsViewModel {
    model_list: ModelListViewModel,
    model_info: ModelInfoViewModel,
    active_pane: Option<Pane>,
    focused_pane: Pane,
}

impl ModelsViewModel {
    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        match response {
            Response::LocalModels(_) => self.model_list.handle_response(response),
            Response::ModelInfo(_) => self.model_info.handle_response(response),
            _ => Err(Error::UnexpectedResponse(response)),
        }
    }

    pub async fn handle_event(&mut self, action: Action) -> Result<Option<AppEvent>> {
        if let Some(pane) = &self.active_pane {
            let model_event = match pane {
                Pane::ModelList => self.model_list.handle_event(action).await?,
                Pane::ModelInfo => todo!(),
            };

            if let Some(model_event) = model_event {
                match model_event {
                    ModelEvent::Activate(pane) => {
                        self.active_pane = Some(pane);
                        Ok(None)
                    }
                    ModelEvent::Deactivate => {
                        self.active_pane = None;
                        Ok(None)
                    }
                    ModelEvent::Refresh => Ok(Some(AppEvent::Submit(Prompt::LocalModels))),
                    ModelEvent::GetInfo(model_name) => {
                        Ok(Some(AppEvent::Submit(Prompt::ModelInfo(model_name))))
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
                Action::Quit => Ok(Some(AppEvent::Quit)),
                Action::LeftWord | Action::RightWord | Action::Unhandled => Ok(None),
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum ModelEvent {
    Activate(Pane),
    Deactivate,
    GetInfo(ModelName),
    Refresh,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Pane {
    #[default]
    ModelList,
    ModelInfo,
}

impl Pane {
    fn next(&self) -> Pane {
        match self {
            Pane::ModelList => Pane::ModelInfo,
            Pane::ModelInfo => Pane::ModelList,
        }
    }

    fn previous(&self) -> Pane {
        match self {
            Pane::ModelList => Pane::ModelInfo,
            Pane::ModelInfo => Pane::ModelList,
        }
    }
}

#[extend::ext(name = ModelsView)]
pub impl<'a> Frame<'a> {
    fn models_view(&mut self, parent: Rect, style: Style, view_model: &mut ModelsViewModel) {
        let vertical = Layout::vertical([Constraint::Min(2), Constraint::Min(1)]);

        let [model_list_area, model_info_area] = vertical.areas(parent);

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
