use std::collections::HashMap;

use modelfile::{
    modelfile::{Instruction, InstructionName},
    Modelfile,
};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, List, ListState, Paragraph, Wrap},
    Frame,
};

use crate::error::{Error, Result};
use crate::{lm::Response, tui::event::Action};

use super::ModelEvent;

#[derive(Debug, Clone, Default)]
pub struct ModelfileViewModel {
    modelfile: Option<Modelfile>,
    model_info: HashMap<String, toml::Value>,
    selected: Option<usize>,
    instructions: Vec<Instruction>,
    details: Option<toml::Value>,
    list_state: ListState,
    active_panel: Option<Panel>,
}

impl ModelfileViewModel {
    pub fn load(&mut self, modelfile: Modelfile) -> Result<()> {
        let string = toml::to_string(&modelfile)?;
        let map: HashMap<String, toml::Value> = toml::from_str(&string)?;

        self.instructions = modelfile.clone().instructions().collect();
        self.modelfile = Some(modelfile);
        self.model_info = map;
        Ok(())
    }

    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        let Response::ModelInfo(model_info) = response else {
            return Ok(());
        };
        let modelfile: Modelfile = model_info.modelfile.parse()?;
        self.load(modelfile)
    }

    pub fn handle_action(&mut self, action: Action) -> Result<Option<ModelEvent>> {
        match action {
            Action::Quit => Ok(Some(ModelEvent::Deactivate)),
            Action::Up => {
                self.prev();
                self.update_details()?;
                Ok(None)
            }
            Action::Down => {
                self.next();
                self.update_details()?;
                Ok(None)
            }
            Action::Enter => {
                if let Some(_selected) = self.selected {
                    self.active_panel = Some(Panel::Details);
                    Ok(None)
                } else {
                    Ok(None)
                }
            }
            Action::Left => todo!(),
            Action::Right => todo!(),
            Action::LeftWord => todo!(),
            Action::RightWord => todo!(),
            Action::Refresh => todo!(),
            _ => todo!(),
        }
    }

    fn next(&mut self) {
        if let Some(selected) = self.selected {
            if selected == self.instructions.len() - 1 {
                self.selected = Some(0);
            } else {
                self.selected = Some(selected + 1);
            }
        } else {
            self.selected = Some(0);
        }
    }

    fn prev(&mut self) {
        if let Some(selected) = self.selected {
            if selected == 0 {
                self.selected = Some(self.instructions.len() - 1);
            } else {
                self.selected = Some(selected - 1);
            }
        } else {
            self.selected = Some(self.instructions.len() - 1);
        }
    }

    fn update_details(&mut self) -> Result<()> {
        tracing::info!(offset = self.list_state.selected(), "updating offset");
        if let Some(selected) = self.selected {
            let instruction = self
                .instructions
                .get(selected)
                .map(InstructionName::from)
                .ok_or(Error::ModelfileIndex(selected))?;
            let details = self
                .model_info
                .get(instruction.into())
                .cloned()
                .ok_or(Error::ModelfileMissing(instruction.to_string()))?;
            self.details = Some(details);
            self.list_state.select(Some(selected));
        }
        Ok(())
    }
}

#[derive(Default, Debug, Clone, Copy)]
enum Panel {
    #[default]
    Instruction,
    Details,
}

#[extend::ext(name = ModelfileView)]
pub impl<'a> Frame<'a> {
    fn modelfile(&mut self, parent: Rect, style: Style, view_model: &mut ModelfileViewModel) {
        let [instruction_panel, detail_panel] =
            Layout::horizontal([Constraint::Min(15), Constraint::Min(2)]).areas(parent);

        let instructions = List::from_iter(view_model.instructions.iter().map(|s| s.as_ref()))
            .block(Block::bordered())
            .style(style)
            .highlight_style(
                style
                    .fg(style.bg.unwrap_or(Color::Black))
                    .bg(style.fg.unwrap_or(Color::White)),
            );

        let details = Paragraph::new(
            view_model
                .details
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_default(),
        )
        .wrap(Wrap { trim: false })
        .block(Block::bordered());

        self.render_stateful_widget(instructions, instruction_panel, &mut view_model.list_state);
        self.render_widget(details, detail_panel);
    }
}
