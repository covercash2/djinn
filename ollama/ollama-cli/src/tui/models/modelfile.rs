use std::collections::HashMap;

use ratatui::{
    layout::{Constraint, Layout, Offset, Rect},
    style::{Color, Style},
    widgets::{Block, List, ListState, Paragraph, Wrap},
    Frame,
};

use crate::error::{Error, Result};
use crate::{lm::Response, ollama::modelfile::Modelfile, tui::event::Action};

use super::ModelEvent;

#[derive(Debug, Clone, Default)]
pub struct ModelfileViewModel {
    modelfile: Option<Modelfile>,
    model_info: HashMap<String, toml::Value>,
    instructions: Vec<String>,
    details: Option<toml::Value>,
    list_state: ListState,
    /// x, y scroll offset
    scroll_offset: Offset,
    wrap: bool,
    active_panel: Panel,
}

impl ModelfileViewModel {
    pub fn load(&mut self, modelfile: Modelfile) -> Result<()> {
        let string = toml::to_string(&modelfile)?;
        let map: HashMap<String, toml::Value> = toml::from_str(&string)?;

        self.modelfile = Some(modelfile);
        self.model_info = map;
        let mut instructions: Vec<String> = self.model_info.keys().cloned().collect();
        instructions.sort();
        self.instructions = instructions;
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
                self.list_state.select_previous();
                self.update_details()?;
                Ok(None)
            }
            Action::Down => {
                self.list_state.select_next();
                self.update_details()?;
                Ok(None)
            }
            Action::Enter => todo!(),
            Action::Left => todo!(),
            Action::Right => todo!(),
            Action::LeftWord => todo!(),
            Action::RightWord => todo!(),
            Action::Refresh => todo!(),
            Action::Unhandled => todo!(),
        }
    }

    fn update_details(&mut self) -> Result<()> {
        tracing::info!(offset = self.list_state.selected(), "updating offset");
        if let Some(selected) = self.list_state.selected() {
            let details = self
                .instructions
                .get(selected)
                .and_then(|instruction| self.model_info.get(instruction))
                .cloned()
                .ok_or(Error::ModelfileIndex)?;
            self.details = Some(details);
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

        let instructions = List::from_iter(view_model.instructions.iter().map(|s| s.as_str()))
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
