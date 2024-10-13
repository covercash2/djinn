use ollama_rs::models::ModelInfo;
use ratatui::{
    layout::Rect,
    style::Style,
    text::Text,
    widgets::{Block, Paragraph, Wrap},
    Frame,
};

use crate::{
    error::{Error, Result},
    lm::Response,
    tui::event::Action,
};

use super::ModelEvent;

#[derive(Debug, Clone, Default)]
pub struct ModelInfoViewModel {
    info: Option<ModelInfo>,
    scroll_offset: u16,
}

impl ModelInfoViewModel {
    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        if let Response::ModelInfo(model_info) = response {
            self.info = Some(model_info);
            Ok(())
        } else {
            Err(Error::UnexpectedResponse(response))
        }
    }

    pub fn handle_action(&mut self, action: Action) -> Result<Option<ModelEvent>> {
        match action {
            Action::Up => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
                Ok(None)
            }
            Action::Down => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
                Ok(None)
            }
            Action::Refresh => Ok(Some(ModelEvent::Refresh)),
            Action::Quit => Ok(Some(ModelEvent::Deactivate)),
            Action::Enter => {
                if let Some(ref model) = self.info {
                    Ok(Some(ModelEvent::EditInfo(model.clone())))
                } else {
                    tracing::info!("no model selected to edit",);
                    Ok(None)
                }
            }
            Action::Left
            | Action::Right
            | Action::LeftWord
            | Action::RightWord
            | Action::Unhandled => Ok(None),
        }
    }
}

#[extend::ext(name = ModelInfoView)]
pub impl<'a> Frame<'a> {
    fn model_info(&mut self, parent: Rect, style: Style, view_model: &mut ModelInfoViewModel) {
        let text = if let Some(ref info) = view_model.info {
            Text::from(info.modelfile.as_str())
        } else {
            Text::from("select a model")
        };

        let widget = Paragraph::new(text)
            .scroll((view_model.scroll_offset, 0))
            .wrap(Wrap { trim: false })
            .block(Block::bordered())
            .style(style);

        self.render_widget(widget, parent);
    }
}
