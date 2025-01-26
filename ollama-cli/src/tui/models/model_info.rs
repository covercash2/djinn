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
    tui::{event::Action, ResponseEvent},
};

use super::ModelEvent;

#[derive(Debug, Clone, Default)]
pub struct ModelInfoViewModel {
    info: Option<ModelInfo>,
    /// x, y scroll offset
    scroll_offset: Offset,
    wrap: bool,
}

#[derive(Debug, Clone, Default)]
struct Offset {
    x: u16,
    y: u16,
}

impl Offset {
    fn up(&self) -> Self {
        Self {
            y: self.y.saturating_sub(1),
            x: self.x,
        }
    }

    fn down(&self) -> Self {
        Self {
            y: self.y.saturating_add(1),
            x: self.x,
        }
    }

    fn left(&self) -> Self {
        Self {
            x: self.x.saturating_sub(1),
            y: self.y,
        }
    }

    fn right(&self) -> Self {
        Self {
            x: self.x.saturating_add(1),
            y: self.y,
        }
    }

    fn as_tuple(&self) -> (u16, u16) {
        (self.y, self.x)
    }
}

impl ModelInfoViewModel {
    pub fn handle_response_event(&mut self, event: ResponseEvent) -> Result<()> {
        if let ResponseEvent::OllamaResponse(Response::ModelInfo(model_info)) = event {
            self.info = Some(model_info);
            Ok(())
        } else {
            Err(Error::UnexpectedResponse(event))
        }
    }

    pub fn handle_action(&mut self, action: Action) -> Result<Option<ModelEvent>> {
        match action {
            Action::Up => {
                self.scroll_offset = self.scroll_offset.up();
                Ok(None)
            }
            Action::Down => {
                self.scroll_offset = self.scroll_offset.down();
                Ok(None)
            }
            Action::Refresh => Ok(Some(ModelEvent::Refresh)),
            Action::Quit => Ok(Some(ModelEvent::Deactivate)),
            Action::Enter => {
                if let Some(ref model) = self.info {
                    Ok(Some(ModelEvent::EditFullModelfile(model.clone())))
                } else {
                    tracing::info!("no model selected to edit",);
                    Ok(None)
                }
            }
            Action::Left => {
                self.scroll_offset = self.scroll_offset.left();
                Ok(None)
            }
            Action::Right => {
                self.scroll_offset = self.scroll_offset.right();
                Ok(None)
            }
            _ => Ok(None),
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
            .scroll(view_model.scroll_offset.as_tuple())
            .block(Block::bordered())
            .style(style);

        let widget = if view_model.wrap {
            widget.wrap(Wrap { trim: false })
        } else {
            widget
        };

        self.render_widget(widget, parent);
    }
}
