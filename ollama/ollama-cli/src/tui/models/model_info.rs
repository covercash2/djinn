use ollama_rs::models::ModelInfo;
use ratatui::{layout::Rect, style::Style, text::Text, widgets::{Block, Paragraph}, Frame};

use crate::{error::{Error, Result }, lm::Response};

#[derive(Debug, Clone, Default)]
pub struct ModelInfoViewModel {
    info: Option<ModelInfo>,
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
            .block(Block::bordered())
            .style(style);

        self.render_widget(widget, parent);
    }
}

