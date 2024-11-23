use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Style,
    widgets::{Block, Paragraph, Wrap},
    Frame,
};

use crate::{
    error::Result,
    lm::{Prompt, Response},
};

use super::{
    event::Action,
    input::{InputView, TextInputEvent, TextInputViewModel},
    AppEvent,
};

#[derive(Debug, Default, Clone, Copy, PartialEq)]
enum Pane {
    #[default]
    Input,
    Output,
}

impl Pane {
    fn next(self) -> Pane {
        match self {
            Pane::Input => Pane::Output,
            Pane::Output => Pane::Input,
        }
    }

    fn previous(self) -> Pane {
        self.next()
    }
}

#[derive(Clone, Debug, Default)]
pub struct GenerateViewModel {
    input: TextInputViewModel,
    output: String,
    scroll_state: u16,
    active_pane: Option<Pane>,
    focused_pane: Pane,
}

impl GenerateViewModel {
    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        match response {
            Response::Token(arc) => {
                self.output.push_str(arc.as_ref());
            }
            Response::Eos => {}
            Response::Error(arc) => {
                self.output = arc.to_string();
            }
            _ => {}
        }
        Ok(())
    }

    pub fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
        if let Some(pane) = &self.active_pane {
            match pane {
                Pane::Input => {
                    Ok(self.input.handle_action(action)?.and_then(
                        |input_action| match input_action {
                            TextInputEvent::InputMode(input_mode) => {
                                Some(AppEvent::InputMode(input_mode))
                            }
                            TextInputEvent::Submit(input) => {
                                Some(AppEvent::Submit(Prompt::Generate(input)))
                            }
                            TextInputEvent::Quit => {
                                self.active_pane = None;
                                None
                            }
                        },
                    ))
                }
                Pane::Output => match action {
                    Action::Beginning => {
                        self.scroll_state = 0;
                        Ok(None)
                    }
                    Action::End => {
                        self.scroll_state = (self.output.lines().count() - 1) as u16;
                        Ok(None)
                    }
                    Action::Up => {
                        self.scroll_state = self.scroll_state.saturating_sub(1);
                        Ok(None)
                    }
                    Action::Down => {
                        self.scroll_state = self.scroll_state.saturating_add(1);
                        Ok(None)
                    }
                    Action::Quit | Action::Escape => {
                        self.active_pane = None;
                        Ok(None)
                    }
                    Action::Enter => todo!(),
                    _ => Ok(None),
                },
            }
        } else {
            match action {
                Action::Up => {
                    self.focused_pane = self.focused_pane.previous();
                    Ok(None)
                }
                Action::Down => {
                    self.focused_pane = self.focused_pane.next();
                    Ok(None)
                }
                Action::Refresh => todo!(),
                Action::Enter => {
                    self.active_pane = Some(self.focused_pane);
                    Ok(None)
                }
                Action::Quit => Ok(Some(AppEvent::Deactivate)),
                _ => Ok(None),
            }
        }
    }
}

#[extend::ext(name = GenerateView)]
pub impl<'a> Frame<'a> {
    fn generate_view(&mut self, parent: Rect, style: Style, view_model: &GenerateViewModel) {
        let vertical = Layout::vertical([Constraint::Percentage(20), Constraint::Min(1)]);

        let [input_area, output_area] = vertical.areas(parent);

        self.input_view(input_area, style, &view_model.input);

        let output = Paragraph::new(view_model.output.as_str())
            .wrap(Wrap { trim: true })
            .block(Block::bordered());

        self.render_widget(output, output_area);
    }
}
