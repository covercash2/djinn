use crate::error::Result;

use super::{event::Action, input::TextInputViewModel, AppEvent};

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

pub struct GenerateViewModel {
    input: TextInputViewModel,
    output: String,
    active_pane: Option<Pane>,
    focused_pane: Pane,
}

impl GenerateViewModel {
    pub async fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
        if let Some(pane) = &self.active_pane {
            todo!()
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
