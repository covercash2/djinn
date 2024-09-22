use std::io::{stdout, Stdout};
use std::{sync::Arc, time::Duration};

use futures::StreamExt as _;
use input::{InputMode, InputView, InputViewModel};
use messages::MessagesViewModel;
use modalkit::actions::Action;
use modalkit::{
    editing::{application::EmptyInfo, context::EditContext, key::KeyManager, store::Store},
    env::vim::keybindings::default_vim_keys,
};
use modalkit_ratatui::textbox::TextBoxState;
use model_context::ModelContext;
use ratatui::prelude::CrosstermBackend;
use ratatui::Terminal;
use ratatui::{
    crossterm::event::Event,
    layout::{Constraint, Layout},
    style::{Modifier, Style, Stylize},
    text::{Line, Text},
    widgets::Paragraph,
    DefaultTerminal, Frame,
};

use crate::{
    lm::Prompt,
    ollama::{
        self,
        chat::{ChatRequest, Message},
    },
    tui::messages::MessagesView as _,
};

use modalkit::crossterm::terminal::EnterAlternateScreen;
use modalkit_ratatui::TerminalExtOps;

mod input;
mod messages;
mod model_context;
mod widgets_ext;

pub struct AppContext {
    input_context: InputViewModel,
    model_context: ModelContext,
    messages_view_model: MessagesViewModel,
}

impl AppContext {
    pub fn new(client: ollama::Client) -> Self {
        let mut store: Store<EmptyInfo> = Store::default();
        let bindings = KeyManager::new(default_vim_keys::<EmptyInfo>());
        let text_box_state = TextBoxState::new(store.load_buffer(String::from("")));

        let input_context = InputViewModel {
            input: "".into(),
            cursor_position: Default::default(),
            mode: Default::default(),
            text_box_state,
            bindings,
            store,
        };

        Self {
            input_context,
            model_context: ModelContext::spawn(client),
            messages_view_model: Default::default(),
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let vertical = Layout::vertical([
            Constraint::Length(1),
            Constraint::Max(5),
            Constraint::Min(1),
        ]);

        let [help_area, input_area, messages_area] = vertical.areas(frame.area());

        let (msg, style) = match self.input_context.mode {
            InputMode::Normal => (
                vec![
                    "Press ".into(),
                    "q".bold(),
                    " to exit, ".into(),
                    "e".bold(),
                    " to start editing.".bold(),
                ],
                Style::default().add_modifier(Modifier::RAPID_BLINK),
            ),
            InputMode::Edit => (
                vec![
                    "Press ".into(),
                    "Esc".bold(),
                    " to stop editing, ".into(),
                    "Enter".bold(),
                    " to record the message".into(),
                ],
                Style::default(),
            ),
        };

        let text = Text::from(Line::from(msg)).patch_style(style);
        let help_message = Paragraph::new(text);
        frame.render_widget(help_message, help_area);

        frame.input_view(input_area, &mut self.input_context);
        frame.message_view(messages_area, &self.messages_view_model);
    }

    pub async fn run(
        mut self,
        mut terminal: Terminal<CrosstermBackend<Stdout>>,
    ) -> anyhow::Result<()> {
        let mut stdout = stdout();

        crossterm::terminal::enable_raw_mode()?;
        crossterm::execute!(stdout, EnterAlternateScreen)?;

        let period = Duration::from_secs_f32(1.0 / 15.0);
        let mut interval = tokio::time::interval(period);
        let mut events = ratatui::crossterm::event::EventStream::new();

        terminal.clear()?;

        loop {
            tokio::select! {
                _ = interval.tick() => { terminal.draw(|frame| self.draw(frame))?; },
                Some(Ok(Event::Key(key))) = events.next() => {
                    if let Some(app_event) = self.input_context.handle_key_event(key)
                        .expect("should be able to handle key events, i guess. these errors aren't designed very well")
                    {
                        match app_event {
                            AppEvent::Submit(message) => self.submit_message(message).await,
                            AppEvent::Suspend => {
                                terminal.program_suspend().expect("could not suspend program");
                            },
                            AppEvent::Quit => return Ok(()),
                        }
                    }
                },
                Some(response) = self.model_context.response_receiver.recv() => {
                    self.messages_view_model.handle_response(response);
                }
            }
        }
    }

    async fn submit_message(&mut self, prompt: Arc<str>) {
        self.messages_view_model
            .push_message(Message::User(prompt.clone()));

        let chat = Prompt::Chat(ChatRequest {
            prompt,
            model: Default::default(),
            history: self.messages_view_model.history(),
        });
        self.model_context
            .prompt_sender
            .send(chat)
            .await
            .expect("unable to send message")
    }
}

pub enum AppEvent {
    Submit(Arc<str>),
    Suspend,
    Quit,
}
