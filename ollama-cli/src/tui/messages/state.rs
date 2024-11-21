use ratatui::widgets::ListState;

#[derive(Debug, Clone, Default)]
pub struct MessagesState {
    pub list_state: ListState,
}

impl MessagesState {
    pub fn select(&mut self, item: Option<usize>) {
        self.list_state.select(item)
    }

    pub fn select_next(&mut self) {
        self.list_state.select_next()
    }

    pub fn select_previous(&mut self) {
        self.list_state.select_previous()
    }

    pub fn selected(&mut self) -> Option<usize> {
        self.list_state.selected()
    }
}
