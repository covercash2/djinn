use ratatui::{
    layout::{Constraint, Flex, Layout, Rect}, style::Style, widgets::{Block, Clear, Paragraph, Wrap}, Frame
};

#[derive(Debug, Clone)]
pub struct PopupViewModel {
    content: String,
}

impl PopupViewModel {
    pub fn new(content: impl ToString) -> Self {
        PopupViewModel { content: content.to_string() }
    }
}

pub fn popup_area(parent: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vertical = Layout::vertical([Constraint::Percentage(percent_y)]).flex(Flex::Center);
    let horizontal = Layout::horizontal([Constraint::Percentage(percent_x)]).flex(Flex::Center);

    let [area] = vertical.areas(parent);
    let [area] = horizontal.areas(area);

    area
}

#[extend::ext(name = PopupView)]
pub impl<'a> Frame<'a> {
    fn popup(&mut self, parent: Rect, style: Style, view_model: &mut PopupViewModel) {
        let block = Block::bordered().title("popup");
        let content = Paragraph::new(view_model.content.as_str())
            .wrap(Wrap { trim: true })
            .block(block)
            .style(style);

        let area = popup_area(parent, 60, 20);
        self.render_widget(Clear, area);
        self.render_widget(content, area);
    }
}
