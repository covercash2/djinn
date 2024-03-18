use askama::Template;
use axum::response::Html;

#[derive(Template)]
#[template(path = "index.html")]
struct PromptingTemplate {
    prompt: String,
    response: String,
}

pub async fn index() -> Html<String> {
    let prompt = String::new();
    let response = String::new();

    let template = PromptingTemplate { prompt, response };

    Html(template.render().unwrap())
}
