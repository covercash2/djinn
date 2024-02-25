pub trait TextGenerator {
    async fn generate(&mut self, prompt: String);
}
