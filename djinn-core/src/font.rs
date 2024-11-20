pub fn get_default_font<'a>() -> Option<rusttype::Font<'a>> {
    let bytes = include_bytes!("../data/font/roboto-mono-stripped.ttf") as &[u8];
    rusttype::Font::try_from_vec(bytes.into())
}
