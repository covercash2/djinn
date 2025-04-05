use ab_glyph::{Font, FontArc, InvalidFont};

pub fn get_default_font<'a>() -> Result<impl Font, InvalidFont> {
    let bytes = include_bytes!("../data/font/roboto-mono-stripped.ttf") as &[u8];

    FontArc::try_from_slice(bytes)
}
