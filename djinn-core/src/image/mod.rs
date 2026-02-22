use candle_core::{DType, Device, Result, Tensor};

pub mod clip;
pub mod config;
pub mod error;
pub mod gen;
pub mod siglip;

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, height, width).
pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        candle_core::bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => candle_core::bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle_core::Error::wrap)?;
    Ok(())
}

/// Load a set of images
pub fn load_images<P: AsRef<std::path::Path>>(
    paths: &[P],
    image_size: usize,
) -> anyhow::Result<Tensor> {
    paths.iter()
        .map(|p| load_image(p, image_size))
        .collect::<anyhow::Result<Vec<Tensor>>>()
        .and_then(|images| Tensor::stack(&images, 0).map_err(|e| anyhow::anyhow!(e)))
}

/// Load an image into a [`Tensor`]
pub fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    Ok(img)
}
