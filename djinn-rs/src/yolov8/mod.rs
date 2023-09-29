pub mod model;

use candle_core::{IndexOp, Result, Tensor};
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use image::DynamicImage;

// Keypoints as reported by ChatGPT :)
// Nose
// Left Eye
// Right Eye
// Left Ear
// Right Ear
// Left Shoulder
// Right Shoulder
// Left Elbow
// Right Elbow
// Left Wrist
// Right Wrist
// Left Hip
// Right Hip
// Left Knee
// Right Knee
// Left Ankle
// Right Ankle
const KP_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];
// Model architecture from https://github.com/ultralytics/ultralytics/issues/189
// https://github.com/tinygrad/tinygrad/blob/master/examples/yolov8.py

pub fn report_detect(
    pred: &Tensor,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
) -> Result<DynamicImage> {
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);

    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    let font = crate::font::get_default_font();
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", crate::coco_classes::NAMES[class_index], b);
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let dx = (b.xmax - b.xmin) * w_ratio;
            let dy = (b.ymax - b.ymin) * h_ratio;
            if dx >= 0. && dy >= 0. {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                    image::Rgb([255, 0, 0]),
                );
            }
            if legend_size > 0 {
                if let Some(font) = font.as_ref() {
                    imageproc::drawing::draw_filled_rect_mut(
                        &mut img,
                        imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, legend_size),
                        image::Rgb([170, 0, 0]),
                    );
                    let legend = format!(
                        "{}   {:.0}%",
                        crate::coco_classes::NAMES[class_index],
                        100. * b.confidence
                    );
                    imageproc::drawing::draw_text_mut(
                        &mut img,
                        image::Rgb([255, 255, 255]),
                        xmin,
                        ymin,
                        rusttype::Scale::uniform(legend_size as f32 - 1.),
                        font,
                        &legend,
                    )
                }
            }
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

pub fn report_pose(
    pred: &Tensor,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<DynamicImage> {
    let (pred_size, npreds) = pred.dims2()?;
    if pred_size != 17 * 3 + 4 + 1 {
        candle_core::bail!("unexpected pred-size {pred_size}");
    }
    let mut bboxes = vec![];
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = pred[4];
        if confidence > confidence_threshold {
            let keypoints = (0..17)
                .map(|i| KeyPoint {
                    x: pred[3 * i + 5],
                    y: pred[3 * i + 6],
                    mask: pred[3 * i + 7],
                })
                .collect::<Vec<_>>();
            let bbox = Bbox {
                xmin: pred[0] - pred[2] / 2.,
                ymin: pred[1] - pred[3] / 2.,
                xmax: pred[0] + pred[2] / 2.,
                ymax: pred[1] + pred[3] / 2.,
                confidence,
                data: keypoints,
            };
            bboxes.push(bbox)
        }
    }

    let mut bboxes = vec![bboxes];
    non_maximum_suppression(&mut bboxes, nms_threshold);
    let bboxes = &bboxes[0];

    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    for b in bboxes.iter() {
        println!("{b:?}");
        let xmin = (b.xmin * w_ratio) as i32;
        let ymin = (b.ymin * h_ratio) as i32;
        let dx = (b.xmax - b.xmin) * w_ratio;
        let dy = (b.ymax - b.ymin) * h_ratio;
        if dx >= 0. && dy >= 0. {
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                image::Rgb([255, 0, 0]),
            );
        }
        for kp in b.data.iter() {
            if kp.mask < 0.6 {
                continue;
            }
            let x = (kp.x * w_ratio) as i32;
            let y = (kp.y * h_ratio) as i32;
            imageproc::drawing::draw_filled_circle_mut(
                &mut img,
                (x, y),
                2,
                image::Rgb([0, 255, 0]),
            );
        }

        for &(idx1, idx2) in KP_CONNECTIONS.iter() {
            let kp1 = &b.data[idx1];
            let kp2 = &b.data[idx2];
            if kp1.mask < 0.6 || kp2.mask < 0.6 {
                continue;
            }
            imageproc::drawing::draw_line_segment_mut(
                &mut img,
                (kp1.x * w_ratio, kp1.y * h_ratio),
                (kp2.x * w_ratio, kp2.y * h_ratio),
                image::Rgb([255, 255, 0]),
            );
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}
