use crate::error::Result;
use image::{ImageBuffer, Rgb};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TiledVaeConfig {
    pub tile_sample_min_height: u32,
    pub tile_sample_min_width: u32,
    pub tile_overlap_factor: f32,
    pub temporal_tile_frames: u32,
}

impl Default for TiledVaeConfig {
    fn default() -> Self {
        Self {
            tile_sample_min_height: 256,
            tile_sample_min_width: 256,
            tile_overlap_factor: 0.25,
            temporal_tile_frames: 33,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TiledVaeDecoder {
    pub config: TiledVaeConfig,
}

impl TiledVaeDecoder {
    pub fn new(config: TiledVaeConfig) -> Self {
        Self { config }
    }

    pub fn decode_to_frames(
        &self,
        _latents: &[f32],
        width: u32,
        height: u32,
        num_frames: u32,
        output_dir: &Path,
    ) -> Result<Vec<String>> {
        std::fs::create_dir_all(output_dir)?;
        let mut frame_paths = Vec::with_capacity(num_frames as usize);

        for f in 0..num_frames {
            let mut img = ImageBuffer::new(width, height);
            let time_ratio = f as f32 / num_frames as f32;

            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let r = ((x as f32 / width as f32) * 220.0 + time_ratio * 35.0) as u8;
                let g = ((y as f32 / height as f32) * 180.0) as u8;
                let b = 160u8;
                *pixel = Rgb([r, g, b]);
            }

            let path = output_dir.join(format!("frame_{:04}.png", f));
            img.save(&path)?;
            frame_paths.push(path.display().to_string());
        }

        Ok(frame_paths)
    }
}
