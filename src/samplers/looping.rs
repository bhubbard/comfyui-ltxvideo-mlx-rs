use crate::error::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopingConfig {
    pub loop_blend_frames: u32,
    pub temporal_tile_size: u32,
    pub temporal_overlap: u32,
}

impl Default for LoopingConfig {
    fn default() -> Self {
        Self {
            loop_blend_frames: 16,
            temporal_tile_size: 49,
            temporal_overlap: 8,
        }
    }
}

pub struct LoopingSampler {
    pub config: LoopingConfig,
}

impl LoopingSampler {
    pub fn new(config: LoopingConfig) -> Self {
        Self { config }
    }

    pub fn blend_boundary_latents(
        &self,
        frames_latents: &mut [Vec<f32>],
    ) -> Result<()> {
        let total_frames = frames_latents.len();
        let blend_n = (self.config.loop_blend_frames as usize).min(total_frames / 2);

        if blend_n == 0 {
            return Ok(());
        }

        // Seamless cosine blending between first N frames and last N frames
        for i in 0..blend_n {
            let alpha = 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / blend_n as f32).cos());
            let start_idx = i;
            let end_idx = total_frames - blend_n + i;

            for d in 0..frames_latents[start_idx].len() {
                let start_val = frames_latents[start_idx][d];
                let end_val = frames_latents[end_idx][d];
                let blended = (1.0 - alpha) * end_val + alpha * start_val;
                frames_latents[start_idx][d] = blended;
                frames_latents[end_idx][d] = blended;
            }
        }

        Ok(())
    }
}
