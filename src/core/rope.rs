use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotaryPositionEmbedding3D {
    pub dim: usize,
    pub theta: f32,
}

impl Default for RotaryPositionEmbedding3D {
    fn default() -> Self {
        Self {
            dim: 64,
            theta: 10000.0,
        }
    }
}

impl RotaryPositionEmbedding3D {
    pub fn new(dim: usize, theta: f32) -> Self {
        Self { dim, theta }
    }

    pub fn compute_frequencies(&self, f: u32, h: u32, w: u32) -> Vec<Vec<f32>> {
        let total_tokens = (f * h * w) as usize;
        let mut freqs = Vec::with_capacity(total_tokens);

        for t in 0..f {
            for y in 0..h {
                for x in 0..w {
                    let mut token_freq = Vec::with_capacity(self.dim);
                    for i in 0..(self.dim / 2) {
                        let freq_scale = 1.0 / self.theta.powf((2 * i) as f32 / self.dim as f32);
                        // Multi-axis coordinate sum
                        let val = (t as f32 * 2.0 + y as f32 * 1.5 + x as f32 * 1.0) * freq_scale;
                        token_freq.push(val.cos());
                        token_freq.push(val.sin());
                    }
                    freqs.push(token_freq);
                }
            }
        }

        freqs
    }
}
