use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APGConfig {
    pub eta: f32,
    pub norm_threshold: f32,
}

impl Default for APGConfig {
    fn default() -> Self {
        Self {
            eta: 1.0,
            norm_threshold: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdaptiveProjectedGuidance;

impl AdaptiveProjectedGuidance {
    pub fn guide(
        noise_pred_pos: &[f32],
        noise_pred_neg: &[f32],
        cfg_scale: f32,
        config: &APGConfig,
    ) -> Vec<f32> {
        let n = noise_pred_pos.len();
        let mut diff = Vec::with_capacity(n);
        let mut diff_norm_sq = 0.0f32;

        for i in 0..n {
            let d = noise_pred_pos[i] - noise_pred_neg.get(i).copied().unwrap_or(0.0);
            diff_norm_sq += d * d;
            diff.push(d);
        }

        let diff_norm = diff_norm_sq.sqrt();
        if config.norm_threshold > 0.0 && diff_norm > config.norm_threshold {
            let scale = config.norm_threshold / diff_norm;
            for d in &mut diff {
                *d *= scale;
            }
        }

        // Project diff onto noise_pred_pos: diff = diff_parallel + diff_orthogonal
        let mut dot_prod = 0.0f32;
        let mut pos_norm_sq = 0.0f32;
        for i in 0..n {
            dot_prod += diff[i] * noise_pred_pos[i];
            pos_norm_sq += noise_pred_pos[i] * noise_pred_pos[i];
        }

        let proj_factor = if pos_norm_sq > 1e-6 { dot_prod / pos_norm_sq } else { 0.0 };

        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let diff_par = proj_factor * noise_pred_pos[i];
            let diff_orth = diff[i] - diff_par;
            let normalized_update = diff_orth + config.eta * diff_par;
            output.push(noise_pred_pos[i] + (cfg_scale - 1.0) * normalized_update);
        }

        output
    }
}
