use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STGConfig {
    pub enabled: bool,
    pub cfg_scale: f32,
    pub stg_scale: f32,
    pub rescale_scale: f32,
    pub momentum: f32,
}

impl Default for STGConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cfg_scale: 3.5,
            stg_scale: 1.0,
            rescale_scale: 0.7,
            momentum: 0.85,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatioTemporalGuidance {
    pub config: STGConfig,
    running_average: Vec<f32>,
}

impl SpatioTemporalGuidance {
    pub fn new(config: STGConfig) -> Self {
        Self {
            config,
            running_average: Vec::new(),
        }
    }

    pub fn compute_guided_noise(
        &mut self,
        noise_pred_pos: &[f32],
        noise_pred_neg: &[f32],
        noise_pred_perturbed: &[f32],
    ) -> Vec<f32> {
        let n = noise_pred_pos.len();
        let mut guided = Vec::with_capacity(n);

        let cfg_factor = self.config.cfg_scale - 1.0;
        let stg_scale = self.config.stg_scale;

        for i in 0..n {
            let pos = noise_pred_pos[i];
            let neg = noise_pred_neg.get(i).copied().unwrap_or(0.0);
            let pert = noise_pred_perturbed.get(i).copied().unwrap_or(pos);

            let val = pos + cfg_factor * (pos - neg) + stg_scale * (pos - pert);
            guided.push(val);
        }

        // Rescaling based on standard deviation
        if self.config.rescale_scale > 0.0 {
            let std_pos = compute_std(noise_pred_pos);
            let std_guided = compute_std(&guided);

            if std_guided > 1e-6 {
                let factor = std_pos / std_guided;
                let final_factor = self.config.rescale_scale * factor + (1.0 - self.config.rescale_scale);
                for v in &mut guided {
                    *v *= final_factor;
                }
            }
        }

        // Apply momentum smoothing if enabled
        if self.config.momentum > 0.0 {
            if self.running_average.is_empty() {
                self.running_average = guided.clone();
            } else {
                for (avg, g) in self.running_average.iter_mut().zip(guided.iter()) {
                    *avg = self.config.momentum * (*avg) + (1.0 - self.config.momentum) * (*g);
                }
                guided = self.running_average.clone();
            }
        }

        guided
    }
}

fn compute_std(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 1.0;
    }
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    var.sqrt()
}
