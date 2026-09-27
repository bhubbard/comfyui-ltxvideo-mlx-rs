use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeaCacheConfig {
    pub enabled: bool,
    pub threshold: f32,
    pub warmup_steps: u32,
}

impl Default for TeaCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 0.5,
            warmup_steps: 2,
        }
    }
}

pub struct TeaCacheState {
    pub config: TeaCacheConfig,
    pub cached_activations: Option<Vec<f32>>,
    pub previous_timestep_emb: Option<Vec<f32>>,
    pub skipped_blocks_count: u32,
    pub evaluated_blocks_count: u32,
}

impl TeaCacheState {
    pub fn new(config: TeaCacheConfig) -> Self {
        Self {
            config,
            cached_activations: None,
            previous_timestep_emb: None,
            skipped_blocks_count: 0,
            evaluated_blocks_count: 0,
        }
    }

    pub fn should_skip(&mut self, step: u32, current_emb: &[f32]) -> bool {
        if !self.config.enabled || step < self.config.warmup_steps || self.cached_activations.is_none() {
            self.evaluated_blocks_count += 1;
            self.previous_timestep_emb = Some(current_emb.to_vec());
            return false;
        }

        if let Some(ref prev_emb) = self.previous_timestep_emb {
            let mut diff_norm_sq = 0.0f32;
            let mut base_norm_sq = 0.0f32;

            for (a, b) in prev_emb.iter().zip(current_emb.iter()) {
                let diff = a - b;
                diff_norm_sq += diff * diff;
                base_norm_sq += a * a;
            }

            let delta = if base_norm_sq > 1e-6 {
                (diff_norm_sq / base_norm_sq).sqrt()
            } else {
                1.0
            };

            if delta < self.config.threshold {
                self.skipped_blocks_count += 1;
                return true;
            }
        }

        self.evaluated_blocks_count += 1;
        self.previous_timestep_emb = Some(current_emb.to_vec());
        false
    }

    pub fn update_cache(&mut self, activations: Vec<f32>) {
        self.cached_activations = Some(activations);
    }

    pub fn speedup_ratio(&self) -> f32 {
        let total = self.evaluated_blocks_count + self.skipped_blocks_count;
        if self.evaluated_blocks_count > 0 {
            total as f32 / self.evaluated_blocks_count as f32
        } else {
            1.0
        }
    }
}
