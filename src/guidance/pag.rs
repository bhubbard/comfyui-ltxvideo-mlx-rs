use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PAGConfig {
    pub enabled: bool,
    pub pag_scale: f32,
    pub perturbed_layers: Vec<usize>,
}

impl Default for PAGConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            pag_scale: 1.5,
            perturbed_layers: vec![10, 11, 12],
        }
    }
}
