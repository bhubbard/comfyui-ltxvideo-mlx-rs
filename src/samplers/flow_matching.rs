use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SamplerKind {
    Euler,
    Res2s,
    Distilled,
}

pub const DISTILLED_SIGMAS: [f32; 9] = [
    1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0,
];

#[derive(Debug, Clone)]
pub struct LTXSampler {
    pub kind: SamplerKind,
    pub sigmas: Vec<f32>,
}

impl LTXSampler {
    pub fn distilled() -> Self {
        Self {
            kind: SamplerKind::Distilled,
            sigmas: DISTILLED_SIGMAS.to_vec(),
        }
    }

    pub fn standard(steps: u32, kind: SamplerKind) -> Self {
        let mut sigmas = Vec::with_capacity(steps as usize + 1);
        for i in 0..=steps {
            let t = 1.0 - (i as f32 / steps as f32);
            sigmas.push(t);
        }
        Self { kind, sigmas }
    }

    pub fn step_euler(&self, step_idx: usize, latents: &[f32], velocity: &[f32]) -> Vec<f32> {
        let dt = self.sigmas[step_idx + 1] - self.sigmas[step_idx];
        latents
            .iter()
            .zip(velocity.iter())
            .map(|(&x, &v)| x + v * dt)
            .collect()
    }

    pub fn step_res_2s(
        &self,
        step_idx: usize,
        latents: &[f32],
        v_current: &[f32],
        v_predictor: &[f32],
    ) -> Vec<f32> {
        let dt = self.sigmas[step_idx + 1] - self.sigmas[step_idx];
        // Heun / 2nd order Runge-Kutta predictor-corrector
        latents
            .iter()
            .zip(v_current.iter().zip(v_predictor.iter()))
            .map(|(&x, (&v1, &v2))| x + 0.5 * (v1 + v2) * dt)
            .collect()
    }
}
