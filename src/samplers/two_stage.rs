use crate::core::teacache::{TeaCacheConfig, TeaCacheState};
use crate::error::Result;
use crate::guidance::stg::{STGConfig, SpatioTemporalGuidance};
use crate::samplers::flow_matching::{LTXSampler, SamplerKind};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoStageConfig {
    pub stage1_steps: u32,
    pub stage2_steps: u32,
    pub sampler_kind: SamplerKind,
    pub teacache: TeaCacheConfig,
    pub stg: STGConfig,
}

impl Default for TwoStageConfig {
    fn default() -> Self {
        Self {
            stage1_steps: 30,
            stage2_steps: 3,
            sampler_kind: SamplerKind::Euler,
            teacache: TeaCacheConfig {
                enabled: true,
                threshold: 0.5,
                warmup_steps: 2,
            },
            stg: STGConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TwoStageSampler {
    pub config: TwoStageConfig,
}

impl TwoStageSampler {
    pub fn new(config: TwoStageConfig) -> Self {
        Self { config }
    }

    pub fn sample_latents(
        &self,
        latent_elements: usize,
        prompt: &str,
    ) -> Result<SamplingResult> {
        let start_time = Instant::now();
        let mut teacache = TeaCacheState::new(self.config.teacache.clone());
        let mut stg = SpatioTemporalGuidance::new(self.config.stg.clone());

        let sampler1 = LTXSampler::standard(self.config.stage1_steps, self.config.sampler_kind);
        let sampler2 = LTXSampler::standard(self.config.stage2_steps, self.config.sampler_kind);

        println!("🍎 Executing LTX-Video Two-Stage MLX Pipeline");
        println!("   Stage 1 Steps: {} ({:?}) | TeaCache: {}", self.config.stage1_steps, self.config.sampler_kind, self.config.teacache.enabled);
        println!("   Stage 2 Steps: {} (Detail Upsampling)", self.config.stage2_steps);
        println!("   Prompt:        \"{}\"", prompt);

        // Initialize pseudo latents
        let mut latents = vec![0.0f32; latent_elements];

        // Stage 1: Base Generation Loop
        for step in 0..self.config.stage1_steps as usize {
            let timestep_emb = vec![step as f32 * 0.1; 128];
            if !teacache.should_skip(step as u32, &timestep_emb) {
                // Simulate MLX DiT block evaluation
                std::thread::sleep(std::time::Duration::from_millis(5));
                teacache.update_cache(vec![0.01f32; 128]);
            }

            let pos = vec![0.05f32; latent_elements];
            let neg = vec![0.0f32; latent_elements];
            let guided = stg.compute_guided_noise(&pos, &neg, &pos);
            latents = sampler1.step_euler(step, &latents, &guided);
        }

        // Stage 2: Detail Refinement Loop
        for step in 0..self.config.stage2_steps as usize {
            std::thread::sleep(std::time::Duration::from_millis(5));
            let pos = vec![0.02f32; latent_elements];
            let neg = vec![0.0f32; latent_elements];
            let guided = stg.compute_guided_noise(&pos, &neg, &pos);
            latents = sampler2.step_euler(step, &latents, &guided);
        }

        let elapsed = start_time.elapsed().as_secs_f32();
        let speedup = teacache.speedup_ratio();

        println!(
            "✨ Two-Stage Sampling complete in {:.2}s (TeaCache effective speedup: {:.2}x, skipped {} blocks)",
            elapsed, speedup, teacache.skipped_blocks_count
        );

        Ok(SamplingResult {
            latents,
            elapsed_sec: elapsed,
            teacache_speedup: speedup,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingResult {
    pub latents: Vec<f32>,
    pub elapsed_sec: f32,
    pub teacache_speedup: f32,
}
