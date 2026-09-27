use crate::error::{LtxVideoError, Result};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryProfile {
    Standard,
    LowVram,
    Q8,
}

impl Default for MemoryProfile {
    fn default() -> Self {
        MemoryProfile::Standard
    }
}

impl FromStr for MemoryProfile {
    type Err = LtxVideoError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "standard" | "std" => Ok(MemoryProfile::Standard),
            "low_vram" | "lowvram" | "low" => Ok(MemoryProfile::LowVram),
            "q8" | "balanced" => Ok(MemoryProfile::Q8),
            other => Err(LtxVideoError::Custom(format!(
                "Invalid memory profile '{}'. Choices: standard, low_vram, q8",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelVariant {
    Ltx23Bf16,
    Ltx23Q8,
    Ltx23Q4,
    Ltx23Distilled,
    Ltx20Full,
}

impl Default for ModelVariant {
    fn default() -> Self {
        ModelVariant::Ltx23Bf16
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTXConfig {
    pub variant: ModelVariant,
    pub memory_profile: MemoryProfile,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub spatial_patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub audio_sample_rate: u32,
    pub max_sequence_length: usize,
}

impl Default for LTXConfig {
    fn default() -> Self {
        Self::ltx_2_3()
    }
}

impl LTXConfig {
    pub fn ltx_2_3() -> Self {
        Self {
            variant: ModelVariant::Ltx23Bf16,
            memory_profile: MemoryProfile::Standard,
            hidden_size: 4096,
            num_layers: 28,
            num_heads: 32,
            spatial_patch_size: 32,
            temporal_patch_size: 8,
            in_channels: 128,
            out_channels: 128,
            audio_sample_rate: 44100,
            max_sequence_length: 4096,
        }
    }

    pub fn compute_latent_shape(&self, num_frames: u32, height: u32, width: u32) -> Result<(u32, u32, u32)> {
        if height % 32 != 0 || width % 32 != 0 {
            return Err(LtxVideoError::Dimension(format!(
                "Height ({}) and width ({}) must be multiples of 32.",
                height, width
            )));
        }
        let f = ((num_frames.saturating_sub(1)) / 8) + 1;
        let h = height / 32;
        let w = width / 32;
        Ok((f, h, w))
    }

    pub fn compute_audio_tokens(&self, num_frames: u32, fps: f32) -> u32 {
        let duration_sec = num_frames as f32 / fps;
        // ~50 tokens per second of audio in LTX-2.3
        ((duration_sec * 50.0).round() as u32).max(1)
    }
}
