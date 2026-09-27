use crate::error::Result;

pub struct VideoLatentPatchifier {
    pub spatial_patch_size: usize,
    pub temporal_patch_size: usize,
}

impl Default for VideoLatentPatchifier {
    fn default() -> Self {
        Self {
            spatial_patch_size: 1,
            temporal_patch_size: 1,
        }
    }
}

impl VideoLatentPatchifier {
    pub fn patchify_shape(&self, f: u32, h: u32, w: u32, channels: u32) -> (usize, usize) {
        let seq_len = (f * h * w) as usize;
        let dim = channels as usize;
        (seq_len, dim)
    }

    pub fn unpatchify_shape(&self, seq_len: usize, f: u32, h: u32, w: u32) -> Result<(u32, u32, u32)> {
        let expected = (f * h * w) as usize;
        if seq_len != expected {
            return Err(crate::error::LtxVideoError::Dimension(format!(
                "Sequence length mismatch: expected {}, got {}",
                expected, seq_len
            )));
        }
        Ok((f, h, w))
    }
}

pub struct AudioPatchifier;

impl AudioPatchifier {
    pub fn patchify_shape(audio_tokens: u32, channels: u32) -> (usize, usize) {
        (audio_tokens as usize, channels as usize)
    }
}
