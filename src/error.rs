use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LtxVideoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("Model load error: {0}")]
    ModelNotFound(PathBuf),

    #[error("Dimension error: {0}")]
    Dimension(String),

    #[error("Sampler error: {0}")]
    Sampler(String),

    #[error("ComfyUI workflow error: {0}")]
    Workflow(String),

    #[error("Memory allocation error: {0}")]
    Memory(String),

    #[error("{0}")]
    Custom(String),
}

pub type Result<T> = std::result::Result<T, LtxVideoError>;
