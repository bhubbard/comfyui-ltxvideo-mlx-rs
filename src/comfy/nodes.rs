use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfyNode {
    pub id: usize,
    pub class_type: String,
    pub inputs: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NodeRegistry;

impl NodeRegistry {
    pub fn is_known_ltx_node(class_type: &str) -> bool {
        matches!(
            class_type,
            "LTXVMLXCheckpointLoader"
                | "LTXVMLXTextEncoderLoader"
                | "LTXVMLXBaseSampler"
                | "LTXVMLXTwoStageSampler"
                | "LTXVMLXTwoStageHQSampler"
                | "LTXVMLXVAELoader"
                | "LTXVMLXVAEDecode"
                | "LTXVSparseMotionTrack"
                | "LTXVICLoRAUnionControl"
                | "LTXVLoopingSampler"
        )
    }
}
