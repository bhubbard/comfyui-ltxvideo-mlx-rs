use crate::comfy::nodes::ComfyNode;
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ComfyWorkflow {
    pub nodes: HashMap<String, ComfyNode>,
}

impl ComfyWorkflow {
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let val: serde_json::Value = serde_json::from_str(&content)?;

        let mut nodes = HashMap::new();

        if let Some(obj) = val.as_object() {
            // Check if ComfyUI prompt/API format
            for (k, v) in obj {
                if let Ok(node) = serde_json::from_value::<ComfyNode>(v.clone()) {
                    nodes.insert(k.clone(), node);
                }
            }
        }

        if nodes.is_empty() {
            // Maybe standard UI exported workflow with "nodes" array
            if let Some(nodes_arr) = val.get("nodes").and_then(|v| v.as_array()) {
                for node_val in nodes_arr {
                    if let Ok(node) = serde_json::from_value::<ComfyNode>(node_val.clone()) {
                        nodes.insert(format!("{}", node.id), node);
                    }
                }
            }
        }

        Ok(Self { nodes })
    }

    pub fn extract_generation_params(&self) -> (String, u32, u32, u32, u32) {
        let mut prompt = String::new();
        let mut width = 704;
        let mut height = 480;
        let mut num_frames = 97;
        let mut steps = 30;

        for node in self.nodes.values() {
            if node.class_type.contains("Sampler") {
                if let Some(w) = node.inputs.get("width").and_then(|v| v.as_u64()) {
                    width = w as u32;
                }
                if let Some(h) = node.inputs.get("height").and_then(|v| v.as_u64()) {
                    height = h as u32;
                }
                if let Some(f) = node.inputs.get("num_frames").and_then(|v| v.as_u64()) {
                    num_frames = f as u32;
                }
                if let Some(s) = node.inputs.get("steps").and_then(|v| v.as_u64()) {
                    steps = s as u32;
                }
            }
            if node.class_type.contains("CLIPTextEncode") || node.class_type.contains("Text") {
                if let Some(text) = node.inputs.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() && prompt.is_empty() {
                        prompt = text.to_string();
                    }
                }
            }
        }

        (prompt, width, height, num_frames, steps)
    }
}
