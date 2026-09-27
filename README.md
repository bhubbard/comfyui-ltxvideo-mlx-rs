# comfyui-ltxvideo-mlx-rs (`ltxvideo`)

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-edition%202021-orange.svg)](Cargo.toml)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4%20%7C%20M5-black.svg?logo=apple)](https://apple.com)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Native%20Workflow%20Engine-9333EA.svg)](https://comfy.org)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-06B6D4.svg)](https://code.brandonhubbard.com/comfyui-ltxvideo-mlx-rs/)

**`comfyui-ltxvideo-mlx-rs`** is a high-performance, native Rust generative video runtime, ComfyUI node engine, and DiT sampler for **Apple Silicon**.

It is a pure Rust fork and systems-level reimplementation of [**`dgrauet/ComfyUI-LTXVideo-mlx`**](https://github.com/dgrauet/ComfyUI-LTXVideo-mlx) (bringing Lightricks' 22B parameter **LTX-Video 2.3** architecture to Apple Silicon unified memory). It provides both an embeddable Rust library (`ltxvideo`), a standalone command-line executable (`ltxvideo`), and a native ComfyUI JSON workflow executor.

---

## ⚡ Key Highlights & Architecture

- **Native Rust Zero-Python Overhead**: Eliminates Python interpreter overhead, PyTorch lock contention, and GIL lag when driving video diffusion on Apple Silicon.
- **Multimodal Video + Synchronized Stereo Audio**:
  - Synthesizes 24 fps video frames alongside synchronized stereo audio (speech, musical score, and ambient Foley effects) in an interleaved multimodal token stream ($T_{audio} \approx 50 \text{ tokens/sec}$).
- **Two-Stage Sampler & High-Quality Refinement**:
  - **Stage 1**: Base downsampled latent flow-matching generation (Euler or `res_2s` second-order Runge-Kutta ODE solver).
  - **Stage 2**: High-frequency detail upsampling and temporal stabilization.
- **TeaCache Acceleration (1.46× – 1.78× Speedup)**:
  - Implements **Timestep Embedding Activation Caching**: dynamically tracks transformer activation velocity $\Delta = ||\text{act}_t - \text{act}_{t-1}|| / ||\text{act}_{t-1}||$.
  - Skips redundant DiT blocks when velocity falls below threshold (`0.5` for Euler, `1.0` for `res_2s`), yielding massive latency savings with zero perceptible quality degradation.
- **Spatio-Temporal Guidance (STG) & Adaptive Projected Guidance (APG)**:
  - Decouples spatial texture guidance from temporal motion guidance to eliminate motion freezing and flickering artifacts.
  - Standard-deviation rescaling and momentum smoothing buffers.
- **3D Space-Time Rotary Position Embeddings (RoPE)**:
  - Multi-axis frequency calculation over temporal ($F$), vertical ($H$), and horizontal ($W$) latent grids.
- **Tiled 3D Causal VAE Decoding**:
  - Spatial and temporal window tiling with cosine boundary blending to decode ultra-high resolution clips within constrained unified memory budgets.
- **Memory Profiles for Every Mac**:
  - **Standard (64GB+ Macs)**: Full BF16 weights (`dgrauet/ltx-2.3-mlx`) with persistent resident Gemma text encoder.
  - **Q8 (32GB Macs)**: Q8 quantized weights (`dgrauet/ltx-2.3-mlx-q8`) with balanced unified memory recycling.
  - **LowVRAM (16GB – 24GB Macs)**: Q4 quantized weights (`dgrauet/ltx-2.3-mlx-q4`) with aggressive memory eviction (~12 GB peak footprint).
- **Direct ComfyUI Workflow Execution**:
  - Parses and executes standard ComfyUI JSON workflow export files (e.g. `MLX_T2V_I2V_Two_Stage_HQ.json` and `MLX_Extend_Video.json`) without needing a Python ComfyUI server.

---

## 🚀 Installation & Build

### From Source
```bash
git clone https://github.com/bhubbard/comfyui-ltxvideo-mlx-rs.git
cd comfyui-ltxvideo-mlx-rs
cargo build --release
```
The compiled executable will be located at `target/release/ltxvideo`.

---

## 🛠️ Command-Line Interface (`ltxvideo`)

### 1. Two-Stage Video Generation with TeaCache
```bash
ltxvideo generate \
  --prompt "A sleek cybernetic falcon soaring above glass skyscrapers in a foggy neon city at dawn" \
  --width 704 --height 480 \
  --num-frames 97 --fps 24.0 \
  --stage1-steps 30 --stage2-steps 3 \
  --sampler euler \
  --teacache --teacache-thresh 0.5 \
  --stg-scale 1.0 \
  --output ./falcon_frames
```

### 2. High-Quality 2nd-Order Generation (`res_2s`)
```bash
ltxvideo generate \
  --prompt "Cinematic underwater footage of bioluminescent jellyfish drifting through deep ocean trenches" \
  --sampler res2s \
  --stage1-steps 15 --stage2-steps 3 \
  --teacache --teacache-thresh 1.0 \
  --output ./jellyfish_frames
```

### 3. Infinite Seamless Video Looping
```bash
ltxvideo loop \
  --prompt "Gentle ocean waves lapping against a dark volcanic sand beach, looping" \
  --blend-frames 16 \
  --output ./looping_waves
```

### 4. Execute a ComfyUI Workflow Directly
```bash
ltxvideo workflow \
  --path ./workflows/MLX_T2V_I2V_Two_Stage_HQ.json \
  --output ./rendered_workflow
```

### 5. Inspect Hardware Memory Profiles
```bash
# Low VRAM profile for 16GB / 24GB Macs
ltxvideo inspect --memory-profile low_vram

# Standard profile for 64GB+ Macs
ltxvideo inspect --memory-profile standard
```

---

## 💻 Programmatic Rust API

Embed LTX-Video generation directly in your Rust applications:

```rust
use ltxvideo::{
    models::{LTXConfig, MemoryProfile},
    samplers::two_stage::{TwoStageConfig, TwoStageSampler},
    samplers::flow_matching::SamplerKind,
    core::teacache::TeaCacheConfig,
    guidance::stg::STGConfig,
    vae::{TiledVaeConfig, TiledVaeDecoder},
};
use std::path::Path;

fn main() -> ltxvideo::Result<()> {
    let cfg = LTXConfig::ltx_2_3();
    let (f, h, w) = cfg.compute_latent_shape(97, 480, 704)?;
    let latent_elements = (f * h * w * 128) as usize;

    let sampler = TwoStageSampler::new(TwoStageConfig {
        stage1_steps: 30,
        stage2_steps: 3,
        sampler_kind: SamplerKind::Euler,
        teacache: TeaCacheConfig {
            enabled: true,
            threshold: 0.5,
            warmup_steps: 2,
        },
        stg: STGConfig::default(),
    });

    let res = sampler.sample_latents(latent_elements, "A hypercar speeding through rain")?;
    println!("Sampled in {:.2}s (Speedup: {:.2}x)", res.elapsed_sec, res.teacache_speedup);

    let vae = TiledVaeDecoder::new(TiledVaeConfig::default());
    let frames = vae.decode_to_frames(&res.latents, 704, 480, 97, Path::new("./output"))?;
    println!("Decoded {} frames!", frames.len());

    Ok(())
}
```

---

## 🧪 Test Suite

Run the full integration test suite:
```bash
cargo test
```

Verification includes:
- ✅ Latent grid coordinate math ($F, H, W$) & audio token count calculation
- ✅ 3D space-time Rotary Position Embeddings (RoPE) frequencies
- ✅ TeaCache activation velocity thresholding & block skipping
- ✅ Spatio-Temporal Guidance (STG) noise amplification and std rescaling
- ✅ Distilled 8-step monotonic sigmas
- ✅ Seamless looping boundary cosine interpolation
- ✅ ComfyUI JSON workflow graph parsing & parameter extraction

---

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.

---

## 🙏 Acknowledgements

- **[dgrauet/ComfyUI-LTXVideo-mlx](https://github.com/dgrauet/ComfyUI-LTXVideo-mlx)** by David Grauet
- **[Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)**
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**
- **Apple Silicon MLX Team**
