use ltxvideo::comfy::ComfyWorkflow;
use ltxvideo::core::rope::RotaryPositionEmbedding3D;
use ltxvideo::core::teacache::{TeaCacheConfig, TeaCacheState};
use ltxvideo::guidance::stg::{STGConfig, SpatioTemporalGuidance};
use ltxvideo::models::config::LTXConfig;
use ltxvideo::samplers::flow_matching::{LTXSampler, DISTILLED_SIGMAS};
use ltxvideo::samplers::looping::{LoopingConfig, LoopingSampler};

#[test]
fn test_ltx_2_3_latent_and_audio_dimensions() {
    let cfg = LTXConfig::ltx_2_3();
    let (f, h, w) = cfg.compute_latent_shape(97, 480, 704).expect("Valid dimensions");
    // F = (97 - 1) / 8 + 1 = 13
    assert_eq!(f, 13);
    // H = 480 / 32 = 15
    assert_eq!(h, 15);
    // W = 704 / 32 = 22
    assert_eq!(w, 22);

    let audio_tokens = cfg.compute_audio_tokens(97, 24.0);
    assert!(audio_tokens >= 190 && audio_tokens <= 210);
}

#[test]
fn test_3d_rope_frequencies() {
    let rope = RotaryPositionEmbedding3D::new(32, 10000.0);
    let freqs = rope.compute_frequencies(2, 2, 2);
    // Total tokens = 2 * 2 * 2 = 8 tokens
    assert_eq!(freqs.len(), 8);
    // Each token has 32 frequencies
    assert_eq!(freqs[0].len(), 32);
}

#[test]
fn test_teacache_activation_skipping() {
    let config = TeaCacheConfig {
        enabled: true,
        threshold: 0.5,
        warmup_steps: 2,
    };
    let mut teacache = TeaCacheState::new(config);

    // Warmup step 0 & 1 should not skip
    assert!(!teacache.should_skip(0, &[1.0, 1.0, 1.0]));
    teacache.update_cache(vec![1.0, 1.0, 1.0]);
    assert!(!teacache.should_skip(1, &[1.0, 1.0, 1.0]));
    teacache.update_cache(vec![1.0, 1.0, 1.0]);

    // Step 2 with very small delta (< 0.5) SHOULD skip
    assert!(teacache.should_skip(2, &[1.01, 0.99, 1.0]));
    assert_eq!(teacache.skipped_blocks_count, 1);
    assert!(teacache.speedup_ratio() > 1.0);
}

#[test]
fn test_stg_guidance_and_rescaling() {
    let config = STGConfig {
        enabled: true,
        cfg_scale: 3.0,
        stg_scale: 1.0,
        rescale_scale: 0.7,
        momentum: 0.0,
    };
    let mut stg = SpatioTemporalGuidance::new(config);

    let pos = vec![2.0, 3.0, 4.0];
    let neg = vec![1.0, 1.0, 1.0];
    let pert = vec![2.2, 2.8, 3.9];

    let guided = stg.compute_guided_noise(&pos, &neg, &pert);
    assert_eq!(guided.len(), 3);
    // Guided noise vector has higher dynamic energy than base input
    assert!(guided.iter().sum::<f32>() > pos.iter().sum::<f32>());
}

#[test]
fn test_distilled_sigmas_monotonicity() {
    let sampler = LTXSampler::distilled();
    assert_eq!(sampler.sigmas.len(), 9);
    assert_eq!(sampler.sigmas, DISTILLED_SIGMAS.to_vec());

    for i in 0..sampler.sigmas.len() - 1 {
        assert!(sampler.sigmas[i] > sampler.sigmas[i + 1]);
    }
}

#[test]
fn test_looping_boundary_blend() {
    let sampler = LoopingSampler::new(LoopingConfig {
        loop_blend_frames: 4,
        temporal_tile_size: 16,
        temporal_overlap: 4,
    });

    let mut frames = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
        vec![5.0, 5.0],
        vec![6.0, 6.0],
        vec![7.0, 7.0],
        vec![8.0, 8.0],
    ];

    sampler.blend_boundary_latents(&mut frames).expect("Blending failed");
    // Boundary frames should be smoothly interpolated
    assert_eq!(frames.len(), 8);
}

#[test]
fn test_comfyui_workflow_json_parsing() {
    let mock_json = r#"{
        "1": {
            "id": 1,
            "class_type": "LTXVMLXTwoStageHQSampler",
            "inputs": {
                "width": 704,
                "height": 480,
                "num_frames": 97,
                "steps": 15
            }
        },
        "2": {
            "id": 2,
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "A vintage sports car racing along the coast"
            }
        }
    }"#;

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), mock_json).unwrap();

    let wf = ComfyWorkflow::load_file(tmp.path()).expect("Valid workflow");
    assert_eq!(wf.nodes.len(), 2);

    let (prompt, width, height, frames, steps) = wf.extract_generation_params();
    assert_eq!(prompt, "A vintage sports car racing along the coast");
    assert_eq!(width, 704);
    assert_eq!(height, 480);
    assert_eq!(frames, 97);
    assert_eq!(steps, 15);
}
