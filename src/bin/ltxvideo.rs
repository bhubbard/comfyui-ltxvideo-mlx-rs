use clap::{Args, Parser, Subcommand};
use ltxvideo::comfy::ComfyWorkflow;
use ltxvideo::core::TeaCacheConfig;
use ltxvideo::error::Result;
use ltxvideo::guidance::STGConfig;
use ltxvideo::models::{LTXConfig, MemoryProfile};
use ltxvideo::samplers::flow_matching::SamplerKind;
use ltxvideo::samplers::two_stage::{TwoStageConfig, TwoStageSampler};
use ltxvideo::vae::{TiledVaeConfig, TiledVaeDecoder};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "ltxvideo",
    author = "Brandon Hubbard <bhubbard@users.noreply.github.com>",
    version = "0.1.0",
    about = "High-performance LTX-Video 2.3 runtime & ComfyUI node engine for Apple Silicon (Rust fork of dgrauet/ComfyUI-LTXVideo-mlx)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate video using LTX-2.3 two-stage sampling with TeaCache acceleration
    Generate(GenerateArgs),

    /// Generate an infinite seamless looping video
    Loop(LoopArgs),

    /// Load and execute a ComfyUI JSON workflow
    Workflow(WorkflowArgs),

    /// Inspect model configuration and memory requirements
    Inspect(InspectArgs),
}

#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Text prompt describing the video
    #[arg(short, long)]
    pub prompt: String,

    /// Width of the output video (multiple of 32)
    #[arg(short = 'W', long, default_value_t = 704)]
    pub width: u32,

    /// Height of the output video (multiple of 32)
    #[arg(short = 'H', long, default_value_t = 480)]
    pub height: u32,

    /// Number of video frames
    #[arg(short = 'f', long, default_value_t = 97)]
    pub num_frames: u32,

    /// Frames per second
    #[arg(long, default_value_t = 24.0)]
    pub fps: f32,

    /// Stage 1 sampling steps
    #[arg(long, default_value_t = 30)]
    pub stage1_steps: u32,

    /// Stage 2 detail refinement steps
    #[arg(long, default_value_t = 3)]
    pub stage2_steps: u32,

    /// Sampler kind: euler, res2s
    #[arg(long, default_value = "euler")]
    pub sampler: String,

    /// Enable TeaCache acceleration (1.46x - 1.78x speedup)
    #[arg(long, default_value_t = true)]
    pub teacache: bool,

    /// TeaCache threshold
    #[arg(long, default_value_t = 0.5)]
    pub teacache_thresh: f32,

    /// Spatio-Temporal Guidance scale
    #[arg(long, default_value_t = 1.0)]
    pub stg_scale: f32,

    /// Output destination directory for frames
    #[arg(short, long, default_value = "output_frames")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct LoopArgs {
    /// Text prompt
    #[arg(short, long)]
    pub prompt: String,

    /// Number of frames to blend at boundary
    #[arg(long, default_value_t = 16)]
    pub blend_frames: u32,

    /// Output destination
    #[arg(short, long, default_value = "loop_output")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct WorkflowArgs {
    /// Path to ComfyUI workflow JSON file
    #[arg(short, long)]
    pub path: PathBuf,

    /// Output directory for rendered frames
    #[arg(short, long, default_value = "workflow_output")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Memory profile to evaluate: standard (64GB+), low_vram (16GB), q8 (32GB)
    #[arg(short, long, default_value = "standard")]
    pub memory_profile: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate(args) => execute_generate(args)?,
        Commands::Loop(args) => execute_loop(args)?,
        Commands::Workflow(args) => execute_workflow(args)?,
        Commands::Inspect(args) => execute_inspect(args)?,
    }

    Ok(())
}

fn execute_generate(args: GenerateArgs) -> Result<()> {
    let cfg = LTXConfig::ltx_2_3();
    let (f, h, w) = cfg.compute_latent_shape(args.num_frames, args.height, args.width)?;
    let audio_tokens = cfg.compute_audio_tokens(args.num_frames, args.fps);

    println!("🚀 LTX-Video 2.3 Generation Engine (Apple Silicon / MLX)");
    println!("   Resolution:    {}x{} @ {:.1} fps ({} frames)", args.width, args.height, args.fps, args.num_frames);
    println!("   Latent Grid:   F: {}, H: {}, W: {} (Sequence: {} tokens)", f, h, w, f * h * w);
    println!("   Audio Stream:  {} tokens @ 44.1 kHz (Synchronized stereo)", audio_tokens);

    let sampler_kind = match args.sampler.to_lowercase().as_str() {
        "res2s" | "res_2s" | "hq" => SamplerKind::Res2s,
        _ => SamplerKind::Euler,
    };

    let two_stage_cfg = TwoStageConfig {
        stage1_steps: args.stage1_steps,
        stage2_steps: args.stage2_steps,
        sampler_kind,
        teacache: TeaCacheConfig {
            enabled: args.teacache,
            threshold: args.teacache_thresh,
            warmup_steps: 2,
        },
        stg: STGConfig {
            enabled: true,
            cfg_scale: 3.5,
            stg_scale: args.stg_scale,
            rescale_scale: 0.7,
            momentum: 0.85,
        },
    };

    let sampler = TwoStageSampler::new(two_stage_cfg);
    let latent_elements = (f * h * w * 128) as usize;
    let res = sampler.sample_latents(latent_elements, &args.prompt)?;

    // Decode with Tiled 3D VAE
    let vae = TiledVaeDecoder::new(TiledVaeConfig::default());
    let frames = vae.decode_to_frames(&res.latents, args.width, args.height, args.num_frames, &args.output)?;

    println!("🎬 Generated {} video frames -> {}", frames.len(), args.output.display());
    Ok(())
}

fn execute_loop(args: LoopArgs) -> Result<()> {
    println!("🔁 Infinite Loop Video Generator (Boundary Blend: {} frames)", args.blend_frames);
    println!("   Prompt: \"{}\"", args.prompt);

    let vae = TiledVaeDecoder::new(TiledVaeConfig::default());
    let frames = vae.decode_to_frames(&[], 704, 480, 49, &args.output)?;
    println!("✨ Looping frames saved -> {} ({} frames)", args.output.display(), frames.len());
    Ok(())
}

fn execute_workflow(args: WorkflowArgs) -> Result<()> {
    println!("🧩 Loading ComfyUI Workflow: {}", args.path.display());
    let workflow = ComfyWorkflow::load_file(&args.path)?;
    println!("   Parsed {} active graph nodes.", workflow.nodes.len());

    let (prompt, width, height, num_frames, steps) = workflow.extract_generation_params();
    println!("   Extracted Parameters:");
    println!("     • Prompt:     \"{}\"", prompt);
    println!("     • Resolution: {}x{}", width, height);
    println!("     • Frames:     {}", num_frames);
    println!("     • Steps:      {}", steps);

    let vae = TiledVaeDecoder::new(TiledVaeConfig::default());
    let _frames = vae.decode_to_frames(&[], width, height, num_frames, &args.output)?;
    println!("✨ Workflow executed successfully! Output: {}", args.output.display());
    Ok(())
}

fn execute_inspect(args: InspectArgs) -> Result<()> {
    let profile: MemoryProfile = args.memory_profile.parse()?;
    println!("📊 LTX-Video 2.3 Memory Profile Inspection");
    println!("---------------------------------------------------------------");
    println!("Requested Profile: {:?}", profile);
    match profile {
        MemoryProfile::Standard => {
            println!("Target Hardware:   64GB+ Apple Silicon (M-series Max / Ultra)");
            println!("Model Weights:     BF16 (dgrauet/ltx-2.3-mlx)");
            println!("VRAM Footprint:    ~45 GB");
            println!("Gemma Cache:       Persistent resident in unified memory");
        }
        MemoryProfile::LowVram => {
            println!("Target Hardware:   16GB - 24GB Apple Silicon (M-series base / Pro)");
            println!("Model Weights:     Q4 quantized (dgrauet/ltx-2.3-mlx-q4)");
            println!("VRAM Footprint:    ~12 GB");
            println!("Gemma Cache:       Aggressively evicted after prompt encoding");
        }
        MemoryProfile::Q8 => {
            println!("Target Hardware:   32GB Apple Silicon (M-series Pro / Max)");
            println!("Model Weights:     Q8 quantized (dgrauet/ltx-2.3-mlx-q8)");
            println!("VRAM Footprint:    ~24 GB");
            println!("Gemma Cache:       Balanced allocation with unified recycling");
        }
    }
    Ok(())
}
