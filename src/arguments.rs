use crate::settings::InferenceSetting;
use clap::{Parser, ValueEnum};
use getset::Getters;
use serde_derive::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone, Copy, ValueEnum, Default)]
pub enum StableDiffusionVersion {
    #[serde(rename = "1.5")]
    V1_5,
    #[serde(rename = "2.1")]
    #[default]
    V2_1,
}

#[derive(Parser, Getters, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Arguments {
    /// The prompt to be used for image generation.
    #[arg(long)]
    #[get = "pub"]
    prompt: String,

    /// When set, use the CPU for the listed devices, can be 'all', 'unet', 'clip', 'vae'.
    /// Multiple values can be set.
    #[arg(long)]
    #[get = "pub"]
    cpu: Vec<String>,

    /// The height in pixels of the generated image.
    #[arg(long)]
    #[get = "pub"]
    height: i64,

    /// The width in pixels of the generated image.
    #[arg(long)]
    #[get = "pub"]
    width: i64,

    /// The `UNet` weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    #[get = "pub"]
    unet_weights: String,

    /// The `CLIP` weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    #[get = "pub"]
    clip_weights: String,

    /// The `VAE` weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    #[get = "pub"]
    vae_weights: String,

    /// The file specifying the vocabulary to used for tokenization.
    #[arg(long, value_name = "FILE")]
    #[get = "pub"]
    vocab_file: String,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default).
    #[arg(long)]
    #[get = "pub"]
    sliced_attention_size: i64,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    #[get = "pub"]
    n_steps: usize,

    /// The random seed to be used for the generation.
    #[arg(long, default_value_t = 50)]
    #[get = "pub"]
    seed: i64,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    #[get = "pub"]
    num_samples: i64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_image.png")]
    #[get = "pub"]
    final_image: String,

    /// Use autocast (disabled by default as it may use more memory in some cases).
    #[arg(long, action)]
    #[get = "pub"]
    autocast: bool,

    /// Version of `Stable Diffusion` model.
    #[arg(long, value_enum, default_value = "v2-1")]
    #[get = "pub"]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    #[get = "pub"]
    intermediary_images: bool,
}

impl Arguments {
    pub fn new(inference: InferenceSetting) -> Self {
        Arguments {
            prompt: inference.prompt,
            cpu: inference.cpu,
            height: inference.height,
            width: inference.width,
            unet_weights: inference.unet_weights,
            clip_weights: inference.clip_weights,
            vae_weights: inference.vae_weights,
            vocab_file: inference.vocab_file,
            sliced_attention_size: inference.sliced_attention_size,
            n_steps: inference.n_steps,
            seed: inference.seed,
            num_samples: inference.num_samples,
            final_image: inference.final_image,
            autocast: inference.autocast,
            sd_version: inference.sd_version,
            intermediary_images: inference.intermediary_images,
        }
    }
}
