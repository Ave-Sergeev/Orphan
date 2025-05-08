use crate::arguments::StableDiffusionVersion;
use crate::image::output_filename;
use crate::Arguments;
use anyhow::{Error, Result};
use diffusers::models::vae::AutoEncoderKL;
use diffusers::pipelines::stable_diffusion::StableDiffusionConfig;
use diffusers::transformers::clip::Tokenizer;
use diffusers::utils::DeviceSetup;
use std::ops::{Add, Div, Mul, Sub};
use tch::nn::Module;
use tch::vision::image;
use tch::{Device, Kind, Tensor};

pub fn run(args: &Arguments) -> Result<(), Error> {
    tch::maybe_init_cuda();

    let in_channels = 4;
    let guidance_scale = 7.5_f64;

    let clip_w = args.clip_weights();
    let vae_w = args.vae_weights();
    let unet_w = args.unet_weights();
    let vocab = args.vocab_file();

    let sliced_attention_size = *args.sliced_attention_size();
    let intermediary_images = *args.intermediary_images();
    let sd_version = args.sd_version();
    let image_path = args.final_image();
    let num_samples = args.num_samples();
    let n_steps = *args.n_steps();
    let prompt = args.prompt();
    let cpu = args.cpu();
    let height = *args.height();
    let width = *args.width();
    let seed = args.seed();

    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => {
            StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(height), Some(width))
        }
        StableDiffusionVersion::V2_1 => {
            StableDiffusionConfig::v2_1(Some(sliced_attention_size), Some(height), Some(width))
        }
    };

    let device_setup = DeviceSetup::new(cpu.clone());
    let clip_device = device_setup.get("clip");
    let unet_device = device_setup.get("unet");
    let vae_device = device_setup.get("vae");

    let scheduler = sd_config.build_scheduler(n_steps);
    let tokenizer = Tokenizer::create(vocab, &sd_config.clip)?;

    log::info!("Running with prompt \"{prompt}\".");
    let tensor = encode_text_to_tensor(&tokenizer, prompt, clip_device)?;
    let uncond_tensor = encode_text_to_tensor(&tokenizer, "", clip_device)?;

    let no_grad_guard = tch::no_grad_guard();

    log::info!("Building the UNet");
    let unet = sd_config.build_unet(unet_w, unet_device, in_channels)?;

    log::info!("Building the AutoEncoder");
    let vae = sd_config.build_vae(vae_w, vae_device)?;

    log::info!("Building the Clip transformer");
    let text_model = sd_config.build_clip_transformer(clip_w, clip_device)?;

    let text_embeddings = text_model.forward(&tensor);
    let uncond_embeddings = text_model.forward(&uncond_tensor);
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(unet_device);

    for idx in 0..*num_samples {
        tch::manual_seed(seed + idx);

        let mut latents = Tensor::randn(
            [1, 4, sd_config.height / 8, sd_config.width / 8],
            (Kind::Float, unet_device),
        );

        latents *= scheduler.init_noise_sigma();

        for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
            log::info!("Timestep {}/{}", timestep_index + 1, n_steps);

            let doubled_latent_tensor = Tensor::cat(&[&latents, &latents], 0);
            let scaled_latent_input = scheduler.scale_model_input(doubled_latent_tensor, timestep);

            let noise_pred_chunked = unet
                .forward(&scaled_latent_input, timestep as f64, &text_embeddings)
                .chunk(2, 0);
            let (noise_pred_uncond, noise_pred_text) =
                (&noise_pred_chunked[0], &noise_pred_chunked[1]);
            let noise_pred =
                noise_pred_uncond.add(noise_pred_text.sub(noise_pred_uncond).mul(guidance_scale));

            latents = scheduler.step(&noise_pred, timestep, &latents);

            if intermediary_images {
                let image = decode_latents_to_image(&vae, &mut latents, vae_device);

                let final_image_path =
                    output_filename(image_path, idx + 1, *num_samples, Some(timestep_index + 1));

                image::save(&image, final_image_path)?;
            }
        }

        log::info!(
            "Generating the final image for sample {}/{}.",
            idx + 1,
            num_samples
        );

        let image = decode_latents_to_image(&vae, &mut latents, vae_device);

        let final_image_path = output_filename(image_path, idx + 1, *num_samples, None);

        image::save(&image, final_image_path)?;
    }

    drop(no_grad_guard);

    Ok(())
}

fn decode_latents_to_image(
    vae: &AutoEncoderKL,
    latents: &mut Tensor,
    vae_device: Device,
) -> Tensor {
    vae.decode(&latents.to(vae_device).div(0.18215))
        .div(2.0)
        .add(0.5)
        .clamp(0.0, 1.0)
        .to_device(Device::Cpu)
        .mul(255.0)
        .to_kind(Kind::Uint8)
}

fn encode_text_to_tensor(
    tokenizer: &Tokenizer,
    text: &str,
    device: Device,
) -> Result<Tensor, Error> {
    let data = tokenizer
        .encode(text)?
        .into_iter()
        .map(|token| i64::try_from(token).expect("token value out of range for i64"))
        .collect::<Vec<_>>();

    Ok(Tensor::from_slice(&data).view((1, -1)).to(device))
}
