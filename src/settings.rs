use crate::arguments::StableDiffusionVersion;
use anyhow::Result;
use config::Config;
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct LogSetting {
    pub log_level: String,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct InferenceSetting {
    pub prompt: String,
    pub cpu: Vec<String>,
    pub height: i64,
    pub width: i64,
    pub unet_weights: String,
    pub clip_weights: String,
    pub vae_weights: String,
    pub vocab_file: String,
    pub sliced_attention_size: i64,
    pub n_steps: usize,
    pub seed: i64,
    pub num_samples: i64,
    pub final_image: String,
    pub autocast: bool,
    pub sd_version: StableDiffusionVersion,
    pub intermediary_images: bool,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Settings {
    pub logging: LogSetting,
    pub inference: InferenceSetting,
}

impl Settings {
    pub fn new(location: &str) -> Result<Self> {
        let mut builder = Config::builder();

        if Path::new(location).exists() {
            builder = builder.add_source(config::File::with_name(location));
        } else {
            log::warn!("Configuration file not found");
        }

        let settings = builder.build()?.try_deserialize()?;

        Ok(settings)
    }

    pub fn json_pretty(&self) -> String {
        to_string_pretty(&self).expect("Failed serialize")
    }
}
