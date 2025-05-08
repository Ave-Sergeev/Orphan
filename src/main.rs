mod arguments;
mod diffusion;
mod image;
mod settings;

use crate::arguments::Arguments;
use crate::diffusion::run;
use crate::settings::Settings;
use env_logger::Builder;
use log::LevelFilter;
use std::error::Error;
use std::str::FromStr;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let settings =
        Settings::new("config.yaml").map_err(|err| format!("Failed to load settings: {err}"))?;

    Builder::new()
        .filter_level(
            LevelFilter::from_str(settings.logging.log_level.as_str()).unwrap_or(LevelFilter::Info),
        )
        .init();

    log::info!("Settings:\n{}", settings.json_pretty());

    let args = Arguments::new(settings.inference);

    let start = Instant::now();

    if *args.autocast() {
        tch::autocast(true, || run(&args))?;
    } else {
        run(&args)?;
    }

    let duration = start.elapsed();

    log::info!(
        "Время выполнения: {} секунд, {} миллисекунд",
        duration.as_secs(),
        duration.subsec_millis()
    );

    Ok(())
}
