use clap::Parser;
use cpal::{traits::*, Device, Host, Stream};
use std::time::Duration;

#[derive(Parser)]
struct Args {
    /// Output device name
    device: String,
    
    /// Frequency in Hz
    frequency: f32,
    
    /// Gain (0.0 to 1.0)
    gain: f32,
}

fn main() {
    let args = Args::parse();
    
    println!("Device: {}", args.device);
    println!("Frequency: {} Hz", args.frequency);
    println!("Gain: {}", args.gain);
    
    // Initialize CPAL
    let host = cpal::default_host();
    let device = find_device(&host, &args.device).expect("Failed to find device");
    let config = device.default_output_config().expect("Failed to get default config");
    
    println!("Using device: {}", device.name().unwrap_or("Unknown".to_string()));
    println!("Config: {:?}", config);
    
    // TODO: Create audio graph with SineWaveGen and Gain nodes
    // TODO: Set up audio callback
    // TODO: Run for 4 seconds
    
    std::thread::sleep(Duration::from_secs(4));
    println!("Done!");
}

fn find_device(host: &Host, name: &str) -> Option<Device> {
    host.output_devices()
        .ok()?
        .find(|device| {
            device.name()
                .map(|device_name| device_name.contains(name))
                .unwrap_or(false)
        })
}