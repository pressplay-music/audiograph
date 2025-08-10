fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("constants.rs");

    // The default value of MAX_CHANNELS can be overridden by an environment variable.
    const DEFAULT_MAX_CHANNELS: usize = 64;

    let max_channels: usize = std::env::var("MAX_CHANNELS")
        .map(|s| s.parse().unwrap_or(DEFAULT_MAX_CHANNELS))
        .unwrap_or(DEFAULT_MAX_CHANNELS);

    std::fs::write(
        &dest_path,
        format!("pub const MAX_CHANNELS: usize = {};", max_channels),
    )
    .unwrap();

    println!("cargo:rerun-if-changed=build.rs");
}
