fn main() {
    // macOS extension-module builds should use dynamic symbol lookup for Python C-API
    // symbols instead of requiring explicit libpython linkage.
    let is_macos = std::env::var("CARGO_CFG_TARGET_OS")
        .map(|v| v == "macos")
        .unwrap_or(false);
    let python_feature_enabled = std::env::var_os("CARGO_FEATURE_PYTHON").is_some();

    if is_macos && python_feature_enabled {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
