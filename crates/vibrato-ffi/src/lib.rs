//! Vibrato FFI – Python (PyO3), C, and WASM bindings
//!
//! All C-API entry points wrap Rust calls in `std::panic::catch_unwind`
//! to prevent panics from crashing host processes (DAWs, Python interpreters).

pub mod c_api;

#[cfg(feature = "python")]
pub mod python;
