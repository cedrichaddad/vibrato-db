//! C-compatible API for DAW plugin integration
//!
//! All functions use `std::panic::catch_unwind` to prevent Rust panics
//! from unwinding into C/C++ code, which would cause undefined behavior
//! (and crash the host DAW).
//!
//! Return values:
//! - 0 = success
//! - -1 = error (call `vibrato_last_error` for details)

use std::ffi::{c_char, CStr, CString};
use std::path::Path;

/// Opaque handle to a Vibrato database
pub struct VibratoHandle {
    // Will hold the actual engine when wired up
    _path: String,
}

// Thread-local last error message
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Open a Vibrato database
///
/// Returns a handle on success, NULL on error.
/// The caller must call `vibrato_close` when done.
///
/// # Safety
/// `path` must be a valid null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn vibrato_open(path: *const c_char) -> *mut VibratoHandle {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            set_last_error("Path is null");
            return std::ptr::null_mut();
        }

        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("Invalid UTF-8 path: {}", e));
                return std::ptr::null_mut();
            }
        };

        if !Path::new(path_str).exists() {
            set_last_error(&format!("File not found: {}", path_str));
            return std::ptr::null_mut();
        }

        let handle = Box::new(VibratoHandle {
            _path: path_str.to_string(),
        });

        Box::into_raw(handle)
    })
    .unwrap_or_else(|_| {
        set_last_error("Panic in vibrato_open");
        std::ptr::null_mut()
    })
}

/// Close a Vibrato database handle and free memory
///
/// # Safety
/// `handle` must be a valid pointer returned by `vibrato_open`,
/// or NULL (in which case this is a no-op).
#[no_mangle]
pub unsafe extern "C" fn vibrato_close(handle: *mut VibratoHandle) {
    let _ = std::panic::catch_unwind(|| {
        if !handle.is_null() {
            unsafe { drop(Box::from_raw(handle)); }
        }
    });
}

/// Get the last error message
///
/// Returns a pointer to a null-terminated C string, or NULL if no error.
/// The returned string is valid until the next API call from the same thread.
///
/// # Safety
/// The returned pointer must not be freed by the caller.
#[no_mangle]
pub extern "C" fn vibrato_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_null_path() {
        let handle = unsafe { vibrato_open(std::ptr::null()) };
        assert!(handle.is_null());
    }

    #[test]
    fn test_open_nonexistent_file() {
        let path = CString::new("/nonexistent/file.vdb").unwrap();
        let handle = unsafe { vibrato_open(path.as_ptr()) };
        assert!(handle.is_null());
    }

    #[test]
    fn test_close_null_handle() {
        // Should not crash
        unsafe { vibrato_close(std::ptr::null_mut()); }
    }

    #[test]
    fn test_open_close_roundtrip() {
        let dir = std::env::temp_dir();
        let path = CString::new(dir.to_str().unwrap()).unwrap();
        let handle = unsafe { vibrato_open(path.as_ptr()) };
        // temp dir exists, so handle should be non-null
        assert!(!handle.is_null());
        unsafe { vibrato_close(handle); }
    }
}
