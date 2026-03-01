//! FFI Type definitions and utilities

use std::ffi::CString;
use std::ptr;

use crate::{Qwen3AsrDevice, Qwen3AsrStreamOptions};

impl Default for Qwen3AsrStreamOptions {
    fn default() -> Self {
        Self {
            language: ptr::null(),
            chunk_size_sec: 0.5,
            enable_timestamps: true,
            enable_partial_results: true,
        }
    }
}

impl Clone for Qwen3AsrStreamOptions {
    fn clone(&self) -> Self {
        Self {
            language: if self.language.is_null() {
                ptr::null()
            } else {
                self.language
            },
            chunk_size_sec: self.chunk_size_sec,
            enable_timestamps: self.enable_timestamps,
            enable_partial_results: self.enable_partial_results,
        }
    }
}

impl Default for Qwen3AsrDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Helper to create a CString and get its raw pointer
pub fn to_c_string(s: &str) -> Option<CString> {
    CString::new(s).ok()
}

/// Helper to convert a C string pointer to a Rust string
///
/// # Safety
/// The pointer must be a valid null-terminated UTF-8 string
pub unsafe fn from_c_string(ptr: *const libc::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    std::ffi::CStr::from_ptr(ptr)
        .to_str()
        .ok()
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_stream_options() {
        let opts = Qwen3AsrStreamOptions::default();
        assert!(opts.language.is_null());
        assert_eq!(opts.chunk_size_sec, 0.5);
        assert!(opts.enable_timestamps);
        assert!(opts.enable_partial_results);
    }

    #[test]
    fn test_default_device() {
        let device = Qwen3AsrDevice::default();
        assert_eq!(device, Qwen3AsrDevice::Cpu);
    }
}
