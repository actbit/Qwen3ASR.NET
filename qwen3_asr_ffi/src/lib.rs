//! Qwen3-ASR FFI - Thin C-compatible wrapper for qwen3-asr-rs
//!
//! This crate provides minimal C-compatible bindings for the Qwen3-ASR library.
//! All business logic (streaming, buffering, async) should be implemented on the .NET side.

use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

use candle_core::Device;
use qwen3_asr::{AudioInput, LoadOptions, Qwen3Asr, TranscribeOptions, Batch};
use serde::{Deserialize, Serialize};

// ============================================================================
// FFI Type Definitions
// ============================================================================

/// Opaque handle to an ASR instance
pub type Qwen3AsrHandle = *mut Qwen3Asr;

/// Device type for inference
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3AsrDevice {
    Cpu = 0,
    Cuda = 1,
    Metal = 2,
}

/// Result codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3AsrResultCode {
    Success = 0,
    InvalidHandle = 1,
    InvalidParameter = 2,
    ModelNotLoaded = 3,
    InferenceError = 4,
    MemoryError = 5,
    UnknownError = 99,
}

/// Load options for model initialization
#[repr(C)]
pub struct Qwen3AsrLoadOptions {
    pub device: Qwen3AsrDevice,
    pub model_path: *const libc::c_char,
}

/// Transcribe options
#[repr(C)]
pub struct Qwen3AsrTranscribeOptions {
    pub context: *const libc::c_char,
    pub language: *const libc::c_char,
    pub return_timestamps: bool,
    pub max_new_tokens: libc::c_int,
    pub max_batch_size: libc::c_int,
    pub chunk_max_sec: libc::c_float,
    pub bucket_by_length: bool,
}

/// Transcription result (JSON format for flexibility)
#[repr(C)]
pub struct Qwen3AsrResult {
    /// JSON result: {"text": "...", "language": "...", "timestamps": [...]}
    pub json: *mut libc::c_char,
    /// Result code
    pub code: Qwen3AsrResultCode,
    /// Error message (null if success)
    pub error: *mut libc::c_char,
}

/// Transcription detail (for JSON serialization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionJson {
    pub text: String,
    pub language: String,
    pub timestamps: Option<serde_json::Value>,
}

// ============================================================================
// Helper functions
// ============================================================================

fn create_device(device_type: Qwen3AsrDevice) -> anyhow::Result<Device> {
    match device_type {
        Qwen3AsrDevice::Cpu => Ok(Device::Cpu),
        Qwen3AsrDevice::Cuda => Device::new_cuda(0).map_err(|e| anyhow::anyhow!("{}", e)),
        Qwen3AsrDevice::Metal => Device::new_metal(0).map_err(|e| anyhow::anyhow!("{}", e)),
    }
}

fn parse_c_string(ptr: *const libc::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_string()) }
}

fn to_c_string(s: String) -> *mut libc::c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

fn error_result(code: Qwen3AsrResultCode, msg: &str) -> Qwen3AsrResult {
    Qwen3AsrResult {
        json: ptr::null_mut(),
        code,
        error: to_c_string(msg.to_string()),
    }
}

fn success_result(json: TranscriptionJson) -> Qwen3AsrResult {
    Qwen3AsrResult {
        json: to_c_string(serde_json::to_string(&json).unwrap_or_default()),
        code: Qwen3AsrResultCode::Success,
        error: ptr::null_mut(),
    }
}

// ============================================================================
// FFI Functions - Model Management
// ============================================================================

/// Load model and return handle
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_load(
    model_path: *const libc::c_char,
    device: Qwen3AsrDevice,
    error_out: *mut *mut libc::c_char,
) -> Qwen3AsrHandle {
    let model_path = match parse_c_string(model_path) {
        Some(s) => s,
        None => {
            if !error_out.is_null() {
                *error_out = to_c_string("model_path is null or invalid".to_string());
            }
            return ptr::null_mut();
        }
    };

    let candle_device = match create_device(device) {
        Ok(d) => d,
        Err(e) => {
            if !error_out.is_null() {
                *error_out = to_c_string(format!("Failed to create device: {}", e));
            }
            return ptr::null_mut();
        }
    };

    let load_opts = LoadOptions::default();

    match Qwen3Asr::from_pretrained(&model_path, &candle_device, &load_opts) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(e) => {
            if !error_out.is_null() {
                *error_out = to_c_string(format!("Failed to load model: {}", e));
            }
            ptr::null_mut()
        }
    }
}

/// Destroy an ASR instance
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_destroy(handle: Qwen3AsrHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Check if model is loaded (handle is valid)
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_is_loaded(handle: Qwen3AsrHandle) -> bool {
    !handle.is_null()
}

/// Get supported languages as JSON array
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_supported_languages(
    handle: Qwen3AsrHandle,
) -> *mut libc::c_char {
    if handle.is_null() {
        return to_c_string("[]".to_string());
    }

    let model = &*handle;
    let languages: Vec<&str> = model.supported_languages().to_vec();
    to_c_string(serde_json::to_string(&languages).unwrap_or_else(|_| "[]".to_string()))
}

// ============================================================================
// FFI Functions - Transcription
// ============================================================================

/// Transcribe audio samples (f32, mono)
///
/// sample_rate: Audio sample rate (will be resampled to 16kHz if needed)
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_transcribe(
    handle: Qwen3AsrHandle,
    samples: *const libc::c_float,
    sample_count: libc::size_t,
    sample_rate: libc::c_uint,
    options: *const Qwen3AsrTranscribeOptions,
) -> Qwen3AsrResult {
    if handle.is_null() {
        return error_result(Qwen3AsrResultCode::InvalidHandle, "Handle is null");
    }

    if samples.is_null() && sample_count > 0 {
        return error_result(Qwen3AsrResultCode::InvalidParameter, "samples is null but count > 0");
    }

    let audio_slice = if sample_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(samples, sample_count)
    };

    // Build options
    let opts = if options.is_null() {
        Qwen3AsrTranscribeOptions {
            context: ptr::null(),
            language: ptr::null(),
            return_timestamps: false,
            max_new_tokens: 0,
            max_batch_size: 0,
            chunk_max_sec: 0.0,
            bucket_by_length: false,
        }
    } else {
        (*options).clone()
    };

    let context = parse_c_string(opts.context).unwrap_or_default();
    let language = parse_c_string(opts.language);

    let transcribe_opts = TranscribeOptions {
        context: Batch::one(context),
        language: Batch::one(language),
        return_timestamps: opts.return_timestamps,
        max_new_tokens: if opts.max_new_tokens > 0 { opts.max_new_tokens as usize } else { 0 },
        max_batch_size: if opts.max_batch_size > 0 { opts.max_batch_size as usize } else { 32 },
        chunk_max_sec: if opts.chunk_max_sec > 0.0 { Some(opts.chunk_max_sec) } else { None },
        bucket_by_length: opts.bucket_by_length,
    };

    let model = &*handle;
    let audio_input = AudioInput::Waveform {
        samples: audio_slice,
        sample_rate,
    };

    match model.transcribe(vec![audio_input], transcribe_opts) {
        Ok(results) => {
            if let Some(result) = results.first() {
                success_result(TranscriptionJson {
                    text: result.text.clone(),
                    language: result.language.clone(),
                    timestamps: result.timestamps.clone(),
                })
            } else {
                error_result(Qwen3AsrResultCode::InferenceError, "No transcription result")
            }
        }
        Err(e) => error_result(Qwen3AsrResultCode::InferenceError, &e.to_string()),
    }
}

/// Transcribe from file path
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_transcribe_file(
    handle: Qwen3AsrHandle,
    file_path: *const libc::c_char,
    options: *const Qwen3AsrTranscribeOptions,
) -> Qwen3AsrResult {
    if handle.is_null() {
        return error_result(Qwen3AsrResultCode::InvalidHandle, "Handle is null");
    }

    let path = match parse_c_string(file_path) {
        Some(s) => s,
        None => return error_result(Qwen3AsrResultCode::InvalidParameter, "file_path is null"),
    };

    let opts = if options.is_null() {
        Qwen3AsrTranscribeOptions {
            context: ptr::null(),
            language: ptr::null(),
            return_timestamps: false,
            max_new_tokens: 0,
            max_batch_size: 0,
            chunk_max_sec: 0.0,
            bucket_by_length: false,
        }
    } else {
        (*options).clone()
    };

    let context = parse_c_string(opts.context).unwrap_or_default();
    let language = parse_c_string(opts.language);

    let transcribe_opts = TranscribeOptions {
        context: Batch::one(context),
        language: Batch::one(language),
        return_timestamps: opts.return_timestamps,
        max_new_tokens: if opts.max_new_tokens > 0 { opts.max_new_tokens as usize } else { 0 },
        max_batch_size: if opts.max_batch_size > 0 { opts.max_batch_size as usize } else { 32 },
        chunk_max_sec: if opts.chunk_max_sec > 0.0 { Some(opts.chunk_max_sec) } else { None },
        bucket_by_length: opts.bucket_by_length,
    };

    let model = &*handle;
    let audio_input = AudioInput::Path(std::path::Path::new(&path));

    match model.transcribe(vec![audio_input], transcribe_opts) {
        Ok(results) => {
            if let Some(result) = results.first() {
                success_result(TranscriptionJson {
                    text: result.text.clone(),
                    language: result.language.clone(),
                    timestamps: result.timestamps.clone(),
                })
            } else {
                error_result(Qwen3AsrResultCode::InferenceError, "No transcription result")
            }
        }
        Err(e) => error_result(Qwen3AsrResultCode::InferenceError, &e.to_string()),
    }
}

// ============================================================================
// FFI Functions - Memory Management
// ============================================================================

/// Free a result
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_free_result(result: *mut Qwen3AsrResult) {
    if result.is_null() {
        return;
    }

    let r = &mut *result;
    if !r.json.is_null() {
        drop(CString::from_raw(r.json));
        r.json = ptr::null_mut();
    }
    if !r.error.is_null() {
        drop(CString::from_raw(r.error));
        r.error = ptr::null_mut();
    }
}

/// Free a string
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_free_string(ptr: *mut libc::c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Get library version
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_version() -> *mut libc::c_char {
    to_c_string(env!("CARGO_PKG_VERSION").to_string())
}

// ============================================================================
// Clone implementation for options
// ============================================================================

impl Clone for Qwen3AsrTranscribeOptions {
    fn clone(&self) -> Self {
        Self {
            context: self.context,
            language: self.language,
            return_timestamps: self.return_timestamps,
            max_new_tokens: self.max_new_tokens,
            max_batch_size: self.max_batch_size,
            chunk_max_sec: self.chunk_max_sec,
            bucket_by_length: self.bucket_by_length,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        unsafe {
            let version = qwen3_asr_version();
            let version_str = CStr::from_ptr(version).to_str().unwrap();
            assert!(!version_str.is_empty());
            qwen3_asr_free_string(version);
        }
    }
}
