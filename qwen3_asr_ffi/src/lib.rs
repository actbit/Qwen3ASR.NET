//! Qwen3-ASR FFI - Thin C-compatible wrapper for qwen3-asr-rs
//!
//! This crate provides minimal C-compatible bindings for the Qwen3-ASR library.
//! Supports native streaming transcription via stream API.

use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

use candle_core::Device;
use qwen3_asr::{AudioInput, LoadOptions, Qwen3Asr, TranscribeOptions, Batch, StreamOptions, AsrStream};
use serde::{Deserialize, Serialize};

// ============================================================================
// FFI Type Definitions
// ============================================================================

/// Opaque handle to an ASR instance
pub type Qwen3AsrHandle = *mut Qwen3Asr;

/// Opaque handle to a streaming transcription session
/// Uses raw pointer with manual lifetime management (caller must ensure model outlives stream)
/// Wrapped in Option to allow taking ownership during finish()
pub type Qwen3AsrStreamHandle = *mut Option<AsrStream<'static>>;

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
    /// Optional forced aligner model path for timestamp prediction
    pub forced_aligner_path: *const libc::c_char,
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

/// Stream options for native streaming transcription
#[repr(C)]
pub struct Qwen3AsrStreamOptions {
    pub language: *const libc::c_char,
    pub context: *const libc::c_char,
    pub chunk_size_sec: libc::c_float,
    pub unfixed_chunk_num: libc::c_int,
    pub unfixed_token_num: libc::c_int,
    pub audio_window_sec: libc::c_float,
    pub text_window_tokens: libc::c_int,
    pub max_new_tokens: libc::c_int,
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
        Qwen3AsrDevice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                match Device::new_cuda(0) {
                    Ok(device) => {
                        log::info!("CUDA device initialized successfully");
                        Ok(device)
                    }
                    Err(e) => {
                        log::warn!("CUDA initialization failed ({}), falling back to CPU", e);
                        Ok(Device::Cpu)
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                log::warn!("CUDA support not compiled in, using CPU");
                Ok(Device::Cpu)
            }
        }
        Qwen3AsrDevice::Metal => {
            #[cfg(feature = "metal")]
            {
                match Device::new_metal(0) {
                    Ok(device) => {
                        log::info!("Metal device initialized successfully");
                        Ok(device)
                    }
                    Err(e) => {
                        log::warn!("Metal initialization failed ({}), falling back to CPU", e);
                        Ok(Device::Cpu)
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                log::warn!("Metal support not compiled in, using CPU");
                Ok(Device::Cpu)
            }
        }
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
    // Call the extended version with null forced_aligner_path
    qwen3_asr_load_ex(model_path, device, ptr::null(), error_out)
}

/// Load model with forced aligner support
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_load_ex(
    model_path: *const libc::c_char,
    device: Qwen3AsrDevice,
    forced_aligner_path: *const libc::c_char,
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

    // Build load options
    // Note: forced_aligner is no longer supported in the current qwen3-asr-rs API
    // The forced_aligner_path parameter is kept for API compatibility but is ignored
    let _forced_aligner = parse_c_string(forced_aligner_path);
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

// ============================================================================
// FFI Functions - Native Streaming Transcription
// ============================================================================

/// Create a streaming transcription session
///
/// Returns a stream handle that must be destroyed with qwen3_asr_stream_destroy.
/// WARNING: The caller must ensure the model handle outlives the stream handle.
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_stream_create(
    handle: Qwen3AsrHandle,
    options: *const Qwen3AsrStreamOptions,
    error_out: *mut *mut libc::c_char,
) -> Qwen3AsrStreamHandle {
    if handle.is_null() {
        if !error_out.is_null() {
            *error_out = to_c_string("Handle is null".to_string());
        }
        return ptr::null_mut();
    }

    // Build stream options
    let opts = if options.is_null() {
        Qwen3AsrStreamOptions {
            language: ptr::null(),
            context: ptr::null(),
            chunk_size_sec: 0.0,
            unfixed_chunk_num: 0,
            unfixed_token_num: 0,
            audio_window_sec: 0.0,
            text_window_tokens: 0,
            max_new_tokens: 0,
        }
    } else {
        (*options).clone()
    };

    let language = parse_c_string(opts.language);
    let context = parse_c_string(opts.context).unwrap_or_default();

    let stream_opts = StreamOptions {
        language,
        context,
        chunk_size_sec: if opts.chunk_size_sec > 0.0 { opts.chunk_size_sec } else { 2.0 },
        unfixed_chunk_num: if opts.unfixed_chunk_num > 0 { opts.unfixed_chunk_num as usize } else { 2 },
        unfixed_token_num: if opts.unfixed_token_num > 0 { opts.unfixed_token_num as usize } else { 5 },
        audio_window_sec: if opts.audio_window_sec > 0.0 { Some(opts.audio_window_sec) } else { None },
        text_window_tokens: if opts.text_window_tokens > 0 { Some(opts.text_window_tokens as usize) } else { None },
        max_new_tokens: if opts.max_new_tokens > 0 { opts.max_new_tokens as usize } else { 256 },
    };

    let model = &*handle;
    match model.start_stream(stream_opts) {
        Ok(stream) => {
            // SAFETY: We transmute the lifetime to 'static. This is safe as long as:
            // 1. The model handle outlives the stream handle
            // 2. The stream handle is properly destroyed before the model
            // The caller is responsible for ensuring these invariants.
            let stream: AsrStream<'static> = std::mem::transmute(stream);
            Box::into_raw(Box::new(Some(stream))) as Qwen3AsrStreamHandle
        }
        Err(e) => {
            if !error_out.is_null() {
                *error_out = to_c_string(format!("Failed to create stream: {}", e));
            }
            ptr::null_mut()
        }
    }
}

/// Push audio chunk to the streaming transcription session
///
/// Returns partial transcription result as JSON
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_stream_push(
    stream_handle: Qwen3AsrStreamHandle,
    samples: *const libc::c_float,
    sample_count: libc::size_t,
    sample_rate: libc::c_uint,
) -> Qwen3AsrResult {
    if stream_handle.is_null() {
        return error_result(Qwen3AsrResultCode::InvalidHandle, "Stream handle is null");
    }

    if samples.is_null() && sample_count > 0 {
        return error_result(Qwen3AsrResultCode::InvalidParameter, "samples is null but count > 0");
    }

    let stream_opt = &mut *stream_handle;
    let stream = match stream_opt {
        Some(s) => s,
        None => return error_result(Qwen3AsrResultCode::InvalidHandle, "Stream already finished"),
    };

    let audio_slice = if sample_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(samples, sample_count)
    };

    let audio_input = AudioInput::Waveform {
        samples: audio_slice,
        sample_rate,
    };

    match stream.push_audio_chunk(&audio_input) {
        Ok(Some(partial)) => {
            success_result(TranscriptionJson {
                text: partial.text,
                language: partial.language,
                timestamps: partial.timestamps,
            })
        }
        Ok(None) => {
            // No partial result available yet
            success_result(TranscriptionJson {
                text: String::new(),
                language: String::new(),
                timestamps: None,
            })
        }
        Err(e) => error_result(Qwen3AsrResultCode::InferenceError, &e.to_string()),
    }
}

/// Finish the streaming transcription session and get final result
///
/// After calling this, the stream handle is still valid and must be destroyed
/// with qwen3_asr_stream_destroy, but the stream is consumed and cannot be used again.
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_stream_finish(
    stream_handle: Qwen3AsrStreamHandle,
) -> Qwen3AsrResult {
    if stream_handle.is_null() {
        return error_result(Qwen3AsrResultCode::InvalidHandle, "Stream handle is null");
    }

    let stream_opt = &mut *stream_handle;
    let stream = match stream_opt.take() {
        Some(s) => s,
        None => return error_result(Qwen3AsrResultCode::InvalidHandle, "Stream already finished"),
    };

    match stream.finish() {
        Ok(final_result) => {
            success_result(TranscriptionJson {
                text: final_result.text,
                language: final_result.language,
                timestamps: final_result.timestamps,
            })
        }
        Err(e) => error_result(Qwen3AsrResultCode::InferenceError, &e.to_string()),
    }
}

/// Destroy a streaming transcription session
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_stream_destroy(stream_handle: Qwen3AsrStreamHandle) {
    if !stream_handle.is_null() {
        drop(Box::from_raw(stream_handle));
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

impl Clone for Qwen3AsrStreamOptions {
    fn clone(&self) -> Self {
        Self {
            language: self.language,
            context: self.context,
            chunk_size_sec: self.chunk_size_sec,
            unfixed_chunk_num: self.unfixed_chunk_num,
            unfixed_token_num: self.unfixed_token_num,
            audio_window_sec: self.audio_window_sec,
            text_window_tokens: self.text_window_tokens,
            max_new_tokens: self.max_new_tokens,
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
