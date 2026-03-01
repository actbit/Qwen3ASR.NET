//! Qwen3-ASR FFI (Foreign Function Interface)
//!
//! This crate provides C-compatible bindings for the Qwen3-ASR speech recognition library.
//! It enables integration with .NET via P/Invoke and other languages via C FFI.

use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

use serde::{Deserialize, Serialize};

mod error;
mod handle;
mod transcriber;
mod streaming;

pub use error::Qwen3AsrError;
pub use handle::AsrHandle;
pub use transcriber::Transcriber;
pub use streaming::StreamingTranscriber;

// Re-export FFI types
mod ffi_types;
pub use ffi_types::*;

// ============================================================================
// FFI Type Definitions
// ============================================================================

/// Opaque handle to an ASR instance
pub type Qwen3AsrHandle = *mut AsrHandle;

/// Opaque handle to a streaming transcription session
pub type Qwen3AsrStreamHandle = *mut StreamingTranscriber;

/// Device type for model inference
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3AsrDevice {
    /// CPU inference
    Cpu = 0,
    /// CUDA GPU inference
    Cuda = 1,
    /// Metal GPU inference (macOS)
    Metal = 2,
}

/// Result codes for FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3AsrResultCode {
    /// Operation successful
    Success = 0,
    /// Invalid handle
    InvalidHandle = 1,
    /// Invalid parameter
    InvalidParameter = 2,
    /// Model not loaded
    ModelNotLoaded = 3,
    /// Inference error
    InferenceError = 4,
    /// Memory allocation error
    MemoryError = 5,
    /// Streaming session error
    StreamError = 6,
    /// Unknown error
    UnknownError = 99,
}

/// Options for loading a model
#[repr(C)]
pub struct Qwen3AsrLoadOptions {
    /// Device to use for inference
    pub device: Qwen3AsrDevice,
    /// Path to the model directory or HuggingFace model ID
    pub model_path: *const libc::c_char,
    /// Optional: specific revision/branch for HuggingFace models
    pub revision: *const libc::c_char,
    /// Number of threads for CPU inference (0 = auto)
    pub num_threads: libc::c_int,
}

/// Options for streaming transcription
#[repr(C)]
pub struct Qwen3AsrStreamOptions {
    /// Language code (e.g., "Japanese", "English", "Chinese")
    pub language: *const libc::c_char,
    /// Chunk size in seconds for streaming
    pub chunk_size_sec: libc::c_float,
    /// Enable timestamp prediction
    pub enable_timestamps: bool,
    /// Enable partial results during streaming
    pub enable_partial_results: bool,
}

/// Transcription result returned by FFI
#[repr(C)]
pub struct Qwen3AsrTranscriptionResult {
    /// Transcribed text (UTF-8, null-terminated)
    pub text: *mut libc::c_char,
    /// JSON-formatted detailed result (optional, may be null)
    pub json_result: *mut libc::c_char,
    /// Result code
    pub code: Qwen3AsrResultCode,
    /// Error message (if code != Success, may be null)
    pub error_message: *mut libc::c_char,
}

/// Detailed transcription information (for JSON serialization)
#[derive(Debug, Serialize, Deserialize)]
pub struct TranscriptionDetail {
    /// Transcribed text
    pub text: String,
    /// Language detected/used
    pub language: Option<String>,
    /// Confidence score (if available)
    pub confidence: Option<f32>,
    /// Timestamps for words/segments
    pub timestamps: Option<Vec<Timestamp>>,
}

/// Timestamp information
#[derive(Debug, Serialize, Deserialize)]
pub struct Timestamp {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Word or segment text
    pub text: String,
}

// ============================================================================
// FFI Functions
// ============================================================================

/// Create a new ASR instance
///
/// # Safety
/// - `model_path` must be a valid null-terminated UTF-8 string
/// - `options` must be a valid pointer to Qwen3AsrLoadOptions
/// - The returned handle must be freed using qwen3_asr_destroy
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_create(
    options: *const Qwen3AsrLoadOptions,
    error_msg: *mut *mut libc::c_char,
) -> Qwen3AsrHandle {
    if options.is_null() {
        if !error_msg.is_null() {
            *error_msg = CString::new("Options pointer is null")
                .unwrap()
                .into_raw();
        }
        return ptr::null_mut();
    }

    let opts = &*options;

    // Convert C strings to Rust strings
    let model_path = if opts.model_path.is_null() {
        if !error_msg.is_null() {
            *error_msg = CString::new("model_path is null")
                .unwrap()
                .into_raw();
        }
        return ptr::null_mut();
    } else {
        match CStr::from_ptr(opts.model_path).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                if !error_msg.is_null() {
                    *error_msg = CString::new(format!("Invalid model_path UTF-8: {}", e))
                        .unwrap()
                        .into_raw();
                }
                return ptr::null_mut();
            }
        }
    };

    let revision = if opts.revision.is_null() {
        None
    } else {
        match CStr::from_ptr(opts.revision).to_str() {
            Ok(s) => Some(s.to_string()),
            Err(_) => None,
        }
    };

    // Create the handle
    match AsrHandle::new(model_path, opts.device, revision, opts.num_threads as usize) {
        Ok(handle) => Box::into_raw(Box::new(handle)),
        Err(e) => {
            if !error_msg.is_null() {
                *error_msg = CString::new(format!("Failed to create ASR instance: {}", e))
                    .unwrap()
                    .into_raw();
            }
            ptr::null_mut()
        }
    }
}

/// Transcribe audio data (offline/batch mode)
///
/// # Safety
/// - `handle` must be a valid handle from qwen3_asr_create
/// - `audio_data` must be a valid pointer to f32 samples
/// - `len` must be the number of f32 samples in audio_data
/// - `language` may be null (auto-detect) or a valid null-terminated UTF-8 string
/// - The returned result must be freed using qwen3_asr_free_result
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_transcribe(
    handle: Qwen3AsrHandle,
    audio_data: *const libc::c_float,
    len: libc::size_t,
    language: *const libc::c_char,
) -> Qwen3AsrTranscriptionResult {
    // Validate handle
    if handle.is_null() {
        return Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InvalidHandle,
            error_message: CString::new("Handle is null")
                .unwrap()
                .into_raw(),
        };
    }

    // Validate audio data
    if audio_data.is_null() && len > 0 {
        return Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InvalidParameter,
            error_message: CString::new("audio_data is null but len > 0")
                .unwrap()
                .into_raw(),
        };
    }

    // Get audio slice
    let audio = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(audio_data, len)
    };

    // Parse language
    let lang = if language.is_null() {
        None
    } else {
        match CStr::from_ptr(language).to_str() {
            Ok(s) if !s.is_empty() => Some(s.to_string()),
            _ => None,
        }
    };

    // Perform transcription
    let asr_handle = &mut *handle;
    match asr_handle.transcribe(audio, lang.as_deref()) {
        Ok(result) => {
            let text = CString::new(result.text.clone()).unwrap().into_raw();
            let json_result = CString::new(serde_json::to_string(&result).unwrap_or_default())
                .unwrap()
                .into_raw();

            Qwen3AsrTranscriptionResult {
                text,
                json_result,
                code: Qwen3AsrResultCode::Success,
                error_message: ptr::null_mut(),
            }
        }
        Err(e) => Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InferenceError,
            error_message: CString::new(e.to_string()).unwrap().into_raw(),
        },
    }
}

/// Transcribe audio from a file
///
/// # Safety
/// - `handle` must be a valid handle from qwen3_asr_create
/// - `file_path` must be a valid null-terminated UTF-8 string
/// - `language` may be null (auto-detect) or a valid null-terminated UTF-8 string
/// - The returned result must be freed using qwen3_asr_free_result
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_transcribe_file(
    handle: Qwen3AsrHandle,
    file_path: *const libc::c_char,
    language: *const libc::c_char,
) -> Qwen3AsrTranscriptionResult {
    // Validate handle
    if handle.is_null() {
        return Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InvalidHandle,
            error_message: CString::new("Handle is null")
                .unwrap()
                .into_raw(),
        };
    }

    // Parse file path
    if file_path.is_null() {
        return Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InvalidParameter,
            error_message: CString::new("file_path is null")
                .unwrap()
                .into_raw(),
        };
    }

    let path = match CStr::from_ptr(file_path).to_str() {
        Ok(s) => s,
        Err(e) => {
            return Qwen3AsrTranscriptionResult {
                text: ptr::null_mut(),
                json_result: ptr::null_mut(),
                code: Qwen3AsrResultCode::InvalidParameter,
                error_message: CString::new(format!("Invalid file_path UTF-8: {}", e))
                    .unwrap()
                    .into_raw(),
            };
        }
    };

    // Parse language
    let lang = if language.is_null() {
        None
    } else {
        match CStr::from_ptr(language).to_str() {
            Ok(s) if !s.is_empty() => Some(s.to_string()),
            _ => None,
        }
    };

    // Perform transcription
    let asr_handle = &mut *handle;
    match asr_handle.transcribe_file(path, lang.as_deref()) {
        Ok(result) => {
            let text = CString::new(result.text.clone()).unwrap().into_raw();
            let json_result = CString::new(serde_json::to_string(&result).unwrap_or_default())
                .unwrap()
                .into_raw();

            Qwen3AsrTranscriptionResult {
                text,
                json_result,
                code: Qwen3AsrResultCode::Success,
                error_message: ptr::null_mut(),
            }
        }
        Err(e) => Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InferenceError,
            error_message: CString::new(e.to_string()).unwrap().into_raw(),
        },
    }
}

/// Start a streaming transcription session
///
/// # Safety
/// - `handle` must be a valid handle from qwen3_asr_create
/// - `options` must be a valid pointer to Qwen3AsrStreamOptions
/// - The returned stream handle must be freed using qwen3_asr_stream_finish
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_stream_start(
    handle: Qwen3AsrHandle,
    options: *const Qwen3AsrStreamOptions,
) -> Qwen3AsrStreamHandle {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let opts = if options.is_null() {
        Qwen3AsrStreamOptions::default()
    } else {
        (*options).clone()
    };

    // Parse language
    let language = if opts.language.is_null() {
        None
    } else {
        match CStr::from_ptr(opts.language).to_str() {
            Ok(s) if !s.is_empty() => Some(s.to_string()),
            _ => None,
        }
    };

    let asr_handle = &*handle;
    match StreamingTranscriber::new(
        asr_handle,
        language,
        opts.chunk_size_sec,
        opts.enable_timestamps,
        opts.enable_partial_results,
    ) {
        Ok(stream) => Box::into_raw(Box::new(stream)),
        Err(_) => ptr::null_mut(),
    }
}

/// Push audio chunk to streaming session
///
/// # Safety
/// - `stream_handle` must be a valid handle from qwen3_asr_stream_start
/// - `audio_data` must be a valid pointer to f32 samples
/// - `len` must be the number of f32 samples in audio_data
/// - The returned result must be freed using qwen3_asr_free_result
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_stream_push(
    stream_handle: Qwen3AsrStreamHandle,
    audio_data: *const libc::c_float,
    len: libc::size_t,
) -> Qwen3AsrTranscriptionResult {
    if stream_handle.is_null() {
        return Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InvalidHandle,
            error_message: CString::new("Stream handle is null")
                .unwrap()
                .into_raw(),
        };
    }

    if audio_data.is_null() && len > 0 {
        return Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InvalidParameter,
            error_message: CString::new("audio_data is null but len > 0")
                .unwrap()
                .into_raw(),
        };
    }

    let audio = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(audio_data, len)
    };

    let stream = &mut *stream_handle;
    match stream.push(audio) {
        Ok(result) => {
            let text = CString::new(result.text.clone()).unwrap().into_raw();
            let json_result = CString::new(serde_json::to_string(&result).unwrap_or_default())
                .unwrap()
                .into_raw();

            Qwen3AsrTranscriptionResult {
                text,
                json_result,
                code: Qwen3AsrResultCode::Success,
                error_message: ptr::null_mut(),
            }
        }
        Err(e) => Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::StreamError,
            error_message: CString::new(e.to_string()).unwrap().into_raw(),
        },
    }
}

/// Get partial result from streaming session
///
/// # Safety
/// - `stream_handle` must be a valid handle from qwen3_asr_stream_start
/// - The returned result must be freed using qwen3_asr_free_result
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_stream_get_partial(
    stream_handle: Qwen3AsrStreamHandle,
) -> Qwen3AsrTranscriptionResult {
    if stream_handle.is_null() {
        return Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InvalidHandle,
            error_message: CString::new("Stream handle is null")
                .unwrap()
                .into_raw(),
        };
    }

    let stream = &*stream_handle;
    match stream.get_partial() {
        Some(result) => {
            let text = CString::new(result.text.clone()).unwrap().into_raw();
            let json_result = CString::new(serde_json::to_string(&result).unwrap_or_default())
                .unwrap()
                .into_raw();

            Qwen3AsrTranscriptionResult {
                text,
                json_result,
                code: Qwen3AsrResultCode::Success,
                error_message: ptr::null_mut(),
            }
        }
        None => Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::Success,
            error_message: ptr::null_mut(),
        },
    }
}

/// Finish streaming session and get final result
///
/// # Safety
/// - `stream_handle` must be a valid handle from qwen3_asr_stream_start
/// - This function consumes the handle (it becomes invalid after call)
/// - The returned result must be freed using qwen3_asr_free_result
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_stream_finish(
    stream_handle: Qwen3AsrStreamHandle,
) -> Qwen3AsrTranscriptionResult {
    if stream_handle.is_null() {
        return Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::InvalidHandle,
            error_message: CString::new("Stream handle is null")
                .unwrap()
                .into_raw(),
        };
    }

    // Take ownership and drop the box
    let stream = Box::from_raw(stream_handle);
    match stream.finish() {
        Ok(result) => {
            let text = CString::new(result.text.clone()).unwrap().into_raw();
            let json_result = CString::new(serde_json::to_string(&result).unwrap_or_default())
                .unwrap()
                .into_raw();

            Qwen3AsrTranscriptionResult {
                text,
                json_result,
                code: Qwen3AsrResultCode::Success,
                error_message: ptr::null_mut(),
            }
        }
        Err(e) => Qwen3AsrTranscriptionResult {
            text: ptr::null_mut(),
            json_result: ptr::null_mut(),
            code: Qwen3AsrResultCode::StreamError,
            error_message: CString::new(e.to_string()).unwrap().into_raw(),
        },
    }
}

/// Free a transcription result
///
/// # Safety
/// - `result` must be a valid pointer to Qwen3AsrTranscriptionResult
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_free_result(result: *mut Qwen3AsrTranscriptionResult) {
    if result.is_null() {
        return;
    }

    let result = &mut *result;

    if !result.text.is_null() {
        drop(CString::from_raw(result.text));
        result.text = ptr::null_mut();
    }

    if !result.json_result.is_null() {
        drop(CString::from_raw(result.json_result));
        result.json_result = ptr::null_mut();
    }

    if !result.error_message.is_null() {
        drop(CString::from_raw(result.error_message));
        result.error_message = ptr::null_mut();
    }
}

/// Free a string returned by FFI
///
/// # Safety
/// - `ptr` must be a valid pointer returned by this library
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_free_string(ptr: *mut libc::c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Destroy an ASR instance
///
/// # Safety
/// - `handle` must be a valid handle from qwen3_asr_create
/// - The handle becomes invalid after this call
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_destroy(handle: Qwen3AsrHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Get the library version string
///
/// # Safety
/// - The returned string must be freed using qwen3_asr_free_string
#[no_mangle]
pub unsafe extern "C" fn qwen3_asr_version() -> *mut libc::c_char {
    CString::new(env!("CARGO_PKG_VERSION"))
        .unwrap()
        .into_raw()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_version() {
        unsafe {
            let version = qwen3_asr_version();
            let version_str = CStr::from_ptr(version).to_str().unwrap();
            assert!(!version_str.is_empty());
            qwen3_asr_free_string(version);
        }
    }

    #[test]
    fn test_create_with_null_options() {
        unsafe {
            let mut error_msg: *mut libc::c_char = ptr::null_mut();
            let handle = qwen3_asr_create(ptr::null(), &mut error_msg);
            assert!(handle.is_null());
            assert!(!error_msg.is_null());
            qwen3_asr_free_string(error_msg);
        }
    }
}
