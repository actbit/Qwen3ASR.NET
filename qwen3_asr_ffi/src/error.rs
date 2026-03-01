//! Error types for Qwen3-ASR FFI
//!
//! This module provides a comprehensive error handling system with:
//! - Detailed error categories and codes
//! - Error context and chaining
//! - User-friendly messages
//! - FFI-compatible error codes

use std::fmt;
use std::io;

/// Result type alias for Qwen3-ASR operations
pub type Result<T> = std::result::Result<T, Qwen3AsrError>;

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Model-related errors (loading, not found, etc.)
    Model,
    /// Inference-related errors (prediction, processing)
    Inference,
    /// Input validation errors
    Validation,
    /// I/O errors (file, network)
    Io,
    /// Streaming session errors
    Streaming,
    /// Device errors (CUDA, Metal, CPU)
    Device,
    /// Internal errors (bugs, unexpected states)
    Internal,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Model => write!(f, "Model"),
            ErrorCategory::Inference => write!(f, "Inference"),
            ErrorCategory::Validation => write!(f, "Validation"),
            ErrorCategory::Io => write!(f, "I/O"),
            ErrorCategory::Streaming => write!(f, "Streaming"),
            ErrorCategory::Device => write!(f, "Device"),
            ErrorCategory::Internal => write!(f, "Internal"),
        }
    }
}

/// Main error type for Qwen3-ASR FFI operations
#[derive(Debug)]
pub struct Qwen3AsrError {
    /// The kind of error
    kind: ErrorKind,
    /// Detailed error message
    message: String,
    /// Error context (additional information)
    context: Option<String>,
    /// Source error message (for error chaining - stored as string for Clone)
    source_msg: Option<String>,
    /// Backtrace (for debugging - stored as string for Clone)
    backtrace_msg: Option<String>,
}

impl Clone for Qwen3AsrError {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            message: self.message.clone(),
            context: self.context.clone(),
            source_msg: self.source_msg.clone(),
            backtrace_msg: self.backtrace_msg.clone(),
        }
    }
}

/// Specific error kinds with additional data
#[derive(Debug, Clone, thiserror::Error)]
pub enum ErrorKind {
    // Model errors
    #[error("Model not found")]
    ModelNotFound {
        model_id: String,
    },
    #[error("Failed to load model")]
    ModelLoadFailed {
        model_id: String,
        stage: LoadStage,
    },
    #[error("Model already loaded")]
    ModelAlreadyLoaded,
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Invalid model format")]
    InvalidModelFormat {
        expected: String,
        actual: String,
    },
    #[error("Model download failed")]
    ModelDownloadFailed {
        model_id: String,
        reason: String,
    },

    // Inference errors
    #[error("Inference failed")]
    InferenceFailed {
        reason: String,
    },
    #[error("Audio processing failed")]
    AudioProcessingFailed {
        stage: AudioStage,
    },
    #[error("Tokenization failed")]
    TokenizationFailed,
    #[error("Decoding failed")]
    DecodingFailed,
    #[error("Out of memory during inference")]
    InferenceOutOfMemory {
        requested_mb: usize,
        available_mb: Option<usize>,
    },

    // Input validation errors
    #[error("Invalid input")]
    InvalidInput {
        field: String,
        reason: String,
    },
    #[error("Invalid audio format")]
    InvalidAudioFormat {
        format: String,
        supported_formats: Vec<String>,
    },
    #[error("Invalid sample rate")]
    InvalidSampleRate {
        actual: u32,
        expected: u32,
    },
    #[error("Invalid audio channels")]
    InvalidChannels {
        actual: u16,
        expected: u16,
    },
    #[error("Empty audio input")]
    EmptyAudioInput,
    #[error("Audio too short")]
    AudioTooShort {
        duration_ms: u32,
        minimum_ms: u32,
    },
    #[error("Audio too long")]
    AudioTooLong {
        duration_ms: u32,
        maximum_ms: u32,
    },
    #[error("Invalid language code")]
    InvalidLanguageCode {
        code: String,
        supported_languages: Vec<String>,
    },

    // I/O errors
    #[error("File not found")]
    FileNotFound {
        path: String,
    },
    #[error("File read error")]
    FileReadError {
        path: String,
    },
    #[error("File write error")]
    FileWriteError {
        path: String,
    },
    #[error("Invalid file format")]
    InvalidFileFormat {
        path: String,
        expected: String,
    },
    #[error("Network error")]
    NetworkError {
        url: String,
    },

    // Streaming errors
    #[error("Stream not started")]
    StreamNotStarted,
    #[error("Stream already finished")]
    StreamAlreadyFinished,
    #[error("Stream push failed")]
    StreamPushFailed {
        reason: String,
    },
    #[error("Stream buffer overflow")]
    StreamBufferOverflow {
        buffer_size: usize,
        input_size: usize,
    },
    #[error("Stream timeout")]
    StreamTimeout {
        timeout_ms: u64,
    },

    // Device errors
    #[error("Device not available")]
    DeviceNotAvailable {
        device: String,
    },
    #[error("Device initialization failed")]
    DeviceInitFailed {
        device: String,
        reason: String,
    },
    #[error("CUDA error")]
    CudaError {
        code: i32,
        message: String,
    },
    #[error("Metal error")]
    MetalError {
        message: String,
    },

    // Internal errors
    #[error("Internal error")]
    InternalError {
        message: String,
    },
    #[error("Null pointer")]
    NullPointer {
        context: String,
    },
    #[error("UTF-8 conversion error")]
    Utf8Error,
    #[error("JSON error")]
    JsonError {
        message: String,
    },
    #[error("Unknown error")]
    Unknown,
}

/// Stage of model loading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStage {
    /// Downloading model files
    Downloading,
    /// Loading model configuration
    ConfigLoading,
    /// Loading model weights
    WeightLoading,
    /// Initializing tokenizer
    TokenizerInit,
    /// Initializing inference engine
    EngineInit,
    /// Validating model
    Validation,
}

impl fmt::Display for LoadStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadStage::Downloading => write!(f, "downloading"),
            LoadStage::ConfigLoading => write!(f, "loading config"),
            LoadStage::WeightLoading => write!(f, "loading weights"),
            LoadStage::TokenizerInit => write!(f, "initializing tokenizer"),
            LoadStage::EngineInit => write!(f, "initializing engine"),
            LoadStage::Validation => write!(f, "validating model"),
        }
    }
}

/// Stage of audio processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioStage {
    /// Reading audio file
    Reading,
    /// Decoding audio
    Decoding,
    /// Resampling audio
    Resampling,
    /// Converting channels
    ChannelConversion,
    /// Normalizing audio
    Normalization,
    /// Feature extraction
    FeatureExtraction,
}

impl fmt::Display for AudioStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioStage::Reading => write!(f, "reading"),
            AudioStage::Decoding => write!(f, "decoding"),
            AudioStage::Resampling => write!(f, "resampling"),
            AudioStage::ChannelConversion => write!(f, "converting channels"),
            AudioStage::Normalization => write!(f, "normalizing"),
            AudioStage::FeatureExtraction => write!(f, "extracting features"),
        }
    }
}

impl Qwen3AsrError {
    /// Create a new error with the given kind and message
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        let backtrace_msg = std::backtrace::Backtrace::capture().to_string();
        Self {
            kind,
            message: message.into(),
            context: None,
            source_msg: None,
            backtrace_msg: if backtrace_msg.is_empty() { None } else { Some(backtrace_msg) },
        }
    }

    /// Create an error from an error kind
    pub fn from_kind(kind: ErrorKind) -> Self {
        let message = kind.to_string();
        let backtrace_msg = std::backtrace::Backtrace::capture().to_string();
        Self {
            kind,
            message,
            context: None,
            source_msg: None,
            backtrace_msg: if backtrace_msg.is_empty() { None } else { Some(backtrace_msg) },
        }
    }

    /// Add context to the error
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Add a source error for chaining
    pub fn with_source(mut self, source: impl std::error::Error + 'static) -> Self {
        self.source_msg = Some(source.to_string());
        self
    }

    /// Get the error kind
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Get the error message
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Get the error context
    pub fn context(&self) -> Option<&str> {
        self.context.as_deref()
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match &self.kind {
            ErrorKind::ModelNotFound { .. }
            | ErrorKind::ModelLoadFailed { .. }
            | ErrorKind::ModelAlreadyLoaded
            | ErrorKind::ModelNotLoaded
            | ErrorKind::InvalidModelFormat { .. }
            | ErrorKind::ModelDownloadFailed { .. } => ErrorCategory::Model,

            ErrorKind::InferenceFailed { .. }
            | ErrorKind::AudioProcessingFailed { .. }
            | ErrorKind::TokenizationFailed
            | ErrorKind::DecodingFailed
            | ErrorKind::InferenceOutOfMemory { .. } => ErrorCategory::Inference,

            ErrorKind::InvalidInput { .. }
            | ErrorKind::InvalidAudioFormat { .. }
            | ErrorKind::InvalidSampleRate { .. }
            | ErrorKind::InvalidChannels { .. }
            | ErrorKind::EmptyAudioInput
            | ErrorKind::AudioTooShort { .. }
            | ErrorKind::AudioTooLong { .. }
            | ErrorKind::InvalidLanguageCode { .. } => ErrorCategory::Validation,

            ErrorKind::FileNotFound { .. }
            | ErrorKind::FileReadError { .. }
            | ErrorKind::FileWriteError { .. }
            | ErrorKind::InvalidFileFormat { .. }
            | ErrorKind::NetworkError { .. } => ErrorCategory::Io,

            ErrorKind::StreamNotStarted
            | ErrorKind::StreamAlreadyFinished
            | ErrorKind::StreamPushFailed { .. }
            | ErrorKind::StreamBufferOverflow { .. }
            | ErrorKind::StreamTimeout { .. } => ErrorCategory::Streaming,

            ErrorKind::DeviceNotAvailable { .. }
            | ErrorKind::DeviceInitFailed { .. }
            | ErrorKind::CudaError { .. }
            | ErrorKind::MetalError { .. } => ErrorCategory::Device,

            ErrorKind::InternalError { .. }
            | ErrorKind::NullPointer { .. }
            | ErrorKind::Utf8Error
            | ErrorKind::JsonError { .. }
            | ErrorKind::Unknown => ErrorCategory::Internal,
        }
    }

    /// Get a user-friendly error message
    pub fn user_message(&self) -> String {
        let category = self.category();
        let base_message = match &self.kind {
            ErrorKind::ModelNotFound { model_id } => {
                format!("モデル '{}' が見つかりませんでした。モデルIDが正しいか確認してください。", model_id)
            }
            ErrorKind::ModelLoadFailed { model_id, stage } => {
                format!(
                    "モデル '{}' の読み込みに失敗しました（段階: {}）。ディスク容量とネットワーク接続を確認してください。",
                    model_id, stage
                )
            }
            ErrorKind::ModelNotLoaded => {
                "モデルが読み込まれていません。先にモデルを読み込んでください。".to_string()
            }
            ErrorKind::InvalidAudioFormat { format, supported_formats } => {
                format!(
                    "音声フォーマット '{}' はサポートされていません。対応フォーマット: {}",
                    format,
                    supported_formats.join(", ")
                )
            }
            ErrorKind::InvalidSampleRate { actual, expected } => {
                format!(
                    "サンプリングレートが不正です（実際: {}Hz、期待: {}Hz）",
                    actual, expected
                )
            }
            ErrorKind::EmptyAudioInput => {
                "音声入力が空です。有効な音声データを提供してください。".to_string()
            }
            ErrorKind::FileNotFound { path } => {
                format!("ファイル '{}' が見つかりません。パスを確認してください。", path)
            }
            ErrorKind::StreamAlreadyFinished => {
                "ストリームは既に終了しています。".to_string()
            }
            ErrorKind::DeviceNotAvailable { device } => {
                format!("デバイス '{}' が利用できません。", device)
            }
            _ => self.message.clone(),
        };

        if let Some(ctx) = &self.context {
            format!("[{}] {} （詳細: {}）", category, base_message, ctx)
        } else {
            format!("[{}] {}", category, base_message)
        }
    }

    /// Convert to FFI result code
    pub fn as_code(&self) -> crate::Qwen3AsrResultCode {
        match &self.kind {
            // Model errors -> ModelNotLoaded
            ErrorKind::ModelNotFound { .. }
            | ErrorKind::ModelLoadFailed { .. }
            | ErrorKind::ModelAlreadyLoaded
            | ErrorKind::ModelNotLoaded
            | ErrorKind::InvalidModelFormat { .. }
            | ErrorKind::ModelDownloadFailed { .. } => crate::Qwen3AsrResultCode::ModelNotLoaded,

            // Validation errors -> InvalidParameter
            ErrorKind::InvalidInput { .. }
            | ErrorKind::InvalidAudioFormat { .. }
            | ErrorKind::InvalidSampleRate { .. }
            | ErrorKind::InvalidChannels { .. }
            | ErrorKind::EmptyAudioInput
            | ErrorKind::AudioTooShort { .. }
            | ErrorKind::AudioTooLong { .. }
            | ErrorKind::InvalidLanguageCode { .. } => crate::Qwen3AsrResultCode::InvalidParameter,

            // Streaming errors -> StreamError
            ErrorKind::StreamNotStarted
            | ErrorKind::StreamAlreadyFinished
            | ErrorKind::StreamPushFailed { .. }
            | ErrorKind::StreamBufferOverflow { .. }
            | ErrorKind::StreamTimeout { .. } => crate::Qwen3AsrResultCode::StreamError,

            // Memory errors -> MemoryError
            ErrorKind::InferenceOutOfMemory { .. } => crate::Qwen3AsrResultCode::MemoryError,

            // Everything else -> InferenceError or UnknownError
            ErrorKind::InferenceFailed { .. }
            | ErrorKind::AudioProcessingFailed { .. }
            | ErrorKind::TokenizationFailed
            | ErrorKind::DecodingFailed
            | ErrorKind::FileNotFound { .. }
            | ErrorKind::FileReadError { .. }
            | ErrorKind::FileWriteError { .. }
            | ErrorKind::InvalidFileFormat { .. }
            | ErrorKind::NetworkError { .. }
            | ErrorKind::DeviceNotAvailable { .. }
            | ErrorKind::DeviceInitFailed { .. }
            | ErrorKind::CudaError { .. }
            | ErrorKind::MetalError { .. } => crate::Qwen3AsrResultCode::InferenceError,

            // Internal errors
            ErrorKind::InternalError { .. }
            | ErrorKind::NullPointer { .. }
            | ErrorKind::Utf8Error
            | ErrorKind::JsonError { .. }
            | ErrorKind::Unknown => crate::Qwen3AsrResultCode::UnknownError,
        }
    }

    // =========================================================================
    // Convenience constructors
    // =========================================================================

    /// Create a model not found error
    pub fn model_not_found(model_id: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::ModelNotFound {
            model_id: model_id.into(),
        })
    }

    /// Create a model load failed error
    pub fn model_load_failed(model_id: impl Into<String>, stage: LoadStage) -> Self {
        Self::from_kind(ErrorKind::ModelLoadFailed {
            model_id: model_id.into(),
            stage,
        })
    }

    /// Create a model not loaded error
    pub fn model_not_loaded() -> Self {
        Self::from_kind(ErrorKind::ModelNotLoaded)
    }

    /// Create an inference failed error
    pub fn inference_failed(reason: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InferenceFailed {
            reason: reason.into(),
        })
    }

    /// Create an invalid input error
    pub fn invalid_input(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidInput {
            field: field.into(),
            reason: reason.into(),
        })
    }

    /// Create a file not found error
    pub fn file_not_found(path: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::FileNotFound {
            path: path.into(),
        })
    }

    /// Create an audio processing failed error
    pub fn audio_processing_failed(stage: AudioStage) -> Self {
        Self::from_kind(ErrorKind::AudioProcessingFailed { stage })
    }

    /// Create a stream already finished error
    pub fn stream_already_finished() -> Self {
        Self::from_kind(ErrorKind::StreamAlreadyFinished)
    }

    /// Create a stream push failed error
    pub fn stream_push_failed(reason: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::StreamPushFailed {
            reason: reason.into(),
        })
    }

    /// Create a device not available error
    pub fn device_not_available(device: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::DeviceNotAvailable {
            device: device.into(),
        })
    }

    /// Create a null pointer error
    pub fn null_pointer(context: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::NullPointer {
            context: context.into(),
        })
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InternalError {
            message: message.into(),
        })
    }
}

impl fmt::Display for Qwen3AsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.category(), self.message)?;

        if let Some(ctx) = &self.context {
            write!(f, " (context: {})", ctx)?;
        }

        if let Some(source) = &self.source_msg {
            write!(f, " (caused by: {})", source)?;
        }

        Ok(())
    }
}

impl std::error::Error for Qwen3AsrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // We store source as a string, so we can't return a reference
        None
    }
}

// =========================================================================
// From implementations for common error types
// =========================================================================

impl From<io::Error> for Qwen3AsrError {
    fn from(err: io::Error) -> Self {
        Self::new(ErrorKind::FileReadError { path: String::new() }, err.to_string())
            .with_source(err)
    }
}

impl From<serde_json::Error> for Qwen3AsrError {
    fn from(err: serde_json::Error) -> Self {
        Self::new(
            ErrorKind::JsonError {
                message: err.to_string(),
            },
            "JSON serialization/deserialization failed",
        )
        .with_source(err)
    }
}

impl From<std::string::FromUtf8Error> for Qwen3AsrError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        Self::from_kind(ErrorKind::Utf8Error).with_source(err)
    }
}

impl From<std::str::Utf8Error> for Qwen3AsrError {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::from_kind(ErrorKind::Utf8Error).with_source(err)
    }
}

impl From<String> for Qwen3AsrError {
    fn from(msg: String) -> Self {
        Self::new(ErrorKind::Unknown, msg)
    }
}

impl From<&str> for Qwen3AsrError {
    fn from(msg: &str) -> Self {
        Self::new(ErrorKind::Unknown, msg)
    }
}

// =========================================================================
// anyhow integration
// =========================================================================

impl From<anyhow::Error> for Qwen3AsrError {
    fn from(err: anyhow::Error) -> Self {
        // Try to downcast to Qwen3AsrError first
        if let Some(asr_err) = err.downcast_ref::<Qwen3AsrError>() {
            return asr_err.clone();
        }

        // Otherwise, wrap as internal error
        Self::new(
            ErrorKind::InternalError {
                message: err.to_string(),
            },
            "An internal error occurred",
        )
    }
}

// Note: From<Qwen3AsrError> for anyhow::Error is automatically provided
// because Qwen3AsrError implements std::error::Error + Send + Sync + 'static

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categories() {
        let err = Qwen3AsrError::model_not_found("test-model");
        assert_eq!(err.category(), ErrorCategory::Model);

        let err = Qwen3AsrError::invalid_input("field", "reason");
        assert_eq!(err.category(), ErrorCategory::Validation);

        let err = Qwen3AsrError::stream_already_finished();
        assert_eq!(err.category(), ErrorCategory::Streaming);
    }

    #[test]
    fn test_error_context() {
        let err = Qwen3AsrError::model_not_found("test-model")
            .with_context("Additional context");

        assert!(err.context().is_some());
        assert!(err.to_string().contains("Additional context"));
    }

    #[test]
    fn test_user_message() {
        let err = Qwen3AsrError::model_not_found("Qwen/Qwen3-ASR-0.6B");
        let msg = err.user_message();

        assert!(msg.contains("Qwen/Qwen3-ASR-0.6B"));
        assert!(msg.contains("見つかりません"));
    }

    #[test]
    fn test_ffi_code_conversion() {
        let err = Qwen3AsrError::model_not_found("test");
        assert_eq!(err.as_code(), crate::Qwen3AsrResultCode::ModelNotLoaded);

        let err = Qwen3AsrError::invalid_input("field", "reason");
        assert_eq!(err.as_code(), crate::Qwen3AsrResultCode::InvalidParameter);

        let err = Qwen3AsrError::stream_already_finished();
        assert_eq!(err.as_code(), crate::Qwen3AsrResultCode::StreamError);
    }

    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let asr_err: Qwen3AsrError = io_err.into();

        assert_eq!(asr_err.category(), ErrorCategory::Io);
    }

    #[test]
    fn test_error_chaining() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "permission denied");
        let asr_err = Qwen3AsrError::file_not_found("/path/to/file")
            .with_source(io_err);

        // Source is stored as string, check the display output
        assert!(asr_err.to_string().contains("caused by"));
    }
}
