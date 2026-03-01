//! Error types for Qwen3-ASR FFI

use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum Qwen3AsrError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Audio processing error: {0}")]
    AudioProcessingError(String),

    #[error("Streaming error: {0}")]
    StreamingError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl Qwen3AsrError {
    pub fn as_code(&self) -> crate::Qwen3AsrResultCode {
        match self {
            Qwen3AsrError::ModelNotFound(_) => crate::Qwen3AsrResultCode::ModelNotLoaded,
            Qwen3AsrError::ModelLoadError(_) => crate::Qwen3AsrResultCode::ModelNotLoaded,
            Qwen3AsrError::InferenceError(_) => crate::Qwen3AsrResultCode::InferenceError,
            Qwen3AsrError::InvalidInput(_) => crate::Qwen3AsrResultCode::InvalidParameter,
            Qwen3AsrError::AudioProcessingError(_) => crate::Qwen3AsrResultCode::InferenceError,
            Qwen3AsrError::StreamingError(_) => crate::Qwen3AsrResultCode::StreamError,
            Qwen3AsrError::DeviceError(_) => crate::Qwen3AsrResultCode::InferenceError,
            Qwen3AsrError::IoError(_) => crate::Qwen3AsrResultCode::InferenceError,
            Qwen3AsrError::JsonError(_) => crate::Qwen3AsrResultCode::InferenceError,
            Qwen3AsrError::Unknown(_) => crate::Qwen3AsrResultCode::UnknownError,
        }
    }
}
