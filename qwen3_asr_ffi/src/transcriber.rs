//! Transcriber implementation

use crate::error::Qwen3AsrError;
use crate::{AsrHandle, TranscriptionDetail};

/// Transcriber for batch transcription
pub struct Transcriber<'a> {
    handle: &'a AsrHandle,
}

impl<'a> Transcriber<'a> {
    /// Create a new transcriber
    pub fn new(handle: &'a AsrHandle) -> Self {
        Self { handle }
    }

    /// Transcribe audio samples
    pub fn transcribe(
        &self,
        audio: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionDetail, Qwen3AsrError> {
        self.handle.transcribe(audio, language)
    }

    /// Transcribe audio from a file
    pub fn transcribe_file(
        &self,
        path: &str,
        language: Option<&str>,
    ) -> Result<TranscriptionDetail, Qwen3AsrError> {
        self.handle.transcribe_file(path, language)
    }
}
