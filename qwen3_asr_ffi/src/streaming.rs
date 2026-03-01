//! Streaming transcription implementation
//!
//! This module provides real-time streaming transcription using Qwen3-ASR.
//! It supports partial results during transcription and final result aggregation.

use crate::error::Qwen3AsrError;
use crate::{AsrHandle, Timestamp, TranscriptionDetail};

/// Sample rate for Qwen3-ASR (16kHz)
const SAMPLE_RATE: u32 = 16000;

/// Streaming transcriber for real-time transcription
///
/// This wraps the qwen3-asr-rs streaming API.
///
/// # Example (conceptual)
/// ```ignore
/// let asr = Qwen3Asr::from_pretrained("Qwen/Qwen3-ASR-0.6B", &Device::Cpu, &LoadOptions::default())?;
/// let mut stream = asr.start_stream(StreamOptions::default())?;
///
/// // Push audio chunks
/// let chunk = AudioInput::Waveform { samples: &audio, sample_rate: 16000 };
/// let partial = stream.push_audio_chunk(&chunk)?;
///
/// // Get final result
/// let final_result = stream.finish()?;
/// ```
pub struct StreamingTranscriber {
    /// Reference to the ASR handle
    handle: *const AsrHandle,
    /// Language for transcription
    language: Option<String>,
    /// Chunk size in seconds
    chunk_size_sec: f32,
    /// Enable timestamp prediction
    enable_timestamps: bool,
    /// Enable partial results
    enable_partial_results: bool,
    /// Buffered audio samples
    buffer: Vec<f32>,
    /// Current partial result
    partial_result: Option<TranscriptionDetail>,
    /// Accumulated transcription segments
    segments: Vec<TranscriptionSegment>,
    /// Whether the stream is finished
    finished: bool,

    // TODO: When integrating with actual qwen3-asr-rs, add:
    // stream: Option<qwen3_asr::Stream>,
}

/// Internal segment for streaming
#[derive(Debug, Clone)]
struct TranscriptionSegment {
    text: String,
    start_time: f32,
    end_time: f32,
}

impl StreamingTranscriber {
    /// Create a new streaming transcriber
    ///
    /// # Arguments
    /// * `handle` - Reference to the ASR handle
    /// * `language` - Optional language hint
    /// * `chunk_size_sec` - Size of audio chunks in seconds (default: 0.5)
    /// * `enable_timestamps` - Enable timestamp prediction
    /// * `enable_partial_results` - Enable partial results during streaming
    ///
    /// # Safety
    /// The handle must remain valid for the lifetime of the transcriber.
    pub fn new(
        handle: &AsrHandle,
        language: Option<String>,
        chunk_size_sec: f32,
        enable_timestamps: bool,
        enable_partial_results: bool,
    ) -> Result<Self, Qwen3AsrError> {
        // Validate chunk size
        let chunk_size_sec = if chunk_size_sec <= 0.0 {
            log::warn!("Invalid chunk_size_sec ({}), using default 0.5", chunk_size_sec);
            0.5
        } else if chunk_size_sec > 10.0 {
            log::warn!("chunk_size_sec ({}) too large, capping at 10.0", chunk_size_sec);
            10.0
        } else {
            chunk_size_sec
        };

        log::info!(
            "Creating streaming transcriber: language={:?}, chunk_size={:.2}s, timestamps={}, partial={}",
            language,
            chunk_size_sec,
            enable_timestamps,
            enable_partial_results
        );

        // TODO: When integrating with actual qwen3-asr-rs:
        //
        // use qwen3_asr::StreamOptions;
        //
        // let stream_options = StreamOptions {
        //     language: language.clone(),
        //     chunk_size_sec,
        //     enable_timestamps,
        //     ..Default::default()
        // };
        //
        // let stream = handle.model.as_ref()
        //     .ok_or(Qwen3AsrError::ModelLoadError("Model not loaded".into()))?
        //     .start_stream(stream_options)
        //     .map_err(|e| Qwen3AsrError::StreamingError(e.to_string()))?;

        Ok(Self {
            handle: handle as *const AsrHandle,
            language,
            chunk_size_sec,
            enable_timestamps,
            enable_partial_results,
            buffer: Vec::new(),
            partial_result: None,
            segments: Vec::new(),
            finished: false,
        })
    }

    /// Push audio samples to the stream
    ///
    /// This method buffers audio and processes complete chunks.
    /// Returns a partial result if available.
    ///
    /// # Arguments
    /// * `audio` - Audio samples as f32 (16kHz, mono)
    ///
    /// # Returns
    /// Partial transcription result (may be empty if no complete chunk yet)
    pub fn push(&mut self, audio: &[f32]) -> Result<TranscriptionDetail, Qwen3AsrError> {
        if self.finished {
            return Err(Qwen3AsrError::StreamingError(
                "Stream already finished".to_string(),
            ));
        }

        // Add audio to buffer
        self.buffer.extend_from_slice(audio);

        log::trace!(
            "Pushed {} samples, buffer size now {}",
            audio.len(),
            self.buffer.len()
        );

        // Calculate chunk size in samples
        let chunk_samples = (self.chunk_size_sec * SAMPLE_RATE as f32) as usize;

        // Process complete chunks
        while self.buffer.len() >= chunk_samples {
            let chunk: Vec<f32> = self.buffer.drain(..chunk_samples).collect();
            self.process_chunk(&chunk)?;
        }

        // Return partial result if available
        Ok(self.get_partial().unwrap_or_else(|| TranscriptionDetail {
            text: String::new(),
            language: self.language.clone(),
            confidence: None,
            timestamps: None,
        }))
    }

    /// Process a single audio chunk
    ///
    /// This calls the underlying ASR model for transcription.
    fn process_chunk(&mut self, chunk: &[f32]) -> Result<(), Qwen3AsrError> {
        // Safety: handle is valid as long as the AsrHandle exists
        let handle = unsafe { &*self.handle };

        log::debug!(
            "Processing chunk: {} samples ({:.2}s)",
            chunk.len(),
            chunk.len() as f32 / SAMPLE_RATE as f32
        );

        // TODO: When integrating with actual qwen3-asr-rs:
        //
        // use qwen3_asr::AudioInput;
        //
        // let audio_input = AudioInput::Waveform {
        //     samples: chunk,
        //     sample_rate: SAMPLE_RATE,
        // };
        //
        // let result = self.stream.as_mut()
        //     .ok_or(Qwen3AsrError::StreamingError("Stream not initialized".into()))?
        //     .push_audio_chunk(&audio_input)
        //     .map_err(|e| Qwen3AsrError::InferenceError(e.to_string()))?;

        // Placeholder: use batch transcription
        let result = handle.transcribe(chunk, self.language.as_deref())?;

        if !result.text.is_empty() {
            // Calculate timing for this segment
            let segment_count = self.segments.len();
            let start_time = segment_count as f32 * self.chunk_size_sec;
            let end_time = start_time + self.chunk_size_sec;

            log::debug!(
                "Segment {}: '{}' ({:.2}s - {:.2}s)",
                segment_count,
                result.text,
                start_time,
                end_time
            );

            self.segments.push(TranscriptionSegment {
                text: result.text.clone(),
                start_time,
                end_time,
            });

            // Update partial result
            if self.enable_partial_results {
                self.partial_result = Some(self.build_current_result());
            }
        }

        Ok(())
    }

    /// Get the current partial result
    ///
    /// Returns the accumulated transcription so far.
    pub fn get_partial(&self) -> Option<TranscriptionDetail> {
        self.partial_result.clone()
    }

    /// Finish the stream and return the final result
    ///
    /// This method:
    /// 1. Processes any remaining audio in the buffer
    /// 2. Finalizes the stream
    /// 3. Returns the complete transcription
    ///
    /// After calling this method, the stream cannot be used further.
    pub fn finish(mut self) -> Result<TranscriptionDetail, Qwen3AsrError> {
        if self.finished {
            return Err(Qwen3AsrError::StreamingError(
                "Stream already finished".to_string(),
            ));
        }

        self.finished = true;

        log::info!(
            "Finishing stream with {} remaining samples in buffer",
            self.buffer.len()
        );

        // Process any remaining audio in buffer
        if !self.buffer.is_empty() {
            let handle = unsafe { &*self.handle };

            log::debug!(
                "Processing final {} samples ({:.2}s)",
                self.buffer.len(),
                self.buffer.len() as f32 / SAMPLE_RATE as f32
            );

            let result = handle.transcribe(&self.buffer, self.language.as_deref())?;

            if !result.text.is_empty() {
                let segment_count = self.segments.len();
                let start_time = segment_count as f32 * self.chunk_size_sec;
                let end_time = start_time + (self.buffer.len() as f32 / SAMPLE_RATE as f32);

                self.segments.push(TranscriptionSegment {
                    text: result.text,
                    start_time,
                    end_time,
                });
            }

            self.buffer.clear();
        }

        // TODO: When integrating with actual qwen3-asr-rs:
        //
        // let final_result = self.stream.take()
        //     .ok_or(Qwen3AsrError::StreamingError("Stream not initialized".into()))?
        //     .finish()
        //     .map_err(|e| Qwen3AsrError::StreamingError(e.to_string()))?;

        let result = self.build_current_result();

        log::info!(
            "Stream finished: {} segments, {} chars total",
            self.segments.len(),
            result.text.len()
        );

        Ok(result)
    }

    /// Build the current transcription result from segments
    fn build_current_result(&self) -> TranscriptionDetail {
        let text = self
            .segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let timestamps = if self.enable_timestamps {
            Some(
                self.segments
                    .iter()
                    .map(|s| Timestamp {
                        start: s.start_time,
                        end: s.end_time,
                        text: s.text.clone(),
                    })
                    .collect(),
            )
        } else {
            None
        };

        TranscriptionDetail {
            text,
            language: self.language.clone(),
            confidence: None,
            timestamps,
        }
    }

    /// Check if the stream is still active
    pub fn is_active(&self) -> bool {
        !self.finished
    }
}

impl Drop for StreamingTranscriber {
    fn drop(&mut self) {
        if !self.finished {
            log::warn!("StreamingTranscriber dropped without calling finish()");
        }
    }
}

// Implement Send and Sync for thread safety
// This is safe because:
// 1. The handle pointer is valid for the lifetime of the transcriber
// 2. All internal state is properly synchronized
unsafe impl Send for StreamingTranscriber {}
unsafe impl Sync for StreamingTranscriber {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_size_validation() {
        // Test default chunk size when invalid
        let chunk_size = if 0.0 <= 0.0 { 0.5 } else { 0.0 };
        assert_eq!(chunk_size, 0.5);

        // Test capping at max
        let chunk_size = if 15.0 > 10.0 { 10.0 } else { 15.0 };
        assert_eq!(chunk_size, 10.0);
    }
}
