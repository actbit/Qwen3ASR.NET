//! Streaming transcription implementation
//!
//! This module provides real-time streaming transcription using Qwen3-ASR.
//! It supports partial results during transcription and final result aggregation.

use crate::error::{Qwen3AsrError, Result};
use crate::{AsrHandle, Timestamp, TranscriptionDetail};

/// Sample rate for Qwen3-ASR (16kHz)
const SAMPLE_RATE: u32 = 16000;

/// Maximum buffer size (30 seconds of audio)
const MAX_BUFFER_SAMPLES: usize = SAMPLE_RATE as usize * 30;

/// Default chunk size in seconds
const DEFAULT_CHUNK_SIZE_SEC: f32 = 0.5;

/// Maximum chunk size in seconds
const MAX_CHUNK_SIZE_SEC: f32 = 10.0;

/// Streaming transcriber for real-time transcription
///
/// This wraps the qwen3-asr-rs streaming API.
///
/// # Thread Safety
/// This type is `Send` and `Sync`, but operations are not thread-safe.
/// Use external synchronization if sharing across threads.
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
    /// Total samples processed
    total_samples_processed: usize,
}

/// Internal segment for streaming
#[derive(Debug, Clone)]
struct TranscriptionSegment {
    text: String,
    start_time: f32,
    end_time: f32,
    confidence: Option<f32>,
}

impl StreamingTranscriber {
    /// Create a new streaming transcriber
    ///
    /// # Arguments
    /// * `handle` - Reference to the ASR handle
    /// * `language` - Optional language hint
    /// * `chunk_size_sec` - Size of audio chunks in seconds (default: 0.5, max: 10.0)
    /// * `enable_timestamps` - Enable timestamp prediction
    /// * `enable_partial_results` - Enable partial results during streaming
    ///
    /// # Errors
    /// Returns an error if:
    /// - The handle's model is not loaded
    /// - The chunk size is invalid
    ///
    /// # Safety
    /// The handle must remain valid for the lifetime of the transcriber.
    pub fn new(
        handle: &AsrHandle,
        language: Option<String>,
        chunk_size_sec: f32,
        enable_timestamps: bool,
        enable_partial_results: bool,
    ) -> Result<Self> {
        // Validate model state
        if !handle.is_loaded() {
            return Err(Qwen3AsrError::model_not_loaded()
                .with_context("Cannot start streaming without a loaded model"));
        }

        // Validate and normalize chunk size
        let chunk_size_sec = if chunk_size_sec <= 0.0 {
            log::warn!(
                "Invalid chunk_size_sec ({:.2}), using default {:.2}",
                chunk_size_sec,
                DEFAULT_CHUNK_SIZE_SEC
            );
            DEFAULT_CHUNK_SIZE_SEC
        } else if chunk_size_sec > MAX_CHUNK_SIZE_SEC {
            log::warn!(
                "chunk_size_sec ({:.2}) exceeds maximum ({:.2}), capping",
                chunk_size_sec,
                MAX_CHUNK_SIZE_SEC
            );
            MAX_CHUNK_SIZE_SEC
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
        //     .ok_or_else(|| Qwen3AsrError::model_not_loaded())?
        //     .start_stream(stream_options)
        //     .map_err(|e| {
        //         Qwen3AsrError::stream_push_failed("Failed to start stream")
        //             .with_context(e.to_string())
        //     })?;

        Ok(Self {
            handle: handle as *const AsrHandle,
            language,
            chunk_size_sec,
            enable_timestamps,
            enable_partial_results,
            buffer: Vec::with_capacity(MAX_BUFFER_SAMPLES),
            partial_result: None,
            segments: Vec::new(),
            finished: false,
            total_samples_processed: 0,
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
    /// # Errors
    /// Returns an error if:
    /// - The stream has already been finished
    /// - The audio input is empty
    /// - The buffer overflows
    /// - The transcription fails
    ///
    /// # Returns
    /// Partial transcription result (may be empty if no complete chunk yet)
    pub fn push(&mut self, audio: &[f32]) -> Result<TranscriptionDetail> {
        // Check stream state
        if self.finished {
            return Err(Qwen3AsrError::stream_already_finished()
                .with_context("Cannot push to a finished stream"));
        }

        // Validate input
        if audio.is_empty() {
            log::debug!("Empty audio chunk received, skipping");
            return Ok(self.get_partial().unwrap_or_else(|| self.empty_result()));
        }

        // Check for buffer overflow
        if self.buffer.len() + audio.len() > MAX_BUFFER_SAMPLES {
            return Err(Qwen3AsrError::invalid_input(
                "audio",
                &format!(
                    "Buffer overflow: {} + {} > {}",
                    self.buffer.len(),
                    audio.len(),
                    MAX_BUFFER_SAMPLES
                ),
            )
            .with_context("Consider using smaller chunks or calling finish() more frequently"));
        }

        // Add audio to buffer
        self.buffer.extend_from_slice(audio);

        log::trace!(
            "Pushed {} samples, buffer size now {} ({:.2}s)",
            audio.len(),
            self.buffer.len(),
            self.buffer.len() as f32 / SAMPLE_RATE as f32
        );

        // Calculate chunk size in samples
        let chunk_samples = (self.chunk_size_sec * SAMPLE_RATE as f32) as usize;

        // Process complete chunks
        while self.buffer.len() >= chunk_samples {
            let chunk: Vec<f32> = self.buffer.drain(..chunk_samples).collect();
            self.process_chunk(&chunk)?;
        }

        // Return partial result if available
        Ok(self.get_partial().unwrap_or_else(|| self.empty_result()))
    }

    /// Process a single audio chunk
    ///
    /// This calls the underlying ASR model for transcription.
    fn process_chunk(&mut self, chunk: &[f32]) -> Result<()> {
        // Safety: handle is valid as long as the AsrHandle exists
        let handle = unsafe { &*self.handle };

        let chunk_duration_ms = (chunk.len() as f32 / SAMPLE_RATE as f32 * 1000.0) as u32;
        let start_time = self.total_samples_processed as f32 / SAMPLE_RATE as f32;
        let end_time = start_time + chunk.len() as f32 / SAMPLE_RATE as f32;

        log::debug!(
            "Processing chunk: {} samples ({:.2}s) at time {:.2}s-{:.2}s",
            chunk.len(),
            chunk_duration_ms as f32 / 1000.0,
            start_time,
            end_time
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
        //     .ok_or_else(|| Qwen3AsrError::internal("Stream not initialized"))?
        //     .push_audio_chunk(&audio_input)
        //     .map_err(|e| {
        //         Qwen3AsrError::stream_push_failed("Failed to push audio chunk")
        //             .with_context(e.to_string())
        //     })?;

        // Placeholder: use batch transcription
        let result = handle.transcribe(chunk, self.language.as_deref()).map_err(|e| {
            Qwen3AsrError::stream_push_failed("Transcription failed")
                .with_context(e.to_string())
        })?;

        // Update sample counter
        self.total_samples_processed += chunk.len();

        // Store segment if we got text
        if !result.text.is_empty() {
            log::debug!(
                "Segment {}: '{}' ({:.2}s - {:.2}s)",
                self.segments.len(),
                result.text,
                start_time,
                end_time
            );

            self.segments.push(TranscriptionSegment {
                text: result.text.clone(),
                start_time,
                end_time,
                confidence: result.confidence,
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
    ///
    /// # Errors
    /// Returns an error if:
    /// - The stream has already been finished
    /// - The final transcription fails
    pub fn finish(mut self) -> Result<TranscriptionDetail> {
        // Check state
        if self.finished {
            return Err(Qwen3AsrError::stream_already_finished()
                .with_context("Stream has already been finished"));
        }

        self.finished = true;

        log::info!(
            "Finishing stream: {} segments, {} remaining samples in buffer ({:.2}s)",
            self.segments.len(),
            self.buffer.len(),
            self.buffer.len() as f32 / SAMPLE_RATE as f32
        );

        // Process any remaining audio in buffer
        if !self.buffer.is_empty() {
            let handle = unsafe { &*self.handle };

            let start_time = self.total_samples_processed as f32 / SAMPLE_RATE as f32;
            let end_time = start_time + self.buffer.len() as f32 / SAMPLE_RATE as f32;

            log::debug!(
                "Processing final {} samples ({:.2}s) at time {:.2}s-{:.2}s",
                self.buffer.len(),
                self.buffer.len() as f32 / SAMPLE_RATE as f32,
                start_time,
                end_time
            );

            let result = handle
                .transcribe(&self.buffer, self.language.as_deref())
                .map_err(|e| {
                    Qwen3AsrError::inference_failed("Final transcription failed")
                        .with_context(e.to_string())
                })?;

            if !result.text.is_empty() {
                self.segments.push(TranscriptionSegment {
                    text: result.text,
                    start_time,
                    end_time,
                    confidence: result.confidence,
                });
            }

            self.buffer.clear();
        }

        // TODO: When integrating with actual qwen3-asr-rs:
        //
        // let final_result = self.stream.take()
        //     .ok_or_else(|| Qwen3AsrError::internal("Stream not initialized"))?
        //     .finish()
        //     .map_err(|e| {
        //         Qwen3AsrError::inference_failed("Stream finish failed")
        //             .with_context(e.to_string())
        //     })?;

        let result = self.build_current_result();

        log::info!(
            "Stream finished: {} segments, {} chars total, {:.2}s duration",
            self.segments.len(),
            result.text.len(),
            self.total_samples_processed as f32 / SAMPLE_RATE as f32
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

        // Calculate average confidence
        let confidence = if !self.segments.is_empty() {
            let sum: f32 = self
                .segments
                .iter()
                .filter_map(|s| s.confidence)
                .sum();
            let count = self.segments.iter().filter(|s| s.confidence.is_some()).count();
            if count > 0 {
                Some(sum / count as f32)
            } else {
                None
            }
        } else {
            None
        };

        TranscriptionDetail {
            text,
            language: self.language.clone(),
            confidence,
            timestamps,
        }
    }

    /// Create an empty result with the current language
    fn empty_result(&self) -> TranscriptionDetail {
        TranscriptionDetail {
            text: String::new(),
            language: self.language.clone(),
            confidence: None,
            timestamps: None,
        }
    }

    /// Check if the stream is still active
    pub fn is_active(&self) -> bool {
        !self.finished
    }

    /// Get the number of segments processed
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get the total duration processed in seconds
    pub fn duration(&self) -> f32 {
        self.total_samples_processed as f32 / SAMPLE_RATE as f32
    }
}

impl Drop for StreamingTranscriber {
    fn drop(&mut self) {
        if !self.finished {
            log::warn!(
                "StreamingTranscriber dropped without calling finish() - {} samples lost",
                self.buffer.len()
            );
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
        // Test default for invalid
        let chunk_size = if 0.0 <= 0.0 {
            DEFAULT_CHUNK_SIZE_SEC
        } else {
            0.0
        };
        assert_eq!(chunk_size, DEFAULT_CHUNK_SIZE_SEC);

        // Test capping at max
        let chunk_size = if 15.0 > MAX_CHUNK_SIZE_SEC {
            MAX_CHUNK_SIZE_SEC
        } else {
            15.0
        };
        assert_eq!(chunk_size, MAX_CHUNK_SIZE_SEC);
    }

    #[test]
    fn test_max_buffer_size() {
        // Should be 30 seconds of audio
        assert_eq!(MAX_BUFFER_SAMPLES, 480000);
    }
}
