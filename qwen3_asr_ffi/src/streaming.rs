//! Streaming transcription implementation

use crate::error::Qwen3AsrError;
use crate::{AsrHandle, TranscriptionDetail, Timestamp};

/// Streaming transcriber for real-time transcription
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
    /// Sample rate (16kHz for Qwen3-ASR)
    sample_rate: u32,
    /// Current partial result
    partial_result: Option<TranscriptionDetail>,
    /// Accumulated transcription segments
    segments: Vec<TranscriptionSegment>,
    /// Whether the stream is finished
    finished: bool,
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
    /// # Safety
    /// The handle must remain valid for the lifetime of the transcriber
    pub fn new(
        handle: &AsrHandle,
        language: Option<String>,
        chunk_size_sec: f32,
        enable_timestamps: bool,
        enable_partial_results: bool,
    ) -> Result<Self, Qwen3AsrError> {
        Ok(Self {
            handle: handle as *const AsrHandle,
            language,
            chunk_size_sec: if chunk_size_sec <= 0.0 {
                0.5
            } else {
                chunk_size_sec
            },
            enable_timestamps,
            enable_partial_results,
            buffer: Vec::new(),
            sample_rate: 16000, // Qwen3-ASR uses 16kHz
            partial_result: None,
            segments: Vec::new(),
            finished: false,
        })
    }

    /// Push audio samples to the stream
    pub fn push(&mut self, audio: &[f32]) -> Result<TranscriptionDetail, Qwen3AsrError> {
        if self.finished {
            return Err(Qwen3AsrError::StreamingError(
                "Stream already finished".to_string(),
            ));
        }

        // Add audio to buffer
        self.buffer.extend_from_slice(audio);

        // Calculate chunk size in samples
        let chunk_samples = (self.chunk_size_sec * self.sample_rate as f32) as usize;

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
    fn process_chunk(&mut self, chunk: &[f32]) -> Result<(), Qwen3AsrError> {
        // Safety: handle is valid as long as the AsrHandle exists
        let handle = unsafe { &*self.handle };

        // Perform transcription on chunk
        let result = handle.transcribe(chunk, self.language.as_deref())?;

        if !result.text.is_empty() {
            // Calculate timing for this segment
            let segment_count = self.segments.len();
            let start_time = segment_count as f32 * self.chunk_size_sec;
            let end_time = start_time + self.chunk_size_sec;

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
    pub fn get_partial(&self) -> Option<TranscriptionDetail> {
        self.partial_result.clone()
    }

    /// Finish the stream and return the final result
    pub fn finish(mut self) -> Result<TranscriptionDetail, Qwen3AsrError> {
        if self.finished {
            return Err(Qwen3AsrError::StreamingError(
                "Stream already finished".to_string(),
            ));
        }

        self.finished = true;

        // Process any remaining audio in buffer
        if !self.buffer.is_empty() {
            let handle = unsafe { &*self.handle };
            let result = handle.transcribe(&self.buffer, self.language.as_deref())?;

            if !result.text.is_empty() {
                let segment_count = self.segments.len();
                let start_time = segment_count as f32 * self.chunk_size_sec;
                let end_time = start_time + (self.buffer.len() as f32 / self.sample_rate as f32);

                self.segments.push(TranscriptionSegment {
                    text: result.text,
                    start_time,
                    end_time,
                });
            }

            self.buffer.clear();
        }

        Ok(self.build_current_result())
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
}

// Implement Send and Sync for thread safety
unsafe impl Send for StreamingTranscriber {}
unsafe impl Sync for StreamingTranscriber {}
