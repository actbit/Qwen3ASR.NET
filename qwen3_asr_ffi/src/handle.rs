//! ASR Handle implementation
//!
//! This module provides the main ASR handle that wraps the qwen3-asr-rs library.

use crate::error::{AudioStage, Qwen3AsrError, Result};
use crate::{Qwen3AsrDevice, TranscriptionDetail};

/// Sample rate for Qwen3-ASR (16kHz)
const SAMPLE_RATE: u32 = 16000;

/// Handle to an ASR instance
///
/// This wraps the actual Qwen3Asr model from qwen3-asr-rs.
pub struct AsrHandle {
    /// Model path or HuggingFace model ID
    model_path: String,
    /// Device for inference
    device: Qwen3AsrDevice,
    /// Optional revision for HuggingFace models
    revision: Option<String>,
    /// Number of threads for CPU inference
    num_threads: usize,
    /// The actual ASR model
    model: Option<Box<Qwen3AsrModel>>,
}

/// Wrapper for the actual Qwen3-ASR model
enum Qwen3AsrModel {
    /// CPU-based model
    Cpu {
        /// In a real implementation, this would be:
        /// model: qwen3_asr::Qwen3Asr
        _placeholder: (),
    },
    /// CUDA-based model
    Cuda {
        _placeholder: (),
    },
    /// Metal-based model (macOS)
    Metal {
        _placeholder: (),
    },
}

impl AsrHandle {
    /// Create a new ASR handle
    ///
    /// # Arguments
    /// * `model_path` - Path to the model or HuggingFace model ID (e.g., "Qwen/Qwen3-ASR-0.6B")
    /// * `device` - Device to use for inference (CPU, CUDA, Metal)
    /// * `revision` - Optional revision for HuggingFace models
    /// * `num_threads` - Number of threads for CPU inference (0 = auto)
    ///
    /// # Errors
    /// Returns an error if:
    /// - The model cannot be found
    /// - The model fails to load
    /// - The device is not available
    pub fn new(
        model_path: String,
        device: Qwen3AsrDevice,
        revision: Option<String>,
        num_threads: usize,
    ) -> Result<Self> {
        // Validate inputs
        if model_path.is_empty() {
            return Err(Qwen3AsrError::invalid_input(
                "model_path",
                "Model path cannot be empty",
            ));
        }

        // Auto-detect thread count
        let num_threads = if num_threads == 0 {
            let detected = num_cpus::get();
            log::info!("Auto-detected {} CPU threads", detected);
            detected
        } else if num_threads > 256 {
            log::warn!("Thread count {} seems too high, capping at 256", num_threads);
            256
        } else {
            num_threads
        };

        let mut handle = Self {
            model_path,
            device,
            revision,
            num_threads,
            model: None,
        };

        handle.load_model()?;
        Ok(handle)
    }

    /// Load the model
    ///
    /// This method handles the model loading process including:
    /// - Device initialization
    /// - Model download (if from HuggingFace)
    /// - Weight loading
    /// - Tokenizer initialization
    fn load_model(&mut self) -> Result<()> {
        log::info!(
            "Loading Qwen3-ASR model from '{}' on device {:?} with {} threads",
            self.model_path,
            self.device,
            self.num_threads
        );

        // Initialize device
        self.initialize_device()?;

        // TODO: Actual implementation with qwen3-asr-rs
        //
        // Example implementation:
        //
        // use qwen3_asr::{Qwen3Asr, LoadOptions};
        // use candle_core::Device;
        //
        // // Step 1: Initialize device
        // let device = match self.device {
        //     Qwen3AsrDevice::Cpu => Device::Cpu,
        //     Qwen3AsrDevice::Cuda => {
        //         Device::new_cuda(0).map_err(|e| {
        //             Qwen3AsrError::device_init_failed("CUDA", e.to_string())
        //         })?
        //     }
        //     Qwen3AsrDevice::Metal => {
        //         Device::new_metal(0).map_err(|e| {
        //             Qwen3AsrError::device_init_failed("Metal", e.to_string())
        //         })?
        //     }
        // };
        //
        // // Step 2: Configure load options
        // let load_options = LoadOptions {
        //     revision: self.revision.clone(),
        //     ..Default::default()
        // };
        //
        // // Step 3: Load model (this may download from HuggingFace)
        // let model = Qwen3Asr::from_pretrained(&self.model_path, &device, &load_options)
        //     .map_err(|e| {
        //         Qwen3AsrError::model_load_failed(&self.model_path, LoadStage::WeightLoading)
        //             .with_context(e.to_string())
        //     })?;
        //
        // self.model = Some(Box::new(model));

        // Placeholder implementation
        let model = match self.device {
            Qwen3AsrDevice::Cpu => Qwen3AsrModel::Cpu { _placeholder: () },
            Qwen3AsrDevice::Cuda => Qwen3AsrModel::Cuda { _placeholder: () },
            Qwen3AsrDevice::Metal => Qwen3AsrModel::Metal { _placeholder: () },
        };

        self.model = Some(Box::new(model));

        log::info!("Model loaded successfully");
        Ok(())
    }

    /// Initialize the compute device
    fn initialize_device(&self) -> Result<()> {
        match self.device {
            Qwen3AsrDevice::Cpu => {
                log::debug!("Using CPU device with {} threads", self.num_threads);
                Ok(())
            }
            Qwen3AsrDevice::Cuda => {
                log::debug!("Initializing CUDA device");
                // TODO: Check CUDA availability when integrating with qwen3-asr-rs
                // For now, log a warning if CUDA might not be available
                log::info!("CUDA device requested - will use GPU acceleration if available");
                Ok(())
            }
            Qwen3AsrDevice::Metal => {
                log::debug!("Initializing Metal device");
                // TODO: Check Metal availability when integrating with qwen3-asr-rs
                #[cfg(not(target_os = "macos"))]
                {
                    log::warn!("Metal is only available on macOS");
                    return Err(Qwen3AsrError::device_not_available("Metal")
                        .with_context("Metal is only supported on macOS"));
                }
                #[cfg(target_os = "macos")]
                {
                    log::info!("Metal device requested - will use GPU acceleration if available");
                    Ok(())
                }
            }
        }
    }

    /// Transcribe audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples as f32 (16kHz, mono)
    /// * `language` - Optional language hint (e.g., "Japanese", "English")
    ///
    /// # Errors
    /// Returns an error if:
    /// - The model is not loaded
    /// - The audio input is empty or invalid
    /// - The inference fails
    ///
    /// # Returns
    /// The transcription result with text, language, confidence, and timestamps.
    pub fn transcribe(
        &self,
        audio: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionDetail> {
        // Validate model state
        let _model = self.model.as_ref().ok_or_else(|| {
            Qwen3AsrError::model_not_loaded()
                .with_context("Cannot transcribe without a loaded model")
        })?;

        // Validate audio input
        self.validate_audio_input(audio)?;

        let duration_ms = (audio.len() as f64 / SAMPLE_RATE as f64 * 1000.0) as u32;
        log::info!(
            "Transcribing {} audio samples ({:.2}s) with language {:?}",
            audio.len(),
            duration_ms as f32 / 1000.0,
            language
        );

        // TODO: Actual implementation with qwen3-asr-rs
        //
        // use qwen3_asr::AudioInput;
        //
        // let audio_input = AudioInput::Waveform {
        //     samples: audio,
        //     sample_rate: SAMPLE_RATE,
        // };
        //
        // let result = model.transcribe(&audio_input, language)
        //     .map_err(|e| {
        //         Qwen3AsrError::inference_failed("Transcription failed")
        //             .with_context(e.to_string())
        //     })?;
        //
        // return Ok(TranscriptionDetail {
        //     text: result.text,
        //     language: result.language,
        //     confidence: result.confidence,
        //     timestamps: result.timestamps.map(|ts| {
        //         ts.into_iter().map(|t| Timestamp {
        //             start: t.start,
        //             end: t.end,
        //             text: t.text,
        //         }).collect()
        //     }),
        // });

        // Placeholder implementation
        Ok(TranscriptionDetail {
            text: "[Qwen3-ASR transcription placeholder]".to_string(),
            language: language.map(|s| s.to_string()),
            confidence: Some(0.95),
            timestamps: None,
        })
    }

    /// Transcribe audio from a file
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, MP3, etc.)
    /// * `language` - Optional language hint
    ///
    /// # Errors
    /// Returns an error if:
    /// - The file does not exist
    /// - The file format is not supported
    /// - The audio cannot be decoded
    /// - The transcription fails
    pub fn transcribe_file(
        &self,
        path: &str,
        language: Option<&str>,
    ) -> Result<TranscriptionDetail> {
        // Validate model state
        if self.model.is_none() {
            return Err(Qwen3AsrError::model_not_loaded()
                .with_context("Cannot transcribe file without a loaded model"));
        }

        // Validate file path
        if path.is_empty() {
            return Err(Qwen3AsrError::invalid_input("path", "File path cannot be empty"));
        }

        log::info!("Transcribing file '{}' with language {:?}", path, language);

        // Read and process audio file
        let audio_samples = self.read_audio_file(path)?;

        // Transcribe the audio
        self.transcribe(&audio_samples, language)
    }

    // =========================================================================
    // Byte Array / Stream Support
    // =========================================================================

    /// Transcribe audio from WAV bytes
    ///
    /// This method allows transcribing audio directly from a byte array
    /// containing WAV file data, without needing to write to disk.
    ///
    /// # Arguments
    /// * `wav_bytes` - Byte array containing WAV file data
    /// * `language` - Optional language hint
    ///
    /// # Errors
    /// Returns an error if:
    /// - The model is not loaded
    /// - The byte array is empty
    /// - The WAV format is invalid
    /// - The audio cannot be decoded
    /// - The transcription fails
    ///
    /// # Example
    /// ```ignore
    /// let wav_data = std::fs::read("audio.wav")?;
    /// let result = asr.transcribe_wav_bytes(&wav_data, Some("Japanese"))?;
    /// ```
    pub fn transcribe_wav_bytes(
        &self,
        wav_bytes: &[u8],
        language: Option<&str>,
    ) -> Result<TranscriptionDetail> {
        // Validate model state
        if self.model.is_none() {
            return Err(Qwen3AsrError::model_not_loaded()
                .with_context("Cannot transcribe without a loaded model"));
        }

        // Validate input
        if wav_bytes.is_empty() {
            return Err(Qwen3AsrError::invalid_input("wav_bytes", "WAV bytes cannot be empty"));
        }

        log::info!(
            "Transcribing WAV bytes ({} bytes) with language {:?}",
            wav_bytes.len(),
            language
        );

        // Parse WAV from bytes
        let audio_samples = self.read_wav_from_bytes(wav_bytes)?;

        // Transcribe the audio
        self.transcribe(&audio_samples, language)
    }

    /// Transcribe audio from raw bytes with format specification
    ///
    /// This method allows transcribing raw audio data (non-WAV) by specifying
    /// the format parameters.
    ///
    /// # Arguments
    /// * `audio_bytes` - Raw audio byte data
    /// * `sample_rate` - Sample rate of the audio (e.g., 16000, 44100)
    /// * `channels` - Number of channels (1 = mono, 2 = stereo)
    /// * `bits_per_sample` - Bits per sample (16, 24, 32)
    /// * `is_float` - Whether the audio is in floating-point format
    /// * `language` - Optional language hint
    ///
    /// # Errors
    /// Returns an error if:
    /// - The model is not loaded
    /// - The audio parameters are invalid
    /// - The audio cannot be converted
    /// - The transcription fails
    ///
    /// # Example
    /// ```ignore
    /// // 16-bit PCM audio at 44.1kHz stereo
    /// let result = asr.transcribe_raw_bytes(
    ///     &raw_audio,
    ///     44100,
    ///     2,
    ///     16,
    ///     false,
    ///     Some("English")
    /// )?;
    /// ```
    pub fn transcribe_raw_bytes(
        &self,
        audio_bytes: &[u8],
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        is_float: bool,
        language: Option<&str>,
    ) -> Result<TranscriptionDetail> {
        // Validate model state
        if self.model.is_none() {
            return Err(Qwen3AsrError::model_not_loaded()
                .with_context("Cannot transcribe without a loaded model"));
        }

        // Validate input
        if audio_bytes.is_empty() {
            return Err(Qwen3AsrError::invalid_input("audio_bytes", "Audio bytes cannot be empty"));
        }

        if channels > 2 {
            return Err(Qwen3AsrError::invalid_input(
                "channels",
                &format!("Unsupported channel count: {} (expected 1 or 2)", channels),
            ));
        }

        if sample_rate == 0 {
            return Err(Qwen3AsrError::invalid_input("sample_rate", "Sample rate cannot be zero"));
        }

        log::info!(
            "Transcribing raw bytes ({} bytes, {} Hz, {} ch, {} bit, float={}) with language {:?}",
            audio_bytes.len(),
            sample_rate,
            channels,
            bits_per_sample,
            is_float,
            language
        );

        // Convert bytes to samples
        let audio_samples = self.bytes_to_samples(
            audio_bytes,
            sample_rate,
            channels,
            bits_per_sample,
            is_float,
        )?;

        // Transcribe the audio
        self.transcribe(&audio_samples, language)
    }

    /// Transcribe 16-bit PCM audio bytes (mono, 16kHz)
    ///
    /// This is a convenience method for the most common raw audio format.
    /// The input should be 16-bit signed integer PCM data at 16kHz mono.
    ///
    /// # Arguments
    /// * `pcm_bytes` - 16-bit PCM audio bytes (little-endian)
    /// * `language` - Optional language hint
    ///
    /// # Errors
    /// Returns an error if:
    /// - The model is not loaded
    /// - The byte count is not a multiple of 2
    /// - The transcription fails
    pub fn transcribe_pcm16(
        &self,
        pcm_bytes: &[u8],
        language: Option<&str>,
    ) -> Result<TranscriptionDetail> {
        // Validate model state
        if self.model.is_none() {
            return Err(Qwen3AsrError::model_not_loaded()
                .with_context("Cannot transcribe without a loaded model"));
        }

        // Validate input
        if pcm_bytes.is_empty() {
            return Err(Qwen3AsrError::invalid_input("pcm_bytes", "PCM bytes cannot be empty"));
        }

        if pcm_bytes.len() % 2 != 0 {
            return Err(Qwen3AsrError::invalid_input(
                "pcm_bytes",
                &format!("PCM byte count must be even (got {})", pcm_bytes.len()),
            ));
        }

        log::info!(
            "Transcribing 16-bit PCM ({} bytes = {} samples) with language {:?}",
            pcm_bytes.len(),
            pcm_bytes.len() / 2,
            language
        );

        // Convert 16-bit PCM to f32 samples
        let samples: Vec<f32> = pcm_bytes
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        // Transcribe the audio
        self.transcribe(&samples, language)
    }

    /// Read WAV data from bytes and convert to f32 samples
    fn read_wav_from_bytes(&self, wav_bytes: &[u8]) -> Result<Vec<f32>> {
        use hound::WavReader;

        log::debug!("Parsing WAV from {} bytes", wav_bytes.len());

        let cursor = std::io::Cursor::new(wav_bytes);
        let reader = WavReader::new(cursor).map_err(|e| {
            Qwen3AsrError::audio_processing_failed(AudioStage::Reading)
                .with_context(format!("Failed to parse WAV: {}", e))
        })?;

        let spec = reader.spec();
        log::debug!(
            "WAV format: {} channels, {} Hz, {} bits, {:?}",
            spec.channels,
            spec.sample_rate,
            spec.bits_per_sample,
            spec.sample_format
        );

        // Validate format
        if spec.channels > 2 {
            return Err(Qwen3AsrError::invalid_input(
                "channels",
                &format!("Unsupported channel count: {} (expected 1 or 2)", spec.channels),
            ));
        }

        // Convert to f32 samples
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().filter_map(|s| s.ok()).collect()
            }
            hound::SampleFormat::Int => {
                let max_val = (1 << (spec.bits_per_sample - 1)) as f64;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| (s as f64 / max_val) as f32)
                    .collect()
            }
        };

        log::debug!("Read {} raw samples", samples.len());

        // Convert to mono if stereo
        let mono_samples = if spec.channels == 2 {
            log::debug!("Converting stereo to mono");
            samples
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
                .collect()
        } else {
            samples
        };

        // Resample to 16kHz if needed
        let resampled = if spec.sample_rate != SAMPLE_RATE {
            log::debug!(
                "Resampling from {} Hz to {} Hz",
                spec.sample_rate,
                SAMPLE_RATE
            );
            self.resample(&mono_samples, spec.sample_rate, SAMPLE_RATE)?
        } else {
            mono_samples
        };

        log::debug!("Final audio: {} samples at {} Hz", resampled.len(), SAMPLE_RATE);
        Ok(resampled)
    }

    /// Convert raw bytes to f32 samples
    fn bytes_to_samples(
        &self,
        bytes: &[u8],
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        is_float: bool,
    ) -> Result<Vec<f32>> {
        let bytes_per_sample = (bits_per_sample / 8) as usize;
        let frame_size = bytes_per_sample * channels as usize;

        if bytes.len() % frame_size != 0 {
            return Err(Qwen3AsrError::invalid_input(
                "audio_bytes",
                &format!(
                    "Byte count {} is not a multiple of frame size {}",
                    bytes.len(),
                    frame_size
                ),
            ));
        }

        log::debug!(
            "Converting {} bytes to samples ({} ch, {} bit, float={})",
            bytes.len(),
            channels,
            bits_per_sample,
            is_float
        );

        let samples: Vec<f32> = if is_float {
            match bits_per_sample {
                32 => bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
                64 => bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        val as f32
                    })
                    .collect(),
                _ => {
                    return Err(Qwen3AsrError::invalid_input(
                        "bits_per_sample",
                        &format!("Unsupported float bit depth: {}", bits_per_sample),
                    ));
                }
            }
        } else {
            let _max_val = (1i64 << (bits_per_sample - 1)) as f64;
            match bits_per_sample {
                8 => bytes.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
                16 => bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect(),
                24 => bytes
                    .chunks_exact(3)
                    .map(|chunk| {
                        let sample = ((chunk[0] as u32)
                            | ((chunk[1] as u32) << 8)
                            | ((chunk[2] as u32) << 16)) as i32;
                        // Sign extend from 24-bit to 32-bit
                        let sample = if sample & 0x800000 != 0 {
                            sample | (!0xFFFFFFu32 as i32)
                        } else {
                            sample
                        };
                        sample as f32 / 8388608.0
                    })
                    .collect(),
                32 => bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        sample as f32 / 2147483648.0
                    })
                    .collect(),
                _ => {
                    return Err(Qwen3AsrError::invalid_input(
                        "bits_per_sample",
                        &format!("Unsupported integer bit depth: {}", bits_per_sample),
                    ));
                }
            }
        };

        // Convert to mono if stereo
        let mono_samples = if channels == 2 {
            log::debug!("Converting stereo to mono");
            samples
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
                .collect()
        } else {
            samples
        };

        // Resample to 16kHz if needed
        let resampled = if sample_rate != SAMPLE_RATE {
            log::debug!("Resampling from {} Hz to {} Hz", sample_rate, SAMPLE_RATE);
            self.resample(&mono_samples, sample_rate, SAMPLE_RATE)?
        } else {
            mono_samples
        };

        Ok(resampled)
    }

    /// Validate audio input
    fn validate_audio_input(&self, audio: &[f32]) -> Result<()> {
        if audio.is_empty() {
            return Err(Qwen3AsrError::invalid_input("audio", "Audio samples cannot be empty")
                .with_context("Empty audio input provided"));
        }

        // Check for minimum duration (at least 100ms)
        let min_samples = (SAMPLE_RATE as f32 * 0.1) as usize;
        if audio.len() < min_samples {
            return Err(Qwen3AsrError::invalid_input(
                "audio",
                &format!("Audio too short: {} samples (minimum: {})", audio.len(), min_samples),
            ));
        }

        // Check for NaN or infinity values
        let has_invalid = audio.iter().any(|&s| s.is_nan() || s.is_infinite());
        if has_invalid {
            return Err(Qwen3AsrError::invalid_input(
                "audio",
                "Audio contains NaN or infinite values",
            ));
        }

        Ok(())
    }

    /// Read an audio file and convert to f32 samples
    fn read_audio_file(&self, path: &str) -> Result<Vec<f32>> {
        use hound::WavReader;

        log::debug!("Reading audio file: {}", path);

        // Check if file exists first
        if !std::path::Path::new(path).exists() {
            return Err(Qwen3AsrError::file_not_found(path));
        }

        // Open file
        let reader = WavReader::open(path).map_err(|e| {
            Qwen3AsrError::audio_processing_failed(AudioStage::Reading)
                .with_context(format!("Failed to open file: {}", e))
        })?;

        let spec = reader.spec();
        log::debug!(
            "Audio format: {} channels, {} Hz, {} bits, {:?}",
            spec.channels,
            spec.sample_rate,
            spec.bits_per_sample,
            spec.sample_format
        );

        // Validate format
        if spec.channels > 2 {
            return Err(Qwen3AsrError::invalid_input(
                "channels",
                &format!("Unsupported channel count: {} (expected 1 or 2)", spec.channels),
            ));
        }

        // Convert to f32 samples
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader
                    .into_samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect()
            }
            hound::SampleFormat::Int => {
                let max_val = (1 << (spec.bits_per_sample - 1)) as f64;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| (s as f64 / max_val) as f32)
                    .collect()
            }
        };

        log::debug!("Read {} raw samples", samples.len());

        // Convert to mono if stereo
        let mono_samples = if spec.channels == 2 {
            log::debug!("Converting stereo to mono");
            samples
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
                .collect()
        } else {
            samples
        };

        // Resample to 16kHz if needed
        let resampled = if spec.sample_rate != SAMPLE_RATE {
            log::debug!(
                "Resampling from {} Hz to {} Hz",
                spec.sample_rate,
                SAMPLE_RATE
            );
            self.resample(&mono_samples, spec.sample_rate, SAMPLE_RATE)?
        } else {
            mono_samples
        };

        log::debug!("Final audio: {} samples at {} Hz", resampled.len(), SAMPLE_RATE);
        Ok(resampled)
    }

    /// Simple linear resampling
    ///
    /// For production use, consider using a proper resampling library like rubr.
    fn resample(
        &self,
        samples: &[f32],
        from_rate: u32,
        to_rate: u32,
    ) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }

        if from_rate == 0 || to_rate == 0 {
            return Err(Qwen3AsrError::audio_processing_failed(AudioStage::Resampling)
                .with_context("Invalid sample rate (zero)"));
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let output_len = (samples.len() as f64 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 / ratio;
            let idx = src_idx as usize;
            let frac = src_idx - idx as f64;

            let sample = if idx + 1 < samples.len() {
                samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
            } else if idx < samples.len() {
                samples[idx]
            } else {
                0.0
            };

            output.push(sample);
        }

        Ok(output)
    }

    /// Get the model path
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Get the device
    pub fn device(&self) -> Qwen3AsrDevice {
        self.device
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }
}

impl Drop for AsrHandle {
    fn drop(&mut self) {
        log::info!("Destroying ASR handle for '{}'", self.model_path);
        // The model will be automatically dropped when this struct is dropped
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for AsrHandle {}
unsafe impl Sync for AsrHandle {}
