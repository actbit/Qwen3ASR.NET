# Qwen3ASR.NET

[![NuGet](https://img.shields.io/nuget/v/Qwen3ASR.NET.svg)](https://www.nuget.org/packages/Qwen3ASR.NET/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

**Qwen3ASR.NET** is a .NET wrapper for [Qwen3-ASR](https://github.com/lumosimmo/qwen3-asr-rs), Alibaba's state-of-the-art speech recognition implementation in Rust.

## Features

- 🎯 **High Accuracy** - Based on Qwen3-ASR model
- 🌍 **Multi-language Support** - Japanese, English, Chinese, and more
- 📦 **Easy Integration** - NuGet packages for .NET 8.0+
- 🔄 **Streaming Support** - Real-time transcription with partial results
- ⏱️ **Timestamp Prediction** - Word-level timing information
- 🖥️ **Cross-Platform** - Windows, Linux, and macOS (x64 and ARM64)

## Installation

### All Platforms (Meta Package)
```xml
<PackageReference Include="Qwen3ASR.NET" Version="1.0.0" />
```

### Platform-Specific (Smaller Deployment)
```xml
<PackageReference Include="Qwen3ASR.NET.Core" Version="1.0.0" />
<!-- Choose your platform -->
<PackageReference Include="Qwen3ASR.NET.Runtime.Win-x64" Version="1.0.0" />
```

Available runtime packages:
- `Qwen3ASR.NET.Runtime.Win-x64` - Windows x64
- `Qwen3ASR.NET.Runtime.Linux-x64` - Linux x64
- `Qwen3ASR.NET.Runtime.OSX-x64` - macOS Intel
- `Qwen3ASR.NET.Runtime.OSX-arm64` - macOS Apple Silicon

## Quick Start

### Offline Transcription

```csharp
using Qwen3ASR.NET;
using Qwen3ASR.NET.Enums;

// Load model (first run will download from HuggingFace)
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cpu
);

// Transcribe audio file
var result = await asr.TranscribeFileAsync("audio.wav", language: "Japanese");
Console.WriteLine(result.Text);

// Transcribe raw audio samples (16kHz mono f32)
float[] samples = LoadAudioSamples(); // Your audio loading code
var result2 = await asr.TranscribeAsync(samples, language: "Japanese");
Console.WriteLine(result2.Text);
```

### Streaming Transcription

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");

// Start streaming session
await using var stream = asr.StartStream(new StreamOptions
{
    Language = "Japanese",
    ChunkSizeSec = 0.5f,
    EnableTimestamps = true,
    EnablePartialResults = true
});

// Push audio chunks as they arrive
foreach (var chunk in audioChunks)
{
    var partial = await stream.PushAsync(chunk);
    Console.WriteLine($"Partial: {partial.Text}");
}

// Get final result
var final = await stream.FinishAsync();
Console.WriteLine($"Final: {final.Text}");
```

### GPU Acceleration

```csharp
// CUDA (NVIDIA GPU)
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cuda
);

// Metal (macOS Apple Silicon)
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Metal
);
```

## API Reference

### `Qwen3Asr`

Main class for speech recognition.

| Method | Description |
|--------|-------------|
| `FromPretrainedAsync(modelPath, device)` | Load a pretrained model |
| `TranscribeFileAsync(filePath, language)` | Transcribe an audio file |
| `TranscribeAsync(samples, language)` | Transcribe audio samples |
| `StartStream(options)` | Start streaming transcription |
| `GetVersion()` | Get library version |

### `StreamingTranscriber`

Real-time transcription session.

| Method | Description |
|--------|-------------|
| `PushAsync(samples)` | Push audio samples |
| `GetPartialResult()` | Get current partial result |
| `FinishAsync()` | Finish and get final result |

### `TranscriptionResult`

Transcription output.

| Property | Type | Description |
|----------|------|-------------|
| `Text` | `string` | Transcribed text |
| `Language` | `string?` | Detected/used language |
| `Confidence` | `float?` | Confidence score |
| `Timestamps` | `List<Timestamp>?` | Word/segment timestamps |

## Requirements

- .NET 8.0 or later
- For GPU: CUDA 11.x+ (NVIDIA) or Metal support (macOS)

## Building from Source

### Prerequisites

- .NET SDK 8.0+
- Rust 1.75+ with cargo
- Target Rust toolchains:
  ```bash
  rustup target add x86_64-pc-windows-msvc
  rustup target add x86_64-unknown-linux-gnu
  rustup target add x86_64-apple-darwin
  rustup target add aarch64-apple-darwin
  ```

### Build

```bash
# Build everything (Rust FFI is built automatically via MSBuild)
dotnet build

# Or build a specific runtime package
dotnet build src/Qwen3ASR.NET.Runtime.Win-x64

# Skip Rust build (use existing native library)
dotnet build -p:SkipRustBuild=true

# Run tests
dotnet test

# Create NuGet packages
dotnet pack --configuration Release
```

The native Rust library (`qwen3_asr_ffi`) is automatically built when building the Runtime packages. MSBuild will:
1. Run `cargo build --release` before the .NET build
2. Copy the resulting native library to the `runtimes/<platform>/native/` folder
3. Include it in the NuGet package

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Qwen3ASR.NET (NuGet)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Qwen3Asr    │  │ Streaming   │  │ Models      │ │
│  │ (Main API)  │  │ Transcriber │  │ (DTOs)      │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘ │
│         │                │                          │
│  ┌──────▼────────────────▼──────┐                  │
│  │     NativeBindings (P/Invoke)│                  │
│  └──────────────┬───────────────┘                  │
└─────────────────┼───────────────────────────────────┘
                  │ FFI (cdylib)
┌─────────────────▼───────────────────────────────────┐
│            qwen3_asr_ffi (Rust cdylib)              │
│  ┌──────────────────────────────────────────────┐  │
│  │  C-compatible API layer                       │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │     qwen3_asr + candle-core (ML framework)   │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Dependencies

### Core Dependencies

| Library | Version | License | Description |
|---------|---------|---------|-------------|
| [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs) | main | MIT/Apache-2.0 | Qwen3-ASR Rust/Candle implementation |
| [candle-core](https://github.com/huggingface/candle) | 0.9 | MIT/Apache-2.0 | ML framework for Rust (Hugging Face) |
| [candle-nn](https://github.com/huggingface/candle) | 0.9 | MIT/Apache-2.0 | Neural network layers for Candle |
| [candle-transformers](https://github.com/huggingface/candle) | 0.9 | MIT/Apache-2.0 | Transformer models for Candle |

### Audio Processing

| Library | Version | License | Description |
|---------|---------|---------|-------------|
| [hound](https://github.com/ruuda/hound) | 3.5 | MIT/Apache-2.0 | WAV audio file reader/writer |

### FFI & Utilities

| Library | Version | License | Description |
|---------|---------|---------|-------------|
| [libc](https://github.com/rust-lang/libc) | 0.2 | MIT/Apache-2.0 | FFI bindings to system libraries |
| [thiserror](https://github.com/dtolnay/thiserror) | 1.0 | MIT/Apache-2.0 | Error handling derive macro |
| [anyhow](https://github.com/dtolnay/anyhow) | 1.0 | MIT/Apache-2.0 | Flexible error handling |
| [serde](https://github.com/serde-rs/serde) | 1.0 | MIT/Apache-2.0 | Serialization framework |
| [serde_json](https://github.com/serde-rs/json) | 1.0 | MIT/Apache-2.0 | JSON support for Serde |
| [log](https://github.com/rust-lang/log) | 0.4 | MIT/Apache-2.0 | Logging facade |
| [num_cpus](https://github.com/seanmonstar/num_cpus) | 1.16 | MIT/Apache-2.0 | CPU count detection |

### Transitive Dependencies (via qwen3-asr-rs)

| Library | License | Description |
|---------|---------|-------------|
| [tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | Fast tokenization library (Hugging Face) |
| [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | Safe tensor serialization |
| [hf-hub](https://github.com/huggingface/hf-hub) | Apache-2.0 | Hugging Face Hub API client |
| [ureq](https://github.com/algesten/ureq) | MIT/Apache-2.0 | HTTP client for model downloads |

## Acknowledgments

- [Qwen3-Audio](https://github.com/QwenLM/Qwen3-Audio) - The original Qwen3-ASR model by Alibaba
- [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs) - Rust/Candle implementation by lumosimmo
- [Candle](https://github.com/huggingface/candle) - ML framework for Rust by Hugging Face
- [Hugging Face Hub](https://huggingface.co/) - Model hosting and distribution
