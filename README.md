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
# Build .NET library
dotnet build

# Build Rust FFI library
cd qwen3_asr_ffi
cargo build --release

# Run tests
dotnet test

# Create NuGet packages
dotnet pack --configuration Release
```

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

## Acknowledgments

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-Audio) - The original Qwen3-ASR model
- [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs) - Rust implementation
- [Candle](https://github.com/huggingface/candle) - ML framework for Rust
