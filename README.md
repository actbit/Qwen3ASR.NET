# Qwen3ASR.NET

[![NuGet](https://img.shields.io/nuget/v/Qwen3ASR.NET.svg)](https://www.nuget.org/packages/Qwen3ASR.NET/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Qwen3ASR.NET** is a .NET wrapper for [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs), a Rust/Candle implementation of [Qwen3-Audio](https://github.com/QwenLM/Qwen3-Audio) (Alibaba's state-of-the-art speech recognition model).

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
using Qwen3ASR.NET.Models;

// Load model (first run will download from HuggingFace)
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cpu
);

// Transcribe audio file
var result = await asr.TranscribeFileAsync("audio.wav", new TranscriptionOptions
{
    Language = Language.Japanese
});
Console.WriteLine(result.Text);

// Transcribe raw audio samples (16kHz mono f32)
float[] samples = LoadAudioSamples(); // Your audio loading code
var result2 = await asr.TranscribeAsync(samples, 16000, new TranscriptionOptions
{
    Language = Language.Japanese
});
Console.WriteLine(result2.Text);

// Transcribe from WAV bytes
byte[] wavBytes = File.ReadAllBytes("audio.wav");
var result3 = await asr.TranscribeWavBytesAsync(wavBytes);
Console.WriteLine(result3.Text);
```

### Streaming Transcription

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");

// Start streaming session
await using var stream = asr.StartStream(new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 2.0f,
    UnfixedChunkNum = 2,
    MaxNewTokens = 256
});

// Push audio chunks as they arrive
foreach (var chunk in audioChunks)
{
    var partial = await stream.PushAsync(chunk);
    Console.WriteLine($"Partial: {partial.Text}");
}

// Get final result with timestamps
var final = await stream.FinishAsync();
Console.WriteLine($"Final: {final.Text}");
```

### Streaming from File

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");

// Stream process a large file with partial results
var result = await asr.TranscribeFileStreamAsync(
    "large_audio.wav",
    onPartialResult: partial => Console.WriteLine($"Partial: {partial.Text}"),
    new StreamOptions { Language = Language.Japanese }
);

Console.WriteLine($"Final: {result.Text}");
```

### Batch Processing

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");

// Process multiple files
var files = new[] { "audio1.wav", "audio2.wav", "audio3.wav" };
var results = await asr.TranscribeBatchAsync(files, new TranscriptionOptions
{
    Language = Language.Japanese,
    ReturnTimestamps = true
});

foreach (var result in results)
{
    Console.WriteLine(result.Text);
}
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
| `TranscribeFileAsync(filePath, options)` | Transcribe an audio file |
| `TranscribeAsync(samples, sampleRate, options)` | Transcribe audio samples |
| `TranscribeAsync(stream, options)` | Transcribe from WAV stream |
| `TranscribeWavBytesAsync(wavBytes, options)` | Transcribe from WAV bytes |
| `TranscribeFileStreamAsync(filePath, onPartialResult, options)` | Stream process a file |
| `TranscribeSamplesStreamAsync(samples, onPartialResult, options)` | Stream process samples |
| `TranscribeBatchAsync(filePaths, options)` | Process multiple files |
| `StartStream(options)` | Start streaming transcription |
| `GetSupportedLanguagesAsync()` | Get supported languages |
| `GetVersion()` | Get library version |

### `StreamingTranscriber`

Real-time transcription session with rolling context support.

| Method | Description |
|--------|-------------|
| `PushAsync(samples)` | Push audio samples (16kHz mono f32) |
| `GetPartialResult()` | Get current partial result |
| `FinishAsync()` | Finish and get final result |

| Property | Type | Description |
|----------|------|-------------|
| `IsActive` | `bool` | Whether stream is still active |
| `SegmentCount` | `int` | Number of segments processed |
| `Duration` | `float` | Total duration processed (seconds) |

### `TranscriptionOptions`

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `Language` | `Language` | `Auto` | Language for transcription |
| `Context` | `string?` | `null` | Context string for accuracy |
| `ReturnTimestamps` | `bool` | `false` | Return word timestamps |
| `MaxNewTokens` | `int` | `0` | Max tokens (0 = default) |
| `MaxBatchSize` | `int` | `32` | Batch size for processing |
| `ChunkMaxSec` | `float?` | `null` | Max chunk duration |
| `BucketByLength` | `bool` | `false` | Bucket audio by length |

### `StreamOptions`

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `Language` | `Language` | `Auto` | Language for transcription |
| `Context` | `string?` | `null` | Initial context string |
| `ChunkSizeSec` | `float` | `2.0` | Chunk size in seconds |
| `UnfixedChunkNum` | `int` | `2` | Unfixed chunk count |
| `UnfixedTokenNum` | `int` | `5` | Unfixed token count |
| `MaxNewTokens` | `int` | `256` | Max new tokens |
| `AudioWindowSec` | `float?` | `null` | Rolling audio window |
| `TextWindowTokens` | `int?` | `null` | Rolling text window |

### `TranscriptionResult`

| Property | Type | Description |
|----------|------|-------------|
| `Text` | `string` | Transcribed text |
| `Language` | `string?` | Detected/used language |
| `Confidence` | `float?` | Confidence score |
| `Timestamps` | `List<Timestamp>?` | Word/segment timestamps |
| `IsPartial` | `bool` | Whether this is a partial result |

### `Language` Enum

Supported languages: `Auto`, `Japanese`, `English`, `Chinese`, `Korean`, `French`, `German`, `Spanish`, `Russian`, `Portuguese`, `Italian`, `Dutch`, `Arabic`, `Hindi`, `Thai`, `Vietnamese`, `Indonesian`, `Turkish`, `Polish`, `Swedish`, `Czech`

### `DeviceType` Enum

- `Cpu` - CPU inference (most compatible)
- `Cuda` - NVIDIA GPU with CUDA
- `Metal` - macOS Apple Silicon/AMD

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

Licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Dependencies

| Library | Version | License | Description |
|---------|---------|---------|-------------|
| [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs) | main | MIT/Apache-2.0 | Qwen3-ASR Rust/Candle implementation |
| [candle-core](https://github.com/huggingface/candle) | 0.9 | MIT/Apache-2.0 | ML framework for Rust (Hugging Face) |
| [hound](https://github.com/ruuda/hound) | 3.5 | MIT/Apache-2.0 | WAV audio file reader/writer |
| [serde](https://github.com/serde-rs/serde) | 1.0 | MIT/Apache-2.0 | Serialization framework |
| [serde_json](https://github.com/serde-rs/json) | 1.0 | MIT/Apache-2.0 | JSON support for Serde |
| [anyhow](https://github.com/dtolnay/anyhow) | 1.0 | MIT/Apache-2.0 | Flexible error handling |
| [thiserror](https://github.com/dtolnay/thiserror) | 1.0 | MIT/Apache-2.0 | Error handling derive macro |
| [libc](https://github.com/rust-lang/libc) | 0.2 | MIT/Apache-2.0 | FFI bindings to system libraries |
| [log](https://github.com/rust-lang/log) | 0.4 | MIT/Apache-2.0 | Logging facade |
| [num_cpus](https://github.com/seanmonstar/num_cpus) | 1.16 | MIT/Apache-2.0 | CPU count detection |

## Acknowledgments

- [Qwen3-Audio](https://github.com/QwenLM/Qwen3-Audio) - The original Qwen3-ASR model by Alibaba
- [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs) - Rust/Candle implementation by lumosimmo
- [Candle](https://github.com/huggingface/candle) - ML framework for Rust by Hugging Face
- [Hugging Face Hub](https://huggingface.co/) - Model hosting and distribution
