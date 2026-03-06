# Qwen3ASR.NET

[![NuGet](https://img.shields.io/nuget/v/Qwen3ASR.NET.svg)](https://www.nuget.org/packages/Qwen3ASR.NET/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Qwen3ASR.NET** is a .NET wrapper for [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs), a Rust/Candle implementation of [Qwen3-Audio](https://github.com/QwenLM/Qwen3-Audio) (Alibaba's state-of-the-art speech recognition model).

**English** | [日本語](README.ja.md)

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

### Basic Setup

```csharp
using Qwen3ASR.NET;
using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;
```

### Model Loading

Models are automatically downloaded from HuggingFace on first use and cached locally:

```csharp
// Load from HuggingFace (cached after first download)
using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");

// Load from local path
using var asr = await Qwen3Asr.FromPretrainedAsync("/path/to/model");

// With forced aligner for word-level timestamps
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    forcedAlignerPath: "Qwen/Qwen3-ForcedAligner-0.6B",
    DeviceType.Cpu
);
```

**Available Models:**
| Model | Size | Description |
|-------|------|-------------|
| `Qwen/Qwen3-ASR-0.6B` | ~1.2GB | Fast, good accuracy (recommended) |
| `Qwen/Qwen3-ASR-1.8B` | ~3.5GB | Higher accuracy, slower |
| `Qwen/Qwen3-ForcedAligner-0.6B` | ~200MB | For word timestamps |

### Audio Requirements

- **Format**: WAV (PCM), or raw float32 samples
- **Sample Rate**: 16kHz (recommended), other rates are auto-converted
- **Channels**: Mono (stereo is automatically mixed down)
- **Bit Depth**: 16-bit or 32-bit float

```csharp
// WAV file (auto-converted if needed)
var result = await asr.TranscribeFileAsync("audio.wav");

// Raw float32 samples (must be 16kHz mono, normalized -1.0 to 1.0)
float[] samples = new float[16000]; // 1 second of audio
var result = await asr.TranscribeAsync(samples, 16000);

// WAV bytes (from HTTP upload, etc.)
byte[] wavBytes = GetWavBytesFromSomewhere();
var result = await asr.TranscribeWavBytesAsync(wavBytes);
```

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

### Word-Level Timestamps

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    forcedAlignerPath: "Qwen/Qwen3-ForcedAligner-0.6B"
);

var result = await asr.TranscribeFileAsync("audio.wav", new TranscriptionOptions
{
    Language = Language.Japanese,
    ReturnTimestamps = true
});

Console.WriteLine($"Text: {result.Text}");
if (result.Timestamps != null)
{
    foreach (var ts in result.Timestamps)
    {
        Console.WriteLine($"[{ts.Start:F2}s - {ts.End:F2}s] {ts.Text}");
    }
}
```

### Language Detection

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");

// Auto-detect language
var result = await asr.TranscribeFileAsync("audio.wav", new TranscriptionOptions
{
    Language = Language.Auto  // or omit (default is Auto)
});
Console.WriteLine($"Detected: {result.Language}");
Console.WriteLine($"Text: {result.Text}");

// Force specific language for better accuracy
var resultJp = await asr.TranscribeFileAsync("audio.wav", new TranscriptionOptions
{
    Language = Language.Japanese
});
```

### Context-Aware Transcription

Provide context for domain-specific terminology:

```csharp
var result = await asr.TranscribeFileAsync("medical_audio.wav", new TranscriptionOptions
{
    Language = Language.Japanese,
    Context = "医療・診療録"  // Helps recognize medical terms
});
```

### Error Handling

```csharp
using Qwen3ASR.NET;

try
{
    using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");
    var result = await asr.TranscribeFileAsync("audio.wav");
    Console.WriteLine(result.Text);
}
catch (Qwen3AsrException ex)
{
    Console.WriteLine($"ASR Error: {ex.Message}");
    // Common errors:
    // - Model not found
    // - Invalid audio format
    // - CUDA out of memory
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"File not found: {ex.FileName}");
}
```

### Performance Tips

```csharp
// 1. Use GPU when available
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cuda  // ~10x faster than CPU
);

// 2. Reuse model instance for multiple transcriptions
// DON'T: Load model for each file
// DO: Load once, transcribe many times
using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");
foreach (var file in files)
{
    var result = await asr.TranscribeFileAsync(file);
}

// 3. For streaming, use rolling windows to limit memory
await using var stream = asr.StartStream(new StreamOptions
{
    AudioWindowSec = 3.0f,
    TextWindowTokens = 20
});

// 4. Adjust batch size for batch processing
var results = await asr.TranscribeBatchAsync(files, new TranscriptionOptions
{
    MaxBatchSize = 16  // Lower if out of memory
});
```

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

### GPU Selection (Multi-GPU Systems)

For systems with multiple GPUs, you can select a specific GPU:

```csharp
// Set GPU index before loading model
Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "1"); // Use GPU 1

using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cuda
);
```

### Memory Optimization for Streaming

For long-running streaming sessions, use rolling windows to limit memory usage:

```csharp
await using var stream = asr.StartStream(new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 0.5f,
    UnfixedChunkNum = 2,
    UnfixedTokenNum = 5,
    MaxNewTokens = 128,          // Reduced for lower memory
    // Rolling windows - limit memory for long streams
    AudioWindowSec = 3.0f,       // Keep only last 3 seconds of audio context
    TextWindowTokens = 20        // Keep only last 20 tokens of text context
});
```

This prevents memory from growing indefinitely during long streaming sessions by trimming old audio samples and text tokens that are no longer needed for accurate transcription.

## Documentation

- **[Parameters & Configuration](docs/parameters.md)** / **[日本語](docs/parameters.ja.md)**
  - TranscriptionOptions and StreamOptions detailed explanation
  - Parameter effects and recommended configurations
  - Model selection guide
  - Error handling

- **[GPU Configuration Guide](docs/gpu-config.md)** / **[日本語](docs/gpu-config.ja.md)**
  - CUDA (NVIDIA GPU) setup and troubleshooting
  - Metal (macOS) setup
  - Multi-GPU configuration
  - Performance benchmarks

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
- Git (for submodule initialization)
- Target Rust toolchains:
  ```bash
  rustup target add x86_64-pc-windows-msvc
  rustup target add x86_64-unknown-linux-gnu
  rustup target add x86_64-apple-darwin
  rustup target add aarch64-apple-darwin
  ```
- For CUDA builds: CUDA Toolkit 11.x+ and Visual Studio with C++ workload

### Build

```bash
# Clone with submodules
git clone --recursive https://github.com/actbit/Qwen3ASR.NET.git

# Or initialize submodules after clone
git submodule update --init --recursive

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
1. Apply patches to submodules (CUDA fix, memory optimization)
2. Run `cargo build --release` before the .NET build
3. Copy the resulting native library to the `runtimes/<platform>/native/` folder
4. Include it in the NuGet package

### Patch System

This project uses git submodules with patch files for dependency management:

```
patches/
├── qwen3-asr-rs/              # Git submodule (qwen3-asr-rs repository)
├── candle/                    # Git submodule (huggingface/candle repository)
├── qwen3-asr-cuda.patch       # CUDA contiguous tensor fix
├── qwen3-asr-streaming-memory.patch  # Memory limit for streaming
└── candle-kernels-skip-moe.patch     # Skip MOE kernels for RTX 20 series
```

**Patches are applied automatically during build:**
1. `qwen3-asr-cuda.patch` - Fixes "matmul is only supported for contiguous tensors" error on CUDA
2. `qwen3-asr-streaming-memory.patch` - Limits memory usage in streaming mode with rolling windows
3. `candle-kernels-skip-moe.patch` - Skips MOE kernel compilation for broader GPU support (SM75+)

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

## Samples

This repository includes sample applications demonstrating various use cases:

### Qwen3ASR.NET.Realtime

Real-time transcription from microphone input with VAD (Voice Activity Detection).

```bash
# Run with CPU
dotnet run --project samples/Qwen3ASR.NET.Realtime -- --cpu

# Run with CUDA GPU
dotnet run --project samples/Qwen3ASR.NET.Realtime -- --gpu

# Run with specific GPU index
dotnet run --project samples/Qwen3ASR.NET.Realtime -- --gpu --gpu-index 1

# Specify language
dotnet run --project samples/Qwen3ASR.NET.Realtime -- -l Japanese

# Disable VAD (process all audio including silence)
dotnet run --project samples/Qwen3ASR.NET.Realtime -- --no-vad

# Use with forced aligner for word timestamps
dotnet run --project samples/Qwen3ASR.NET.Realtime -- -a Qwen/Qwen3-ForcedAligner-0.6B
```

**Command-line options:**
| Option | Description |
|--------|-------------|
| `-m, --model <path>` | Model path or HuggingFace ID |
| `-a, --aligner <path>` | Forced aligner for timestamps |
| `-l, --language <lang>` | Language code (Japanese, English, etc.) |
| `-d, --device <index>` | Audio input device index |
| `--cpu` | Force CPU inference |
| `--gpu, --cuda` | Use CUDA GPU |
| `--gpu-index <index>` | GPU device index (0, 1, etc.) |
| `--metal` | Use Metal GPU (macOS only) |
| `--no-vad` | Disable Voice Activity Detection |
| `--vad-threshold <val>` | VAD energy threshold (default: 0.01) |

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
