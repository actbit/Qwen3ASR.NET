# Parameters & Configuration

## TranscriptionOptions (Offline Transcription)

Configuration for batch transcription of files or audio samples.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Language` | `Language` | `Auto` | Transcription language. `Auto` for auto-detection |
| `Context` | `string?` | `null` | Context information for domain-specific terminology |
| `ReturnTimestamps` | `bool` | `false` | Return word-level timestamps |
| `MaxNewTokens` | `int` | `0` | Maximum tokens to generate. 0=default (auto) |
| `MaxBatchSize` | `int` | `32` | Maximum batch size. Lower if out of memory |
| `ChunkMaxSec` | `float?` | `null` | Maximum chunk duration for long audio |
| `BucketByLength` | `bool` | `false` | Bucket by length for processing efficiency |

### Language Enum

```
Auto, Japanese, English, Chinese, Korean, French, German, Spanish,
Russian, Portuguese, Italian, Dutch, Arabic, Hindi, Thai, Vietnamese,
Indonesian, Turkish, Polish, Swedish, Czech
```

### Usage Example

```csharp
var options = new TranscriptionOptions
{
    Language = Language.Japanese,
    Context = "Meeting minutes",        // Improves business term recognition
    ReturnTimestamps = true,            // Get timestamps
    MaxBatchSize = 16                   // Save memory
};
```

---

## StreamOptions (Streaming Transcription)

Configuration for real-time transcription.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Language` | `Language` | `Auto` | Transcription language |
| `Context` | `string?` | `null` | Initial context information |
| `ChunkSizeSec` | `float` | `2.0` | Processing chunk size in seconds. Smaller = lower latency but higher load |
| `UnfixedChunkNum` | `int` | `2` | Number of unfixed chunks. Larger = more stable but more delay |
| `UnfixedTokenNum` | `int` | `5` | Number of unfixed tokens for rollback buffer |
| `MaxNewTokens` | `int` | `256` | Maximum tokens to generate per step |
| `AudioWindowSec` | `float?` | `null` | Rolling audio window in seconds. For memory limiting |
| `TextWindowTokens` | `int?` | `null` | Rolling text window in tokens |

### Parameter Effects

#### ChunkSizeSec (Chunk Size)
- **Small (0.3-0.5s)**: Low latency, real-time focus, higher GPU load
- **Large (2.0-3.0s)**: Higher accuracy, lower GPU load, more delay

#### UnfixedChunkNum (Unfixed Chunks)
- **Small (1-2)**: Fast confirmation, may need corrections
- **Large (3-5)**: Stable results, more delay

#### MaxNewTokens (Max Tokens)
- **Small (64-128)**: Memory saving, for short utterances
- **Large (256-512)**: For long utterances, more memory

#### AudioWindowSec / TextWindowTokens (Rolling Window)
For memory limiting during long streaming sessions:
- `AudioWindowSec = 3.0`: Use only last 3 seconds of audio as context
- `TextWindowTokens = 20`: Keep only last 20 tokens of text

### Usage Examples

```csharp
// Low latency configuration (real-time focus)
var fastOptions = new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 0.3f,
    UnfixedChunkNum = 1,
    MaxNewTokens = 64
};

// High quality configuration (accuracy focus)
var qualityOptions = new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 2.0f,
    UnfixedChunkNum = 3,
    MaxNewTokens = 256
};

// Memory saving configuration (long recordings)
var memoryOptions = new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 0.5f,
    MaxNewTokens = 128,
    AudioWindowSec = 3.0f,       // Memory limit
    TextWindowTokens = 20        // Memory limit
};
```

---

## DeviceType (Device Selection)

| Value | Description | Recommended For |
|-------|-------------|-----------------|
| `Cpu` | CPU inference | Compatibility focus, no GPU |
| `Cuda` | NVIDIA GPU | RTX series, GTX series |
| `Metal` | Apple Silicon/AMD GPU | macOS (M1/M2/M3) |

### Performance Reference

| Device | Real-time Factor | Notes |
|--------|-----------------|-------|
| CPU (8 cores) | 0.1-0.3x | 10s audio takes 30-100s |
| RTX 2070 | 1-2x | Near real-time |
| RTX 3070 | 2-4x | Faster than real-time |
| M1 Pro | 0.5-1x | Semi real-time |

---

## Model Selection

| Model | Parameters | VRAM | Features |
|-------|------------|------|----------|
| `Qwen/Qwen3-ASR-0.6B` | 0.6B | ~1.5GB | Fast, sufficient accuracy |
| `Qwen/Qwen3-ASR-1.8B` | 1.8B | ~4GB | High accuracy, slower |
| `Qwen/Qwen3-ForcedAligner-0.6B` | - | ~0.5GB | For timestamps |

### Recommended Configurations

- **Real-time**: 0.6B + GPU
- **High accuracy offline**: 1.8B + GPU
- **Low memory environment**: 0.6B + CPU

---

## Error Handling

### CUDA_ERROR_OUT_OF_MEMORY

```csharp
// Solution 1: Use smaller model
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",  // Not 1.8B
    DeviceType.Cuda
);

// Solution 2: Limit memory in streaming
var options = new StreamOptions
{
    MaxNewTokens = 64,
    AudioWindowSec = 2.0f
};

// Solution 3: Fallback to CPU
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cpu
);
```

### Model Not Found

```csharp
// Verify model path
// HuggingFace ID: "Qwen/Qwen3-ASR-0.6B"
// Local path: "C:/models/Qwen3-ASR-0.6B"
```

### Audio Format Error

```csharp
// Supported: WAV (PCM), 16kHz recommended
// Convert with FFmpeg:
// ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```
