# GPU Configuration Guide

## CUDA (NVIDIA GPU)

### Requirements
- NVIDIA GPU (Compute Capability 7.5+)
  - RTX 20 series (2060 or higher)
  - RTX 30 series
  - RTX 40 series
  - GTX 16 series (some limitations)
- CUDA Toolkit 11.x or later
- NVIDIA Driver 470.x or later

### Basic Usage

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cuda
);
```

### Multi-GPU Environment

To specify a particular GPU:

```csharp
// Use GPU 0 (default)
Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0");

// Use GPU 1
Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "1");

// Multiple GPUs (currently single GPU only)
Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0,1");
```

### VRAM Usage Reference

| Model | Minimum VRAM | Recommended VRAM |
|-------|-------------|------------------|
| 0.6B | 1.5GB | 2GB |
| 1.8B | 4GB | 6GB |

### Troubleshooting

#### CUDA not found Error

```
1. Check if NVIDIA Driver is installed
   nvidia-smi

2. Check if CUDA Toolkit is installed
   nvcc --version

3. Check if CUDA bin directory is in PATH
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
```

#### Out of Memory Error

```csharp
// Solution 1: Smaller chunk size
var options = new StreamOptions
{
    ChunkSizeSec = 0.5f,
    MaxNewTokens = 64
};

// Solution 2: Rolling window
var options = new StreamOptions
{
    AudioWindowSec = 2.0f,
    TextWindowTokens = 10
};

// Solution 3: Smaller model
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",  // Not 1.8B
    DeviceType.Cuda
);
```

#### Contiguous Tensor Error

This error is fixed by patches. Verify patches are applied during build:

```bash
# Patches are applied automatically during normal build
dotnet build
```

---

## Metal (macOS)

### Requirements
- Apple Silicon (M1/M2/M3) or AMD GPU
- macOS 12.0 (Monterey) or later

### Basic Usage

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Metal
);
```

### Performance Reference

| Device | 0.6B Model | 1.8B Model |
|--------|-----------|------------|
| M1 | 0.3-0.5x | Not recommended |
| M1 Pro | 0.5-1x | 0.2-0.3x |
| M1 Max | 1-1.5x | 0.3-0.5x |
| M2 Pro | 1-1.5x | 0.4-0.6x |

### Troubleshooting

#### Metal not available

```
1. Check macOS version (12.0+ required)
2. Verify Apple Silicon Mac (Intel Mac not supported)
3. Ensure not running under Rosetta
```

---

## CPU (Fallback)

Works on all environments but significantly slower than GPU.

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cpu
);
```

### Performance Reference

| CPU | 0.6B Model |
|-----|-----------|
| 4 cores | 0.05-0.1x |
| 8 cores | 0.1-0.2x |
| 16 cores | 0.2-0.3x |

Note: 0.1x = 10s audio takes 100s to process

### Optimization Tips

```csharp
// Parallelize with batch processing
var results = await asr.TranscribeBatchAsync(files, new TranscriptionOptions
{
    MaxBatchSize = 8  // Adjust based on core count
});
```
