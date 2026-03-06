# GPU設定ガイド

## CUDA（NVIDIA GPU）

### 要件
- NVIDIA GPU (Compute Capability 7.5+)
  - RTX 20シリーズ (2060以上)
  - RTX 30シリーズ
  - RTX 40シリーズ
  - GTX 16シリーズ (一部制限あり)
- CUDA Toolkit 11.x 以降
- NVIDIA Driver 470.x 以降

### 基本的な使い方

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cuda
);
```

### マルチGPU環境

特定のGPUを指定する場合：

```csharp
// GPU 0を使用（デフォルト）
Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0");

// GPU 1を使用
Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "1");

// 複数GPU（現在は単一GPUのみ対応）
Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0,1");
```

### VRAM使用量の目安

| モデル | 最小VRAM | 推奨VRAM |
|--------|---------|---------|
| 0.6B | 1.5GB | 2GB |
| 1.8B | 4GB | 6GB |

### トラブルシューティング

#### CUDA not found エラー

```
1. NVIDIA Driverがインストールされているか確認
   nvidia-smi

2. CUDA Toolkitがインストールされているか確認
   nvcc --version

3. PATHにCUDA binディレクトリが含まれているか確認
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
```

#### Out of Memory エラー

```csharp
// 解決策1: 小さいチャンクサイズ
var options = new StreamOptions
{
    ChunkSizeSec = 0.5f,
    MaxNewTokens = 64
};

// 解決策2: ローリングウィンドウ
var options = new StreamOptions
{
    AudioWindowSec = 2.0f,
    TextWindowTokens = 10
};

// 解決策3: 小さいモデル
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",  // 1.8Bではなく
    DeviceType.Cuda
);
```

#### Contiguous Tensor エラー

このエラーはパッチで修正済みです。ビルド時にパッチが適用されているか確認：

```bash
# 正常にビルドされていれば自動的に修正されます
dotnet build
```

---

## Metal（macOS）

### 要件
- Apple Silicon (M1/M2/M3) または AMD GPU
- macOS 12.0 (Monterey) 以降

### 基本的な使い方

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Metal
);
```

### パフォーマンスの目安

| デバイス | 0.6Bモデル | 1.8Bモデル |
|---------|-----------|-----------|
| M1 | 0.3-0.5x | 非推奨 |
| M1 Pro | 0.5-1x | 0.2-0.3x |
| M1 Max | 1-1.5x | 0.3-0.5x |
| M2 Pro | 1-1.5x | 0.4-0.6x |

### トラブルシューティング

#### Metal not available

```
1. macOSのバージョンを確認（12.0以降必要）
2. Apple Silicon Macか確認（Intel Macは非対応）
3. Rosettaを使用せずに実行しているか確認
```

---

## CPU（フォールバック）

すべての環境で動作しますが、GPUより大幅に遅くなります。

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cpu
);
```

### パフォーマンスの目安

| CPU | 0.6Bモデル |
|-----|-----------|
| 4コア | 0.05-0.1x |
| 8コア | 0.1-0.2x |
| 16コア | 0.2-0.3x |

※ 0.1x = 10秒の音声に100秒かかる

### 最適化のヒント

```csharp
// バッチ処理で並列化
var results = await asr.TranscribeBatchAsync(files, new TranscriptionOptions
{
    MaxBatchSize = 8  // コア数に合わせて調整
});
```
