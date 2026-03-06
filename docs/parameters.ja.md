# パラメータ・設定値説明

## TranscriptionOptions（オフライン文字起こし）

ファイルや音声サンプルを一括で文字起こしする際の設定です。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `Language` | `Language` | `Auto` | 文字起こしの言語。`Auto`で自動検出 |
| `Context` | `string?` | `null` | 文脈情報。専門用語の認識精度向上に使用 |
| `ReturnTimestamps` | `bool` | `false` | 単語レベルのタイムスタンプを返すか |
| `MaxNewTokens` | `int` | `0` | 生成する最大トークン数。0=デフォルト（自動） |
| `MaxBatchSize` | `int` | `32` | バッチ処理時の最大サイズ。メモリ不足の場合は下げる |
| `ChunkMaxSec` | `float?` | `null` | 長い音声を分割する際の最大チャンク秒数 |
| `BucketByLength` | `bool` | `false` | 長さでバケッティングして処理効率化 |

### Language 列挙型

```
Auto, Japanese, English, Chinese, Korean, French, German, Spanish,
Russian, Portuguese, Italian, Dutch, Arabic, Hindi, Thai, Vietnamese,
Indonesian, Turkish, Polish, Swedish, Czech
```

### 使用例

```csharp
var options = new TranscriptionOptions
{
    Language = Language.Japanese,
    Context = "会議の議事録",           // ビジネス用語の認識向上
    ReturnTimestamps = true,           // タイムスタンプ取得
    MaxBatchSize = 16                  // メモリ節約
};
```

---

## StreamOptions（ストリーミング文字起こし）

リアルタイム文字起こしの設定です。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `Language` | `Language` | `Auto` | 文字起こしの言語 |
| `Context` | `string?` | `null` | 初期文脈情報 |
| `ChunkSizeSec` | `float` | `2.0` | 処理単位の秒数。小さいほど低遅延だが負荷増 |
| `UnfixedChunkNum` | `int` | `2` | 確定前のチャンク数。大きいほど安定するが遅延増 |
| `UnfixedTokenNum` | `int` | `5` | 確定前のトークン数。ロールバック用バッファ |
| `MaxNewTokens` | `int` | `256` | 1回の生成で作成する最大トークン数 |
| `AudioWindowSec` | `float?` | `null` | ローリングウィンドウの音声秒数。メモリ制限用 |
| `TextWindowTokens` | `int?` | `null` | ローリングウィンドウのテキストトークン数 |

### パラメータの影響

#### ChunkSizeSec（チャンクサイズ）
- **小さい値（0.3-0.5秒）**: 低遅延、リアルタイム性重視、GPU負荷増
- **大きい値（2.0-3.0秒）**: 高精度、GPU負荷低減、遅延増

#### UnfixedChunkNum（未確定チャンク数）
- **小さい値（1-2）**: 早い確定、修正の可能性あり
- **大きい値（3-5）**: 安定した結果、遅延増

#### MaxNewTokens（最大トークン数）
- **小さい値（64-128）**: メモリ節約、短い発向き
- **大きい値（256-512）**: 長い発言対応、メモリ増

#### AudioWindowSec / TextWindowTokens（ローリングウィンドウ）
長時間ストリーミング時のメモリ制限用：
- `AudioWindowSec = 3.0`: 直近3秒の音声のみコンテキストに使用
- `TextWindowTokens = 20`: 直近20トークンのテキストのみ保持

### 使用例

```csharp
// 低遅延設定（リアルタイム重視）
var fastOptions = new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 0.3f,
    UnfixedChunkNum = 1,
    MaxNewTokens = 64
};

// 高精度設定（品質重視）
var qualityOptions = new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 2.0f,
    UnfixedChunkNum = 3,
    MaxNewTokens = 256
};

// メモリ節約設定（長時間録音用）
var memoryOptions = new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 0.5f,
    MaxNewTokens = 128,
    AudioWindowSec = 3.0f,       // メモリ制限
    TextWindowTokens = 20        // メモリ制限
};
```

---

## DeviceType（デバイス選択）

| 値 | 説明 | 推奨環境 |
|----|------|---------|
| `Cpu` | CPU推論 | 互換性重視、GPUなし |
| `Cuda` | NVIDIA GPU | RTXシリーズ、GTXシリーズ |
| `Metal` | Apple Silicon/AMD GPU | macOS (M1/M2/M3) |

### パフォーマンス目安

| デバイス | リアルタイム倍率 | 備考 |
|---------|-----------------|------|
| CPU (8コア) | 0.1-0.3x | 10秒の音声に30-100秒 |
| RTX 2070 | 1-2x | ほぼリアルタイム |
| RTX 3070 | 2-4x | リアルタイム以上 |
| M1 Pro | 0.5-1x | 準リアルタイム |

---

## モデル選択

| モデル | パラメータ数 | VRAM | 特徴 |
|--------|------------|------|------|
| `Qwen/Qwen3-ASR-0.6B` | 0.6B | ~1.5GB | 高速、十分な精度 |
| `Qwen/Qwen3-ASR-1.8B` | 1.8B | ~4GB | 高精度、低速 |
| `Qwen/Qwen3-ForcedAligner-0.6B` | - | ~0.5GB | タイムスタンプ用 |

### 推奨構成

- **リアルタイム**: 0.6B + GPU
- **高精度オフライン**: 1.8B + GPU
- **低メモリ環境**: 0.6B + CPU

---

## エラー対処

### CUDA_ERROR_OUT_OF_MEMORY

```csharp
// 解決策1: 小さいモデルを使用
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",  // 1.8Bではなく
    DeviceType.Cuda
);

// 解決策2: ストリーミングでメモリ制限
var options = new StreamOptions
{
    MaxNewTokens = 64,
    AudioWindowSec = 2.0f
};

// 解決策3: CPUにフォールバック
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cpu
);
```

### Model Not Found

```csharp
// モデルパスを確認
// HuggingFace ID: "Qwen/Qwen3-ASR-0.6B"
// ローカルパス: "C:/models/Qwen3-ASR-0.6B"
```

### Audio Format Error

```csharp
// 対応形式: WAV (PCM), 16kHz推奨
// FFmpegで変換:
// ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```
