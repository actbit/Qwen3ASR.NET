# Qwen3ASR.NET

[![NuGet](https://img.shields.io/nuget/v/Qwen3ASR.NET.svg)](https://www.nuget.org/packages/Qwen3ASR.NET/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md) | 日本語

**Qwen3ASR.NET** は [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs)（Alibabaの最新音声認識モデル [Qwen3-Audio](https://github.com/QwenLM/Qwen3-Audio) のRust/Candle実装）の.NETラッパーです。

## 特徴

- 🎯 **高精度** - Qwen3-ASRモデルベース
- 🌍 **多言語対応** - 日本語、英語、中国語など
- 📦 **簡単導入** - .NET 8.0+対応NuGetパッケージ
- 🔄 **ストリーミング対応** - リアルタイム文字起こし（部分結果付き）
- ⏱️ **タイムスタンプ予測** - 単語レベルの時間情報
- 🖥️ **クロスプラットフォーム** - Windows、Linux、macOS（x64/ARM64）

## インストール

### 全プラットフォーム（メタパッケージ）
```xml
<PackageReference Include="Qwen3ASR.NET" Version="1.0.0" />
```

### プラットフォーム個別（軽量デプロイ）
```xml
<PackageReference Include="Qwen3ASR.NET.Core" Version="1.0.0" />
<!-- プラットフォームを選択 -->
<PackageReference Include="Qwen3ASR.NET.Runtime.Win-x64" Version="1.0.0" />
```

利用可能なランタイムパッケージ:
- `Qwen3ASR.NET.Runtime.Win-x64` - Windows x64
- `Qwen3ASR.NET.Runtime.Linux-x64` - Linux x64
- `Qwen3ASR.NET.Runtime.OSX-x64` - macOS Intel
- `Qwen3ASR.NET.Runtime.OSX-arm64` - macOS Apple Silicon

## クイックスタート

### モデル読み込み

初回使用時にHuggingFaceから自動ダウンロード＆キャッシュ：

```csharp
using Qwen3ASR.NET;
using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;

// HuggingFaceから読み込み（初回のみダウンロード）
using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");

// ローカルパスから読み込み
using var asr = await Qwen3Asr.FromPretrainedAsync("/path/to/model");

// タイムスタンプ取得用のForced Aligner付き
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    forcedAlignerPath: "Qwen/Qwen3-ForcedAligner-0.6B",
    DeviceType.Cuda  // GPU使用
);
```

**利用可能なモデル:**
| モデル | サイズ | 説明 |
|-------|------|------|
| `Qwen/Qwen3-ASR-0.6B` | ~1.2GB | 高速・十分な精度（推奨） |
| `Qwen/Qwen3-ASR-1.8B` | ~3.5GB | より高精度・低速 |
| `Qwen/Qwen3-ForcedAligner-0.6B` | ~200MB | タイムスタンプ用 |

### 音声フォーマット要件

- **形式**: WAV（PCM）または生のfloat32サンプル
- **サンプリングレート**: 16kHz推奨（他は自動変換）
- **チャンネル**: モノラル（ステレオは自動ミックスダウン）
- **ビット深度**: 16ビットまたは32ビットfloat

### オフライン文字起こし

```csharp
using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cuda
);

// ファイルから文字起こし
var result = await asr.TranscribeFileAsync("audio.wav", new TranscriptionOptions
{
    Language = Language.Japanese
});
Console.WriteLine(result.Text);

// 生の音声サンプルから（16kHzモノラル、-1.0〜1.0正規化必須）
float[] samples = LoadAudioSamples();
var result2 = await asr.TranscribeAsync(samples, 16000, new TranscriptionOptions
{
    Language = Language.Japanese
});
```

### ストリーミング文字起こし

```csharp
using var asr = await Qwen3ASR.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");

// ストリーミングセッション開始
await using var stream = asr.StartStream(new StreamOptions
{
    Language = Language.Japanese,
    ChunkSizeSec = 0.5f,        // 処理単位（小さいほど低遅延）
    UnfixedChunkNum = 2,
    MaxNewTokens = 128,
    AudioWindowSec = 3.0f,      // メモリ制限
    TextWindowTokens = 20
});

// 音声チャンクをプッシュ
foreach (var chunk in audioChunks)
{
    var partial = await stream.PushAsync(chunk);
    Console.WriteLine($"部分結果: {partial.Text}");
}

// 最終結果
var final = await stream.FinishAsync();
Console.WriteLine($"最終結果: {final.Text}");
```

### 単語レベルタイムスタンプ

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

if (result.Timestamps != null)
{
    foreach (var ts in result.Timestamps)
    {
        Console.WriteLine($"[{ts.Start:F2}s - {ts.End:F2}s] {ts.Text}");
    }
}
```

### 文脈を考慮した文字起こし

専門用語の認識精度向上：

```csharp
var result = await asr.TranscribeFileAsync("medical_audio.wav", new TranscriptionOptions
{
    Language = Language.Japanese,
    Context = "医療・診療録"  // 医療用語の認識向上
});
```

### GPU選択（マルチGPU環境）

```csharp
// GPU 1を使用
Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "1");

using var asr = await Qwen3Asr.FromPretrainedAsync(
    "Qwen/Qwen3-ASR-0.6B",
    DeviceType.Cuda
);
```

### エラーハンドリング

```csharp
try
{
    using var asr = await Qwen3Asr.FromPretrainedAsync("Qwen/Qwen3-ASR-0.6B");
    var result = await asr.TranscribeFileAsync("audio.wav");
}
catch (Qwen3AsrException ex)
{
    Console.WriteLine($"ASRエラー: {ex.Message}");
    // よくあるエラー:
    // - モデルが見つからない
    // - 音声フォーマット無効
    // - CUDA メモリ不足
}
```

## ドキュメント

- **[パラメータ・設定値説明](docs/parameters.ja.md)** / **[English](docs/parameters.md)**
  - TranscriptionOptions / StreamOptions 詳細説明
  - パラメータの効果と推奨設定
  - モデル選択ガイド

- **[GPU設定ガイド](docs/gpu-config.ja.md)** / **[English](docs/gpu-config.md)**
  - CUDA (NVIDIA GPU) セットアップとトラブルシューティング
  - Metal (macOS) セットアップ
  - パフォーマンス参考値

## APIリファレンス

### `Qwen3Asr` メインクラス

| メソッド | 説明 |
|--------|------|
| `FromPretrainedAsync(modelPath, device)` | 学習済みモデルを読み込み |
| `TranscribeFileAsync(filePath, options)` | 音声ファイルを文字起こし |
| `TranscribeAsync(samples, sampleRate, options)` | 音声サンプルを文字起こし |
| `TranscribeWavBytesAsync(wavBytes, options)` | WAVバイト列を文字起こし |
| `TranscribeFileStreamAsync(...)` | ファイルをストリーミング処理 |
| `TranscribeBatchAsync(filePaths, options)` | 複数ファイルを一括処理 |
| `StartStream(options)` | ストリーミング文字起こし開始 |

### `StreamOptions` ストリーミング設定

| プロパティ | 型 | デフォルト | 説明 |
|----------|------|---------|------|
| `Language` | `Language` | `Auto` | 文字起こし言語 |
| `ChunkSizeSec` | `float` | `2.0` | 処理単位の秒数 |
| `UnfixedChunkNum` | `int` | `2` | 未確定チャンク数 |
| `MaxNewTokens` | `int` | `256` | 1回の生成トークン数 |
| `AudioWindowSec` | `float?` | `null` | ローリングウィンドウ（メモリ制限用） |
| `TextWindowTokens` | `int?` | `null` | テキストウィンドウ |

### `DeviceType` デバイス選択

- `Cpu` - CPU推論（互換性重視）
- `Cuda` - NVIDIA GPU (CUDA)
- `Metal` - macOS Apple Silicon/AMD

## 要件

- .NET 8.0以降
- GPU使用時: CUDA 11.x+ (NVIDIA) または Metal対応 (macOS)

## サンプルアプリ

### Qwen3ASR.NET.Realtime

マイク入力からのリアルタイム文字起こし（VAD搭載）。

```bash
# CPUで実行
dotnet run --project samples/Qwen3ASR.NET.Realtime -- --cpu

# CUDA GPUで実行
dotnet run --project samples/Qwen3ASR.NET.Realtime -- --gpu

# 特定のGPUを指定
dotnet run --project samples/Qwen3ASR.NET.Realtime -- --gpu --gpu-index 1

# 言語指定
dotnet run --project samples/Qwen3ASR.NET.Realtime -- -l Japanese

# タイムスタンプ取得
dotnet run --project samples/Qwen3ASR.NET.Realtime -- -a Qwen/Qwen3-ForcedAligner-0.6B
```

**コマンドラインオプション:**
| オプション | 説明 |
|--------|------|
| `-m, --model <path>` | モデルパスまたはHuggingFace ID |
| `-a, --aligner <path>` | タイムスタンプ用Forced Aligner |
| `-l, --language <lang>` | 言語コード (Japanese, English等) |
| `-d, --device <index>` | オーディオ入力デバイス番号 |
| `--cpu` | CPU推論を強制 |
| `--gpu, --cuda` | CUDA GPUを使用 |
| `--gpu-index <index>` | GPU番号 (0, 1等) |
| `--metal` | Metal GPUを使用 (macOSのみ) |
| `--no-vad` | VADを無効化 |
| `--vad-threshold <val>` | VAD閾値 (デフォルト: 0.01) |

## ソースからビルド

```bash
# サブモジュール付きでクローン
git clone --recursive https://github.com/actbit/Qwen3ASR.NET.git

# ビルド（Rust FFIは自動ビルド）
dotnet build

# NuGetパッケージ作成
dotnet pack --configuration Release
```

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照。

## 謝辞

- [Qwen3-Audio](https://github.com/QwenLM/Qwen3-Audio) - AlibabaによるオリジナルQwen3-ASRモデル
- [qwen3-asr-rs](https://github.com/lumosimmo/qwen3-asr-rs) - lumosimmoによるRust/Candle実装
- [Candle](https://github.com/huggingface/candle) - Hugging FaceによるRust用MLフレームワーク
