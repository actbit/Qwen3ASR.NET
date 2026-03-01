using Qwen3ASR.NET.Enums;

namespace Qwen3ASR.NET.Models;

/// <summary>
/// Options for batch transcription.
/// </summary>
public class TranscriptionOptions
{
    /// <summary>
    /// Gets or sets the context string for the transcription.
    /// This can help improve accuracy by providing context.
    /// </summary>
    public string? Context { get; set; }

    /// <summary>
    /// Gets or sets the language for transcription.
    /// Default is Auto (auto-detect).
    /// </summary>
    public Language Language { get; set; } = Language.Auto;

    /// <summary>
    /// Gets or sets whether to return timestamps.
    /// Default is false.
    /// </summary>
    public bool ReturnTimestamps { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of new tokens to generate.
    /// 0 means use default.
    /// </summary>
    public int MaxNewTokens { get; set; }

    /// <summary>
    /// Gets or sets the maximum batch size for processing.
    /// Default is 32.
    /// </summary>
    public int MaxBatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the maximum chunk duration in seconds.
    /// null means use default.
    /// </summary>
    public float? ChunkMaxSec { get; set; }

    /// <summary>
    /// Gets or sets whether to bucket audio by length to reduce padding.
    /// Default is false.
    /// </summary>
    public bool BucketByLength { get; set; }

    /// <summary>
    /// Creates default transcription options.
    /// </summary>
    public static TranscriptionOptions Default => new();
}
