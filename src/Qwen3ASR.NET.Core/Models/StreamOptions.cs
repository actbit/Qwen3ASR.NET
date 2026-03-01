using Qwen3ASR.NET.Enums;

namespace Qwen3ASR.NET.Models;

/// <summary>
/// Options for streaming transcription.
/// </summary>
public class StreamOptions
{
    /// <summary>
    /// Gets or sets the context string for the transcription.
    /// </summary>
    public string? Context { get; set; }

    /// <summary>
    /// Gets or sets the language for transcription.
    /// Default is Auto (auto-detect).
    /// </summary>
    public Language Language { get; set; } = Language.Auto;

    /// <summary>
    /// Gets or sets the chunk size in seconds for streaming.
    /// Default is 2.0 seconds.
    /// </summary>
    public float ChunkSizeSec { get; set; } = 2.0f;

    /// <summary>
    /// Gets or sets the number of unfixed chunks.
    /// Default is 2.
    /// </summary>
    public int UnfixedChunkNum { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of unfixed tokens.
    /// Default is 5.
    /// </summary>
    public int UnfixedTokenNum { get; set; } = 5;

    /// <summary>
    /// Gets or sets the maximum number of new tokens to generate.
    /// Default is 256.
    /// </summary>
    public int MaxNewTokens { get; set; } = 256;

    /// <summary>
    /// Gets or sets the audio window in seconds for rolling context.
    /// null means no window (use all audio).
    /// </summary>
    public float? AudioWindowSec { get; set; }

    /// <summary>
    /// Gets or sets the text window in tokens for rolling context.
    /// null means no window (use all text).
    /// </summary>
    public int? TextWindowTokens { get; set; }

    /// <summary>
    /// Creates default stream options.
    /// </summary>
    public static StreamOptions Default => new();
}
