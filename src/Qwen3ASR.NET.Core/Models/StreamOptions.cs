namespace Qwen3ASR.NET.Models;

/// <summary>
/// Options for streaming transcription.
/// </summary>
public class StreamOptions
{
    /// <summary>
    /// Gets or sets the language code for transcription.
    /// Example: "Japanese", "English", "Chinese"
    /// If null, language will be auto-detected.
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Gets or sets the chunk size in seconds for streaming.
    /// Default is 0.5 seconds.
    /// </summary>
    public float ChunkSizeSec { get; set; } = 0.5f;

    /// <summary>
    /// Gets or sets whether to enable timestamp prediction.
    /// Default is true.
    /// </summary>
    public bool EnableTimestamps { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable partial results during streaming.
    /// Default is true.
    /// </summary>
    public bool EnablePartialResults { get; set; } = true;
}
