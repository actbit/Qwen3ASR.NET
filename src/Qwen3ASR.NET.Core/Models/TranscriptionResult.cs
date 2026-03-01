using System.Text.Json.Serialization;

namespace Qwen3ASR.NET.Models;

/// <summary>
/// Represents a timestamp for a word or segment in the transcription.
/// </summary>
public class Timestamp
{
    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    [JsonPropertyName("start")]
    public float Start { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    [JsonPropertyName("end")]
    public float End { get; set; }

    /// <summary>
    /// Gets or sets the word or segment text.
    /// </summary>
    [JsonPropertyName("text")]
    public string Text { get; set; } = string.Empty;
}

/// <summary>
/// Represents the result of a transcription.
/// </summary>
public class TranscriptionResult
{
    /// <summary>
    /// Gets or sets the transcribed text.
    /// </summary>
    [JsonPropertyName("text")]
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the detected or used language.
    /// </summary>
    [JsonPropertyName("language")]
    public string? Language { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (if available).
    /// </summary>
    [JsonPropertyName("confidence")]
    public float? Confidence { get; set; }

    /// <summary>
    /// Gets or sets the timestamps for words/segments.
    /// </summary>
    [JsonPropertyName("timestamps")]
    public List<Timestamp>? Timestamps { get; set; }

    /// <summary>
    /// Gets or sets whether this is a partial result (streaming).
    /// </summary>
    [JsonPropertyName("is_partial")]
    public bool IsPartial { get; set; }

    /// <summary>
    /// Returns the transcribed text.
    /// </summary>
    /// <returns>The transcribed text.</returns>
    public override string ToString() => Text;
}

/// <summary>
/// Detailed transcription result with additional metadata.
/// </summary>
public class TranscriptionDetail : TranscriptionResult
{
    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    [JsonPropertyName("processing_time_ms")]
    public long? ProcessingTimeMs { get; set; }
}
