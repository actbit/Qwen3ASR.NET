using Qwen3ASR.NET.Enums;

namespace Qwen3ASR.NET.Models;

/// <summary>
/// Options for loading a Qwen3-ASR model.
/// </summary>
public class LoadOptions
{
    /// <summary>
    /// Gets or sets the device to use for inference.
    /// Default is CPU.
    /// </summary>
    public DeviceType Device { get; set; } = DeviceType.Cpu;

    /// <summary>
    /// Gets or sets the path to the model directory or HuggingFace model ID.
    /// Example: "Qwen/Qwen3-ASR-0.6B"
    /// </summary>
    public string ModelPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the specific revision/branch for HuggingFace models.
    /// Optional. If null, uses the default branch.
    /// </summary>
    public string? Revision { get; set; }

    /// <summary>
    /// Gets or sets the number of threads for CPU inference.
    /// Set to 0 for automatic detection (recommended).
    /// </summary>
    public int NumThreads { get; set; } = 0;
}
