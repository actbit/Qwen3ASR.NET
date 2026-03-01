namespace Qwen3ASR.NET.Enums;

/// <summary>
/// Specifies the device type for model inference.
/// </summary>
public enum DeviceType
{
    /// <summary>
    /// CPU inference (most compatible, slower)
    /// </summary>
    Cpu = 0,

    /// <summary>
    /// CUDA GPU inference (requires NVIDIA GPU with CUDA support)
    /// </summary>
    Cuda = 1,

    /// <summary>
    /// Metal GPU inference (macOS only, requires Apple Silicon or AMD GPU)
    /// </summary>
    Metal = 2
}
