namespace Qwen3ASR.NET.Models;

/// <summary>
/// Specifies the format of raw audio data.
/// </summary>
public class AudioFormat
{
    /// <summary>
    /// Gets or sets the sample rate in Hz (e.g., 16000, 44100, 48000).
    /// </summary>
    public uint SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the number of channels (1 = mono, 2 = stereo).
    /// </summary>
    public ushort Channels { get; set; } = 1;

    /// <summary>
    /// Gets or sets the bits per sample (8, 16, 24, 32).
    /// </summary>
    public ushort BitsPerSample { get; set; } = 16;

    /// <summary>
    /// Gets or sets whether the audio is in floating-point format.
    /// </summary>
    public bool IsFloat { get; set; } = false;

    /// <summary>
    /// Creates a new AudioFormat instance.
    /// </summary>
    public AudioFormat() { }

    /// <summary>
    /// Creates a new AudioFormat instance with specified parameters.
    /// </summary>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="bitsPerSample">Bits per sample.</param>
    /// <param name="isFloat">Whether the audio is floating-point.</param>
    public AudioFormat(uint sampleRate, ushort channels = 1, ushort bitsPerSample = 16, bool isFloat = false)
    {
        SampleRate = sampleRate;
        Channels = channels;
        BitsPerSample = bitsPerSample;
        IsFloat = isFloat;
    }

    /// <summary>
    /// Gets the standard 16kHz mono 16-bit PCM format.
    /// </summary>
    public static AudioFormat Pcm16kMono => new(16000, 1, 16, false);

    /// <summary>
    /// Gets the standard 44.1kHz stereo 16-bit PCM format (CD quality).
    /// </summary>
    public static AudioFormat CdQuality => new(44100, 2, 16, false);

    /// <summary>
    /// Gets the standard 48kHz stereo 16-bit PCM format (DVD quality).
    /// </summary>
    public static AudioFormat DvdQuality => new(48000, 2, 16, false);

    /// <summary>
    /// Gets the standard 48kHz mono 32-bit float format.
    /// </summary>
    public static AudioFormat Float48kMono => new(48000, 1, 32, true);

    /// <summary>
    /// Gets the bytes per sample (including all channels).
    /// </summary>
    public int BytesPerSample => (BitsPerSample / 8) * Channels;

    /// <summary>
    /// Gets the bytes per second.
    /// </summary>
    public int BytesPerSecond => BytesPerSample * (int)SampleRate;

    /// <summary>
    /// Calculates the duration in seconds for the given byte count.
    /// </summary>
    /// <param name="byteCount">Number of bytes.</param>
    /// <returns>Duration in seconds.</returns>
    public double GetDurationSeconds(long byteCount)
    {
        return (double)byteCount / BytesPerSecond;
    }

    /// <summary>
    /// Calculates the byte count for the given duration.
    /// </summary>
    /// <param name="seconds">Duration in seconds.</param>
    /// <returns>Number of bytes.</returns>
    public long GetByteCount(double seconds)
    {
        return (long)(seconds * BytesPerSecond);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var formatType = IsFloat ? "float" : "PCM";
        var channels = Channels == 1 ? "mono" : Channels == 2 ? "stereo" : $"{Channels}ch";
        return $"{SampleRate}Hz {channels} {BitsPerSample}bit {formatType}";
    }
}
