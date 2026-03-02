using NWaves.Audio;
using NWaves.Operations;
using NWaves.Signals;

namespace Qwen3ASR.NET.Audio;

/// <summary>
/// Cross-platform audio file reader using NWaves.
/// Supports WAV, AIFF, and other formats with high-quality resampling.
/// </summary>
internal static class WavReader
{
    /// <summary>
    /// Target sample rate for ASR (16kHz).
    /// </summary>
    private const int TargetSampleRate = 16000;

    /// <summary>
    /// Reads an audio file and converts it to 16kHz mono f32 samples.
    /// </summary>
    /// <param name="filePath">Path to the audio file.</param>
    /// <returns>Audio data as f32 samples at 16kHz mono.</returns>
    public static WavData ReadWav(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        return ReadWav(stream);
    }

    /// <summary>
    /// Reads audio data from a stream and converts it to 16kHz mono f32 samples.
    /// </summary>
    /// <param name="stream">Stream containing audio data.</param>
    /// <returns>Audio data as f32 samples at 16kHz mono.</returns>
    public static WavData ReadWav(Stream stream)
    {
        WaveFile waveFile;
        using (var readerStream = new NonDisposableStream(stream))
        {
            waveFile = new WaveFile(readerStream);
        }

        // Get the first channel (mono) or left channel (stereo)
        DiscreteSignal signal = waveFile[Channels.Left];

        int originalSampleRate = signal.SamplingRate;

        // Resample to 16kHz if needed using NWaves' bandlimited resampling
        if (originalSampleRate != TargetSampleRate)
        {
            signal = Operation.Resample(signal, TargetSampleRate);
        }

        return new WavData(signal.Samples, TargetSampleRate);
    }

    /// <summary>
    /// Reads audio bytes and converts to 16kHz mono f32 samples.
    /// </summary>
    public static WavData ReadWav(byte[] audioBytes)
    {
        using var stream = new MemoryStream(audioBytes);
        return ReadWav(stream);
    }

    /// <summary>
    /// Reads audio from a file path with explicit format specification.
    /// </summary>
    /// <param name="filePath">Path to the audio file.</param>
    /// <param name="targetSampleRate">Target sample rate (default: 16000).</param>
    /// <returns>Audio data as f32 samples.</returns>
    public static WavData ReadWav(string filePath, int targetSampleRate)
    {
        using var stream = File.OpenRead(filePath);
        return ReadWav(stream, targetSampleRate);
    }

    /// <summary>
    /// Reads audio data from a stream with custom target sample rate.
    /// </summary>
    public static WavData ReadWav(Stream stream, int targetSampleRate)
    {
        WaveFile waveFile;
        using (var readerStream = new NonDisposableStream(stream))
        {
            waveFile = new WaveFile(readerStream);
        }

        DiscreteSignal signal = waveFile[Channels.Left];

        if (signal.SamplingRate != targetSampleRate)
        {
            signal = Operation.Resample(signal, targetSampleRate);
        }

        return new WavData(signal.Samples, targetSampleRate);
    }
}

/// <summary>
/// Represents audio data read from an audio file.
/// </summary>
internal readonly struct WavData
{
    /// <summary>
    /// Audio samples as f32 values.
    /// </summary>
    public float[] Samples { get; }

    /// <summary>
    /// Sample rate in Hz.
    /// </summary>
    public uint SampleRate { get; }

    public WavData(float[] samples, uint sampleRate)
    {
        Samples = samples;
        SampleRate = sampleRate;
    }

    public WavData(float[] samples, int sampleRate)
    {
        Samples = samples;
        SampleRate = (uint)sampleRate;
    }
}

/// <summary>
/// Stream wrapper that doesn't dispose the underlying stream.
/// </summary>
internal sealed class NonDisposableStream : Stream
{
    private readonly Stream _baseStream;

    public NonDisposableStream(Stream baseStream)
    {
        _baseStream = baseStream;
    }

    public override bool CanRead => _baseStream.CanRead;
    public override bool CanSeek => _baseStream.CanSeek;
    public override bool CanWrite => _baseStream.CanWrite;
    public override long Length => _baseStream.Length;
    public override long Position
    {
        get => _baseStream.Position;
        set => _baseStream.Position = value;
    }

    public override void Flush() => _baseStream.Flush();
    public override int Read(byte[] buffer, int offset, int count) => _baseStream.Read(buffer, offset, count);
    public override long Seek(long offset, SeekOrigin origin) => _baseStream.Seek(offset, origin);
    public override void SetLength(long value) => _baseStream.SetLength(value);
    public override void Write(byte[] buffer, int offset, int count) => _baseStream.Write(buffer, offset, count);

    protected override void Dispose(bool disposing)
    {
        // Don't dispose the base stream
    }
}
