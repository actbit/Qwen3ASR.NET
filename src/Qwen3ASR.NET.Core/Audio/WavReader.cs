using System.Runtime.InteropServices;

namespace Qwen3ASR.NET.Audio;

/// <summary>
/// Cross-platform WAV file reader.
/// Supports standard RIFF WAV format with PCM audio.
/// </summary>
internal static class WavReader
{
    /// <summary>
    /// Reads a WAV file and converts it to 16kHz mono f32 samples.
    /// </summary>
    /// <param name="filePath">Path to the WAV file.</param>
    /// <returns>Audio data as f32 samples at 16kHz mono.</returns>
    public static WavData ReadWav(string filePath)
    {
        using var file = File.OpenRead(filePath);
        return ReadWav(file);
    }

    /// <summary>
    /// Reads WAV data from a stream and converts it to 16kHz mono f32 samples.
    /// </summary>
    /// <param name="stream">Stream containing WAV data.</param>
    /// <returns>Audio data as f32 samples at 16kHz mono.</returns>
    public static WavData ReadWav(Stream stream)
    {
        using var reader = new BinaryReader(stream);

        // Read RIFF header
        var riff = reader.ReadChars(4);
        if (riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F')
            throw new InvalidDataException("Not a valid WAV file: RIFF header not found");

        var fileSize = reader.ReadUInt32();

        var wave = reader.ReadChars(4);
        if (wave[0] != 'W' || wave[1] != 'A' || wave[2] != 'V' || wave[3] != 'E')
            throw new InvalidDataException("Not a valid WAV file: WAVE format not found");

        // Read chunks
        ushort channels = 0;
        uint sampleRate = 0;
        ushort bitsPerSample = 0;
        byte[]? audioData = null;

        while (stream.Position < stream.Length)
        {
            var chunkId = reader.ReadChars(4);
            var chunkSize = reader.ReadUInt32();

            if (chunkId[0] == 'f' && chunkId[1] == 'm' && chunkId[2] == 't' && chunkId[3] == ' ')
            {
                // Format chunk
                var audioFormat = reader.ReadUInt16();
                channels = reader.ReadUInt16();
                sampleRate = reader.ReadUInt32();
                var byteRate = reader.ReadUInt32();
                var blockAlign = reader.ReadUInt16();
                bitsPerSample = reader.ReadUInt16();

                // Skip extra format bytes if present
                if (chunkSize > 16)
                {
                    stream.Seek(chunkSize - 16, SeekOrigin.Current);
                }

                if (audioFormat != 1)
                    throw new NotSupportedException($"Unsupported audio format: {audioFormat}. Only PCM is supported.");
            }
            else if (chunkId[0] == 'd' && chunkId[1] == 'a' && chunkId[2] == 't' && chunkId[3] == 'a')
            {
                // Data chunk
                audioData = reader.ReadBytes((int)chunkSize);
            }
            else
            {
                // Skip unknown chunk
                stream.Seek(chunkSize, SeekOrigin.Current);
            }
        }

        if (audioData == null)
            throw new InvalidDataException("WAV file does not contain audio data");

        if (channels == 0 || sampleRate == 0 || bitsPerSample == 0)
            throw new InvalidDataException("WAV file format information is incomplete");

        // Convert to f32 samples
        var samples = ConvertToFloat(audioData, channels, bitsPerSample);

        // Resample to 16kHz if needed
        if (sampleRate != 16000)
        {
            samples = Resample(samples, sampleRate, 16000);
            sampleRate = 16000;
        }

        // Convert to mono if needed
        if (channels > 1)
        {
            samples = ToMono(samples, channels);
        }

        return new WavData(samples, 16000);
    }

    /// <summary>
    /// Reads WAV bytes and converts to 16kHz mono f32 samples.
    /// </summary>
    public static WavData ReadWav(byte[] wavBytes)
    {
        using var stream = new MemoryStream(wavBytes);
        return ReadWav(stream);
    }

    /// <summary>
    /// Converts raw PCM bytes to f32 samples.
    /// </summary>
    private static float[] ConvertToFloat(byte[] data, ushort channels, ushort bitsPerSample)
    {
        int bytesPerSample = bitsPerSample / 8;
        int sampleCount = data.Length / bytesPerSample / channels;
        float[] samples = new float[sampleCount * channels];

        if (bitsPerSample == 16)
        {
            for (int i = 0; i < sampleCount * channels; i++)
            {
                short sample = BitConverter.ToInt16(data, i * 2);
                samples[i] = sample / 32768f;
            }
        }
        else if (bitsPerSample == 8)
        {
            for (int i = 0; i < sampleCount * channels; i++)
            {
                byte sample = data[i];
                samples[i] = (sample - 128) / 128f;
            }
        }
        else if (bitsPerSample == 32)
        {
            if (data.Length % 4 != 0)
                throw new InvalidDataException("32-bit audio data length is not a multiple of 4");

            // Assume 32-bit float
            for (int i = 0; i < sampleCount * channels; i++)
            {
                samples[i] = BitConverter.ToSingle(data, i * 4);
            }
        }
        else if (bitsPerSample == 24)
        {
            for (int i = 0; i < sampleCount * channels; i++)
            {
                int b0 = data[i * 3];
                int b1 = data[i * 3 + 1];
                int b2 = data[i * 3 + 2];

                // Sign extend
                if ((b2 & 0x80) != 0)
                    b2 |= unchecked((int)0xFFFFFF00);

                int sample = (b2 << 16) | (b1 << 8) | b0;
                samples[i] = sample / 8388608f; // 2^23
            }
        }
        else
        {
            throw new NotSupportedException($"Unsupported bits per sample: {bitsPerSample}");
        }

        return samples;
    }

    /// <summary>
    /// Simple linear resampling.
    /// </summary>
    private static float[] Resample(float[] samples, uint fromRate, uint toRate)
    {
        if (fromRate == toRate)
            return samples;

        double ratio = (double)fromRate / toRate;
        int newLength = (int)(samples.Length / ratio);
        float[] result = new float[newLength];

        for (int i = 0; i < newLength; i++)
        {
            double srcPos = i * ratio;
            int srcIndex = (int)srcPos;
            double frac = srcPos - srcIndex;

            if (srcIndex + 1 < samples.Length)
            {
                result[i] = (float)(samples[srcIndex] * (1 - frac) + samples[srcIndex + 1] * frac);
            }
            else
            {
                result[i] = samples[Math.Min(srcIndex, samples.Length - 1)];
            }
        }

        return result;
    }

    /// <summary>
    /// Converts multi-channel audio to mono by averaging channels.
    /// </summary>
    private static float[] ToMono(float[] samples, ushort channels)
    {
        if (channels == 1)
            return samples;

        int sampleCount = samples.Length / channels;
        float[] mono = new float[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++)
            {
                sum += samples[i * channels + c];
            }
            mono[i] = sum / channels;
        }

        return mono;
    }
}

/// <summary>
/// Represents audio data read from a WAV file.
/// </summary>
internal readonly struct WavData
{
    /// <summary>
    /// Audio samples as f32 values.
    /// </summary>
    public float[] Samples { get; }

    /// <summary>
    /// Sample rate in Hz (always 16000 after conversion).
    /// </summary>
    public uint SampleRate { get; }

    public WavData(float[] samples, uint sampleRate)
    {
        Samples = samples;
        SampleRate = sampleRate;
    }
}
