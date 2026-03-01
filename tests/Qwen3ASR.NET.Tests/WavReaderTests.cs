using Qwen3ASR.NET.Audio;
using Xunit;

namespace Qwen3ASR.NET.Tests;

public class WavReaderTests
{
    /// <summary>
    /// Creates a valid WAV file in memory with specified parameters.
    /// </summary>
    private static byte[] CreateWavFile(
        int sampleRate = 16000,
        short channels = 1,
        short bitsPerSample = 16,
        float durationSec = 1.0f,
        float frequency = 440f)
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        int samples = (int)(sampleRate * durationSec);
        int dataSize = samples * channels * (bitsPerSample / 8);
        int fileSize = 36 + dataSize;

        // RIFF header
        writer.Write("RIFF".ToCharArray());
        writer.Write(fileSize);
        writer.Write("WAVE".ToCharArray());

        // fmt chunk
        writer.Write("fmt ".ToCharArray());
        writer.Write(16); // chunk size
        writer.Write((short)1); // audio format (PCM)
        writer.Write(channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * (bitsPerSample / 8)); // byte rate
        writer.Write((short)(channels * (bitsPerSample / 8))); // block align
        writer.Write(bitsPerSample);

        // data chunk
        writer.Write("data".ToCharArray());
        writer.Write(dataSize);

        // Generate sine wave
        for (int i = 0; i < samples; i++)
        {
            double t = i / (double)sampleRate;
            double value = Math.Sin(2 * Math.PI * frequency * t);

            if (bitsPerSample == 16)
            {
                short sample = (short)(value * 32767);
                for (int c = 0; c < channels; c++)
                    writer.Write(sample);
            }
            else if (bitsPerSample == 8)
            {
                byte sample = (byte)((value + 1) * 127);
                for (int c = 0; c < channels; c++)
                    writer.Write(sample);
            }
            else if (bitsPerSample == 32)
            {
                float sample = (float)value;
                for (int c = 0; c < channels; c++)
                    writer.Write(sample);
            }
        }

        return memoryStream.ToArray();
    }

    [Fact]
    public void ReadWav_16BitMono_ReturnsCorrectSamples()
    {
        var wavData = CreateWavFile(16000, 1, 16, 1.0f);
        var result = WavReader.ReadWav(wavData);

        Assert.Equal(16000, (int)result.SampleRate);
        Assert.Equal(16000, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_16BitStereo_ConvertsToMono()
    {
        var wavData = CreateWavFile(16000, 2, 16, 1.0f);
        var result = WavReader.ReadWav(wavData);

        // Should be converted to mono
        Assert.Equal(16000, (int)result.SampleRate);
        Assert.Equal(16000, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_8BitMono_ConvertsCorrectly()
    {
        var wavData = CreateWavFile(16000, 1, 8, 0.5f);
        var result = WavReader.ReadWav(wavData);

        Assert.Equal(16000, (int)result.SampleRate);
        Assert.Equal(8000, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_32BitFloatMono_ConvertsCorrectly()
    {
        var wavData = CreateWavFile(16000, 1, 32, 0.5f);
        var result = WavReader.ReadWav(wavData);

        Assert.Equal(16000, (int)result.SampleRate);
        Assert.Equal(8000, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_ResamplesTo16kHz()
    {
        var wavData = CreateWavFile(44100, 1, 16, 1.0f);
        var result = WavReader.ReadWav(wavData);

        Assert.Equal(16000, (int)result.SampleRate);
        // Should have approximately 16000 samples (resampled from 44100)
        Assert.True(Math.Abs(result.Samples.Length - 16000) < 100);
    }

    [Fact]
    public void ReadWav_ResamplesFrom8kHz()
    {
        var wavData = CreateWavFile(8000, 1, 16, 1.0f);
        var result = WavReader.ReadWav(wavData);

        Assert.Equal(16000, (int)result.SampleRate);
        // Should have 16000 samples (upsampled from 8000)
        Assert.Equal(16000, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_InvalidFile_ThrowsException()
    {
        var invalidData = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        Assert.Throws<InvalidDataException>(() => WavReader.ReadWav(invalidData));
    }

    [Fact]
    public void ReadWav_MissingDataChunk_ThrowsException()
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        // RIFF header only
        writer.Write("RIFF".ToCharArray());
        writer.Write(36);
        writer.Write("WAVE".ToCharArray());
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(16000);
        writer.Write(32000);
        writer.Write((short)2);
        writer.Write((short)16);

        Assert.Throws<InvalidDataException>(() => WavReader.ReadWav(memoryStream.ToArray()));
    }

    [Fact]
    public void ReadWav_SamplesAreNormalized()
    {
        var wavData = CreateWavFile(16000, 1, 16, 0.1f);
        var result = WavReader.ReadWav(wavData);

        // All samples should be in range [-1, 1]
        foreach (var sample in result.Samples)
        {
            Assert.True(sample >= -1.0f && sample <= 1.0f, $"Sample {sample} is out of range");
        }
    }

    [Fact]
    public void ReadWav_FromStream_Works()
    {
        var wavData = CreateWavFile(16000, 1, 16, 0.5f);
        using var stream = new MemoryStream(wavData);
        var result = WavReader.ReadWav(stream);

        Assert.Equal(16000, (int)result.SampleRate);
        Assert.Equal(8000, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_VeryShortAudio_Works()
    {
        // 1ms audio
        var wavData = CreateWavFile(16000, 1, 16, 0.001f);
        var result = WavReader.ReadWav(wavData);

        Assert.Equal(16000, (int)result.SampleRate);
        Assert.True(result.Samples.Length >= 16); // At least 16 samples
    }

    [Fact]
    public void ReadWav_24Bit_Works()
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        int sampleRate = 16000;
        short channels = 1;
        short bitsPerSample = 24;
        int samples = 1600; // 0.1 second
        int dataSize = samples * channels * 3;
        int fileSize = 36 + dataSize;

        // RIFF header
        writer.Write("RIFF".ToCharArray());
        writer.Write(fileSize);
        writer.Write("WAVE".ToCharArray());

        // fmt chunk
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);
        writer.Write((short)1); // PCM
        writer.Write(channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * 3); // byte rate
        writer.Write((short)(channels * 3)); // block align
        writer.Write(bitsPerSample);

        // data chunk
        writer.Write("data".ToCharArray());
        writer.Write(dataSize);

        // Generate samples
        for (int i = 0; i < samples; i++)
        {
            double t = i / (double)sampleRate;
            double value = Math.Sin(2 * Math.PI * 440 * t);
            int sample = (int)(value * 8388607);

            // Write 24-bit little-endian
            writer.Write((byte)(sample & 0xFF));
            writer.Write((byte)((sample >> 8) & 0xFF));
            writer.Write((byte)((sample >> 16) & 0xFF));
        }

        var result = WavReader.ReadWav(memoryStream.ToArray());

        Assert.Equal(16000, (int)result.SampleRate);
        Assert.Equal(1600, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_MultiChannel_ConvertsToMono()
    {
        // 4-channel audio
        var wavData = CreateWavFile(16000, 4, 16, 0.5f);
        var result = WavReader.ReadWav(wavData);

        Assert.Equal(16000, (int)result.SampleRate);
        // Original: 8000 samples * 4 channels = 32000 values
        // After mono: 8000 samples
        Assert.Equal(8000, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_NonPCMFormat_ThrowsException()
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        // RIFF header
        writer.Write("RIFF".ToCharArray());
        writer.Write(50);
        writer.Write("WAVE".ToCharArray());

        // fmt chunk with non-PCM format (format code 3 = IEEE float)
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);
        writer.Write((short)3); // Non-PCM
        writer.Write((short)1);
        writer.Write(16000);
        writer.Write(64000);
        writer.Write((short)4);
        writer.Write((short)32);

        // data chunk
        writer.Write("data".ToCharArray());
        writer.Write(4);
        writer.Write(0f);

        Assert.Throws<NotSupportedException>(() => WavReader.ReadWav(memoryStream.ToArray()));
    }

    [Fact]
    public void ReadWav_EmptyDataChunk_ReturnsEmptySamples()
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        // RIFF header
        writer.Write("RIFF".ToCharArray());
        writer.Write(36);
        writer.Write("WAVE".ToCharArray());

        // fmt chunk
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(16000);
        writer.Write(32000);
        writer.Write((short)2);
        writer.Write((short)16);

        // Empty data chunk
        writer.Write("data".ToCharArray());
        writer.Write(0);

        var result = WavReader.ReadWav(memoryStream.ToArray());
        Assert.Empty(result.Samples);
    }

    [Fact]
    public void ReadWav_SilentAudio_ReturnsZeroSamples()
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        int sampleRate = 16000;
        int samples = 1600;

        // RIFF header
        writer.Write("RIFF".ToCharArray());
        writer.Write(36 + samples * 2);
        writer.Write("WAVE".ToCharArray());

        // fmt chunk
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);
        writer.Write((short)2);
        writer.Write((short)16);

        // data chunk with silence
        writer.Write("data".ToCharArray());
        writer.Write(samples * 2);

        for (int i = 0; i < samples; i++)
        {
            writer.Write((short)0); // Silent
        }

        var result = WavReader.ReadWav(memoryStream.ToArray());

        Assert.Equal(1600, result.Samples.Length);
        Assert.All(result.Samples, s => Assert.Equal(0.0f, s, 3));
    }

    [Fact]
    public void ReadWav_ExtraFmtChunk_IgnoresExtraBytes()
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        int sampleRate = 16000;
        int samples = 100;

        // RIFF header
        writer.Write("RIFF".ToCharArray());
        writer.Write(36 + samples * 2 + 10); // Extra 10 bytes in fmt
        writer.Write("WAVE".ToCharArray());

        // fmt chunk with extra bytes (like in some WAV files)
        writer.Write("fmt ".ToCharArray());
        writer.Write(26); // 16 + 10 extra
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);
        writer.Write((short)2);
        writer.Write((short)16);
        // Extra 10 bytes
        for (int i = 0; i < 10; i++)
            writer.Write((byte)0);

        // data chunk
        writer.Write("data".ToCharArray());
        writer.Write(samples * 2);

        for (int i = 0; i < samples; i++)
            writer.Write((short)0);

        var result = WavReader.ReadWav(memoryStream.ToArray());
        Assert.Equal(100, result.Samples.Length);
    }

    [Fact]
    public void ReadWav_SkipsUnknownChunks()
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        int sampleRate = 16000;
        int samples = 100;

        // RIFF header
        writer.Write("RIFF".ToCharArray());
        writer.Write(36 + samples * 2 + 12); // +12 for unknown chunk
        writer.Write("WAVE".ToCharArray());

        // Unknown chunk (should be skipped)
        writer.Write("LIST".ToCharArray());
        writer.Write(4);
        writer.Write("INFO".ToCharArray());

        // fmt chunk
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);
        writer.Write((short)2);
        writer.Write((short)16);

        // data chunk
        writer.Write("data".ToCharArray());
        writer.Write(samples * 2);

        for (int i = 0; i < samples; i++)
            writer.Write((short)0);

        var result = WavReader.ReadWav(memoryStream.ToArray());
        Assert.Equal(100, result.Samples.Length);
    }
}
