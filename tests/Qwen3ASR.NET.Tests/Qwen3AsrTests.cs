using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;
using Xunit;

namespace Qwen3ASR.NET.Tests;

public class Qwen3AsrTests
{
    [Fact]
    public void LoadOptions_DefaultValues_AreCorrect()
    {
        var options = new LoadOptions
        {
            ModelPath = "test-model"
        };

        Assert.Equal(DeviceType.Cpu, options.Device);
        Assert.Equal("test-model", options.ModelPath);
        Assert.Null(options.Revision);
        Assert.Equal(0, options.NumThreads);
    }

    [Fact]
    public void LoadOptions_WithModelPath_SetsModelPath()
    {
        var options = new LoadOptions { ModelPath = "Qwen/Qwen3-ASR-0.6B" };

        Assert.Equal("Qwen/Qwen3-ASR-0.6B", options.ModelPath);
    }

    [Fact]
    public void StreamOptions_DefaultValues_AreCorrect()
    {
        var options = new StreamOptions();

        Assert.Null(options.Language);
        Assert.Equal(0.5f, options.ChunkSizeSec);
        Assert.True(options.EnableTimestamps);
        Assert.True(options.EnablePartialResults);
    }

    [Fact]
    public void DeviceType_Values_MatchFFI()
    {
        Assert.Equal(0, (int)DeviceType.Cpu);
        Assert.Equal(1, (int)DeviceType.Cuda);
        Assert.Equal(2, (int)DeviceType.Metal);
    }

    [Fact]
    public void TranscriptionResult_DefaultValues_AreCorrect()
    {
        var result = new TranscriptionResult();

        Assert.Equal(string.Empty, result.Text);
        Assert.Null(result.Language);
        Assert.Null(result.Confidence);
        Assert.Null(result.Timestamps);
    }

    [Fact]
    public void TranscriptionResult_ToString_ReturnsText()
    {
        var result = new TranscriptionResult
        {
            Text = "Hello, world!"
        };

        Assert.Equal("Hello, world!", result.ToString());
    }

    [Fact]
    public void Timestamp_Properties_Work()
    {
        var timestamp = new Timestamp
        {
            Start = 0.5f,
            End = 1.5f,
            Text = "test"
        };

        Assert.Equal(0.5f, timestamp.Start);
        Assert.Equal(1.5f, timestamp.End);
        Assert.Equal("test", timestamp.Text);
    }
}
