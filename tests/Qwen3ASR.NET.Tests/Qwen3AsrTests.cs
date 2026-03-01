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

        Assert.Equal(Language.Auto, options.Language);
        Assert.Equal(2.0f, options.ChunkSizeSec);
        Assert.Equal(2, options.UnfixedChunkNum);
        Assert.Equal(5, options.UnfixedTokenNum);
        Assert.Equal(256, options.MaxNewTokens);
        Assert.Null(options.AudioWindowSec);
        Assert.Null(options.TextWindowTokens);
        Assert.Null(options.Context);
    }

    [Fact]
    public void TranscriptionOptions_DefaultValues_AreCorrect()
    {
        var options = new TranscriptionOptions();

        Assert.Equal(Language.Auto, options.Language);
        Assert.False(options.ReturnTimestamps);
        Assert.Equal(32, options.MaxBatchSize);
        Assert.Equal(0, options.MaxNewTokens);
        Assert.Null(options.ChunkMaxSec);
        Assert.False(options.BucketByLength);
        Assert.Null(options.Context);
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
        Assert.False(result.IsPartial);
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

    [Fact]
    public void Language_ToNativeString_ReturnsCorrectValues()
    {
        Assert.Null(Language.Auto.ToNativeString());
        Assert.Equal("Japanese", Language.Japanese.ToNativeString());
        Assert.Equal("English", Language.English.ToNativeString());
        Assert.Equal("Chinese", Language.Chinese.ToNativeString());
        Assert.Equal("Korean", Language.Korean.ToNativeString());
    }

    [Fact]
    public void Qwen3AsrException_WithErrorCode_SetsErrorCode()
    {
        var exception = new Qwen3AsrException("Test error", ErrorCode.InferenceError);

        Assert.Equal("Test error", exception.Message);
        Assert.Equal(ErrorCode.InferenceError, exception.ErrorCode);
    }

    [Fact]
    public void ErrorCode_Values_MatchFFI()
    {
        Assert.Equal(0, (int)ErrorCode.Success);
        Assert.Equal(1, (int)ErrorCode.InvalidHandle);
        Assert.Equal(2, (int)ErrorCode.InvalidParameter);
        Assert.Equal(3, (int)ErrorCode.ModelNotLoaded);
        Assert.Equal(4, (int)ErrorCode.InferenceError);
        Assert.Equal(5, (int)ErrorCode.MemoryError);
        Assert.Equal(99, (int)ErrorCode.UnknownError);
    }

    [Fact]
    public void Qwen3AsrException_WithInnerException_PreservesInner()
    {
        var inner = new InvalidOperationException("Inner error");
        var exception = new Qwen3AsrException("Outer error", inner);

        Assert.Equal("Outer error", exception.Message);
        Assert.Same(inner, exception.InnerException);
        Assert.Equal(ErrorCode.UnknownError, exception.ErrorCode);
    }

    [Fact]
    public void TranscriptionDetail_ProcessingTimeMs_Works()
    {
        var detail = new TranscriptionDetail
        {
            Text = "Test",
            ProcessingTimeMs = 1234
        };

        Assert.Equal("Test", detail.Text);
        Assert.Equal(1234, detail.ProcessingTimeMs);
    }

    [Fact]
    public void TranscriptionDetail_InheritsFromTranscriptionResult()
    {
        var detail = new TranscriptionDetail
        {
            Text = "Test",
            Language = "ja",
            IsPartial = true,
            ProcessingTimeMs = 500
        };

        Assert.IsAssignableFrom<TranscriptionResult>(detail);
        Assert.Equal("Test", detail.Text);
        Assert.Equal("ja", detail.Language);
        Assert.True(detail.IsPartial);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void TranscriptionResult_EmptyText_Works(string? text)
    {
        var result = new TranscriptionResult
        {
            Text = text ?? string.Empty
        };

        Assert.NotNull(result.Text);
    }

    [Fact]
    public void Timestamp_WithNegativeValues_Works()
    {
        var timestamp = new Timestamp
        {
            Start = -1.0f,  // Invalid but should not throw
            End = -0.5f,
            Text = ""
        };

        Assert.Equal(-1.0f, timestamp.Start);
        Assert.Equal(-0.5f, timestamp.End);
    }
}
