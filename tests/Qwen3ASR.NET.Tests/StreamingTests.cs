using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;
using Xunit;

namespace Qwen3ASR.NET.Tests;

public class StreamingTests
{
    [Fact]
    public void StreamOptions_DefaultChunkSize_Is2Seconds()
    {
        var options = new StreamOptions();
        Assert.Equal(2.0f, options.ChunkSizeSec);
    }

    [Fact]
    public void StreamOptions_UnfixedChunkNum_DefaultIs2()
    {
        var options = new StreamOptions();
        Assert.Equal(2, options.UnfixedChunkNum);
    }

    [Fact]
    public void StreamOptions_WithCustomValues_Works()
    {
        var options = new StreamOptions
        {
            ChunkSizeSec = 1.0f,
            UnfixedChunkNum = 3,
            UnfixedTokenNum = 10,
            MaxNewTokens = 512,
            AudioWindowSec = 30.0f,
            TextWindowTokens = 100,
            Context = "Previous context",
            Language = Language.Japanese
        };

        Assert.Equal(1.0f, options.ChunkSizeSec);
        Assert.Equal(3, options.UnfixedChunkNum);
        Assert.Equal(10, options.UnfixedTokenNum);
        Assert.Equal(512, options.MaxNewTokens);
        Assert.Equal(30.0f, options.AudioWindowSec);
        Assert.Equal(100, options.TextWindowTokens);
        Assert.Equal("Previous context", options.Context);
        Assert.Equal(Language.Japanese, options.Language);
    }

    [Fact]
    public void StreamOptions_StaticDefault_ReturnsNewInstance()
    {
        var default1 = StreamOptions.Default;
        var default2 = StreamOptions.Default;

        Assert.NotSame(default1, default2);
        Assert.Equal(default1.ChunkSizeSec, default2.ChunkSizeSec);
    }

    [Fact]
    public void TranscriptionOptions_Default_ReturnsNewInstance()
    {
        var default1 = TranscriptionOptions.Default;
        var default2 = TranscriptionOptions.Default;

        Assert.NotSame(default1, default2);
    }

    [Fact]
    public void TranscriptionOptions_WithAllOptions_Works()
    {
        var options = new TranscriptionOptions
        {
            Language = Language.English,
            Context = "Test context",
            ReturnTimestamps = true,
            MaxNewTokens = 100,
            MaxBatchSize = 16,
            ChunkMaxSec = 30.0f,
            BucketByLength = true
        };

        Assert.Equal(Language.English, options.Language);
        Assert.Equal("Test context", options.Context);
        Assert.True(options.ReturnTimestamps);
        Assert.Equal(100, options.MaxNewTokens);
        Assert.Equal(16, options.MaxBatchSize);
        Assert.Equal(30.0f, options.ChunkMaxSec);
        Assert.True(options.BucketByLength);
    }

    [Fact]
    public void TranscriptionResult_WithPartialFlag_Works()
    {
        var partialResult = new TranscriptionResult
        {
            Text = "Hello",
            IsPartial = true
        };

        var finalResult = new TranscriptionResult
        {
            Text = "Hello world",
            IsPartial = false
        };

        Assert.True(partialResult.IsPartial);
        Assert.False(finalResult.IsPartial);
    }

    [Fact]
    public void TranscriptionResult_WithTimestamps_Works()
    {
        var result = new TranscriptionResult
        {
            Text = "Hello world",
            Timestamps = new List<Timestamp>
            {
                new() { Start = 0.0f, End = 0.5f, Text = "Hello" },
                new() { Start = 0.5f, End = 1.0f, Text = "world" }
            }
        };

        Assert.Equal(2, result.Timestamps.Count);
        Assert.Equal(0.0f, result.Timestamps[0].Start);
        Assert.Equal(1.0f, result.Timestamps[1].End);
    }

    [Fact]
    public void Timestamp_DefaultValues_AreZero()
    {
        var timestamp = new Timestamp();
        Assert.Equal(0.0f, timestamp.Start);
        Assert.Equal(0.0f, timestamp.End);
        Assert.Equal(string.Empty, timestamp.Text);
    }

    [Fact]
    public void LoadOptions_WithAllOptions_Works()
    {
        var options = new LoadOptions
        {
            ModelPath = "Qwen/Qwen3-ASR-0.6B",
            Device = DeviceType.Cuda,
            Revision = "main",
            NumThreads = 4
        };

        Assert.Equal("Qwen/Qwen3-ASR-0.6B", options.ModelPath);
        Assert.Equal(DeviceType.Cuda, options.Device);
        Assert.Equal("main", options.Revision);
        Assert.Equal(4, options.NumThreads);
    }

    [Fact]
    public void Language_AllSupportedLanguages_HaveNativeString()
    {
        var supportedLanguages = new[]
        {
            Language.Japanese,
            Language.English,
            Language.Chinese,
            Language.Korean,
            Language.French,
            Language.German,
            Language.Spanish
        };

        foreach (var lang in supportedLanguages)
        {
            var nativeString = lang.ToNativeString();
            Assert.NotNull(nativeString);
            Assert.NotEmpty(nativeString);
        }
    }

    [Fact]
    public void Language_Auto_ReturnsNull()
    {
        Assert.Null(Language.Auto.ToNativeString());
    }

    [Fact]
    public void ErrorCode_AllValues_AreDefined()
    {
        Assert.Equal(0, (int)ErrorCode.Success);
        Assert.Equal(1, (int)ErrorCode.InvalidHandle);
        Assert.Equal(2, (int)ErrorCode.InvalidParameter);
        Assert.Equal(3, (int)ErrorCode.ModelNotLoaded);
        Assert.Equal(4, (int)ErrorCode.InferenceError);
        Assert.Equal(5, (int)ErrorCode.MemoryError);
        Assert.Equal(6, (int)ErrorCode.StreamError);
        Assert.Equal(99, (int)ErrorCode.UnknownError);
    }
}
