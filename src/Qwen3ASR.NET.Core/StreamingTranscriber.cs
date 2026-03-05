using System.Runtime.InteropServices;
using System.Text;
using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;
using Qwen3ASR.NET.Native;

namespace Qwen3ASR.NET;

/// <summary>
/// Streaming transcriber for real-time speech recognition.
/// Uses native streaming API for efficient incremental transcription.
/// </summary>
public sealed class StreamingTranscriber : IAsyncDisposable, IDisposable
{
    private const int SampleRate = 16000;

    private readonly Qwen3Asr _asr;
    private readonly StreamOptions _options;
    private IntPtr _streamHandle;
    private readonly TranscriptionOptions _transcribeOptions;

    private bool _disposed;
    private bool _finished;
    private int _totalSamplesProcessed;

    /// <summary>
    /// Gets whether the stream is still active.
    /// </summary>
    public bool IsActive => !_disposed && !_finished && _streamHandle != IntPtr.Zero;

    /// <summary>
    /// Gets the total duration processed in seconds.
    /// </summary>
    public float Duration => (float)_totalSamplesProcessed / SampleRate;

    internal StreamingTranscriber(Qwen3Asr asr, StreamOptions options)
    {
        _asr = asr ?? throw new ArgumentNullException(nameof(asr));
        _options = options;

        // Create native stream
        _streamHandle = CreateNativeStream();
        if (_streamHandle == IntPtr.Zero)
        {
            throw new Qwen3AsrException("Failed to create native streaming session");
        }

        _transcribeOptions = new TranscriptionOptions
        {
            Language = options.Language,
            Context = options.Context,
            ReturnTimestamps = false,
            MaxNewTokens = options.MaxNewTokens
        };
    }

    private IntPtr CreateNativeStream()
    {
        var streamOpts = new NativeBindings.StreamOptionsFFI
        {
            Language = IntPtr.Zero,
            Context = IntPtr.Zero,
            ChunkSizeSec = _options.ChunkSizeSec,
            UnfixedChunkNum = _options.UnfixedChunkNum,
            UnfixedTokenNum = _options.UnfixedTokenNum,
            AudioWindowSec = _options.AudioWindowSec ?? 0f,
            TextWindowTokens = _options.TextWindowTokens ?? 0,
            MaxNewTokens = _options.MaxNewTokens
        };

        // Set language if specified
        if (_options.Language != Language.Auto)
        {
            streamOpts.Language = Marshal.StringToHGlobalAnsi(_options.Language.ToNativeString());
        }

        // Set context if specified
        if (!string.IsNullOrEmpty(_options.Context))
        {
            streamOpts.Context = Marshal.StringToHGlobalAnsi(_options.Context);
        }

        try
        {
            var handle = NativeBindings.qwen3_asr_stream_create(
                _asr.Handle,
                ref streamOpts,
                out IntPtr errorOut);

            if (handle == IntPtr.Zero && errorOut != IntPtr.Zero)
            {
                var errorMessage = Marshal.PtrToStringUTF8(errorOut);
                NativeBindings.qwen3_asr_free_string(errorOut);
                throw new Qwen3AsrException($"Failed to create stream: {errorMessage}");
            }

            return handle;
        }
        finally
        {
            if (streamOpts.Language != IntPtr.Zero)
                Marshal.FreeHGlobal(streamOpts.Language);
            if (streamOpts.Context != IntPtr.Zero)
                Marshal.FreeHGlobal(streamOpts.Context);
        }
    }

    /// <summary>
    /// Pushes audio samples to the stream.
    /// </summary>
    /// <param name="audioSamples">Audio samples as 32-bit floats (16kHz, mono).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The partial transcription result.</returns>
    public async Task<TranscriptionResult> PushAsync(
        float[] audioSamples,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_finished)
            throw new InvalidOperationException("Stream is already finished");

        ArgumentNullException.ThrowIfNull(audioSamples);

        if (audioSamples.Length == 0)
            return GetPartialResult();

        cancellationToken.ThrowIfCancellationRequested();

        // Run native API call on thread pool
        return await Task.Run(() =>
        {
            var result = NativeBindings.qwen3_asr_stream_push(
                _streamHandle,
                audioSamples,
                (UIntPtr)audioSamples.Length,
                SampleRate);

            _totalSamplesProcessed += audioSamples.Length;

            return ProcessNativeResult(result, isPartial: true);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the current partial result without pushing more audio.
    /// </summary>
    /// <returns>The partial transcription result.</returns>
    public TranscriptionResult GetPartialResult()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Return empty partial result (native API returns partial on push)
        return new TranscriptionResult
        {
            Text = string.Empty,
            Language = _options.Language.ToNativeString(),
            IsPartial = true
        };
    }

    /// <summary>
    /// Finishes the stream and returns the final transcription result.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The final transcription result.</returns>
    public async Task<TranscriptionResult> FinishAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_finished)
            throw new InvalidOperationException("Stream is already finished");

        _finished = true;

        cancellationToken.ThrowIfCancellationRequested();

        // Run native API call on thread pool
        return await Task.Run(() =>
        {
            var result = NativeBindings.qwen3_asr_stream_finish(_streamHandle);
            return ProcessNativeResult(result, isPartial: false);
        }, cancellationToken).ConfigureAwait(false);
    }

    private TranscriptionResult ProcessNativeResult(NativeBindings.TranscriptionResultFFI ffiResult, bool isPartial)
    {
        try
        {
            if (ffiResult.Code != NativeBindings.ResultCode.Success)
            {
                var errorMessage = ffiResult.Error != IntPtr.Zero
                    ? Marshal.PtrToStringUTF8(ffiResult.Error) ?? "Unknown error"
                    : $"Error code: {ffiResult.Code}";
                throw new Qwen3AsrException($"Transcription failed: {errorMessage}");
            }

            if (ffiResult.Json == IntPtr.Zero)
            {
                return new TranscriptionResult
                {
                    Text = string.Empty,
                    Language = _options.Language.ToNativeString(),
                    IsPartial = isPartial
                };
            }

            var json = Marshal.PtrToStringUTF8(ffiResult.Json);
            if (string.IsNullOrEmpty(json))
            {
                return new TranscriptionResult
                {
                    Text = string.Empty,
                    Language = _options.Language.ToNativeString(),
                    IsPartial = isPartial
                };
            }

            return System.Text.Json.JsonSerializer.Deserialize<TranscriptionResult>(json)
                ?? new TranscriptionResult
                {
                    Text = string.Empty,
                    Language = _options.Language.ToNativeString(),
                    IsPartial = isPartial
                };
        }
        finally
        {
            NativeBindings.qwen3_asr_free_result(ref ffiResult);
        }
    }

    /// <summary>
    /// Releases all resources used by the StreamingTranscriber.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        if (_streamHandle != IntPtr.Zero)
        {
            NativeBindings.qwen3_asr_stream_destroy(_streamHandle);
            _streamHandle = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Asynchronously releases all resources used by the StreamingTranscriber.
    /// </summary>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    /// <summary>
    /// Finalizer to ensure resources are released.
    /// </summary>
    ~StreamingTranscriber()
    {
        Dispose();
    }
}
