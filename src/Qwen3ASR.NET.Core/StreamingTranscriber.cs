using System.Runtime.InteropServices;
using System.Text.Json;
using Qwen3ASR.NET.Models;
using Qwen3ASR.NET.Native;

namespace Qwen3ASR.NET;

/// <summary>
/// Streaming transcriber for real-time speech recognition.
/// </summary>
public sealed class StreamingTranscriber : IAsyncDisposable, IDisposable
{
    private IntPtr _handle;
    private readonly StreamOptions _options;
    private bool _disposed;

    internal StreamingTranscriber(IntPtr handle, StreamOptions options)
    {
        _handle = handle;
        _options = options;
    }

    /// <summary>
    /// Gets whether the stream is still active.
    /// </summary>
    public bool IsActive => _handle != IntPtr.Zero && !_disposed;

    /// <summary>
    /// Pushes audio samples to the stream.
    /// </summary>
    /// <param name="audioSamples">Audio samples as 32-bit floats (16kHz, mono).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The partial transcription result.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when this instance is disposed.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the stream is finished.</exception>
    /// <exception cref="Qwen3AsrException">Thrown when transcription fails.</exception>
    public async Task<TranscriptionResult> PushAsync(
        float[] audioSamples,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_handle == IntPtr.Zero)
            throw new InvalidOperationException("Stream is finished");

        ArgumentNullException.ThrowIfNull(audioSamples);

        return await Task.Run(() =>
        {
            var result = NativeBindings.qwen3_asr_stream_push(
                _handle,
                audioSamples,
                (UIntPtr)audioSamples.Length);
            return ProcessResult(result);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the current partial result without pushing more audio.
    /// </summary>
    /// <returns>The partial transcription result, or null if no partial result is available.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when this instance is disposed.</exception>
    public TranscriptionResult? GetPartialResult()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_handle == IntPtr.Zero)
            return null;

        var result = NativeBindings.qwen3_asr_stream_get_partial(_handle);

        if (result.Code == NativeBindings.ResultCode.Success && result.Text != IntPtr.Zero)
        {
            return ProcessResult(result);
        }

        return null;
    }

    /// <summary>
    /// Finishes the stream and returns the final transcription result.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The final transcription result.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when this instance is disposed.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the stream is already finished.</exception>
    /// <exception cref="Qwen3AsrException">Thrown when finishing fails.</exception>
    public async Task<TranscriptionResult> FinishAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_handle == IntPtr.Zero)
            throw new InvalidOperationException("Stream is already finished");

        return await Task.Run(() =>
        {
            var handle = _handle;
            _handle = IntPtr.Zero;

            var result = NativeBindings.qwen3_asr_stream_finish(handle);
            return ProcessResult(result);
        }, cancellationToken).ConfigureAwait(false);
    }

    private static TranscriptionResult ProcessResult(NativeBindings.TranscriptionResultFFI ffiResult)
    {
        try
        {
            if (ffiResult.Code != NativeBindings.ResultCode.Success)
            {
                var error = PtrToString(ffiResult.ErrorMessage) ?? "Unknown error";
                throw new Qwen3AsrException(error, ffiResult.Code);
            }

            var text = PtrToString(ffiResult.Text) ?? string.Empty;
            var jsonResult = PtrToString(ffiResult.JsonResult);

            if (!string.IsNullOrEmpty(jsonResult))
            {
                try
                {
                    var result = JsonSerializer.Deserialize<TranscriptionResult>(jsonResult);
                    if (result != null)
                    {
                        return result;
                    }
                }
                catch (JsonException)
                {
                    // Fall through to basic result
                }
            }

            return new TranscriptionResult { Text = text };
        }
        finally
        {
            NativeBindings.qwen3_asr_free_result(ref ffiResult);
        }
    }

    private static string? PtrToString(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero)
            return null;

        return Marshal.PtrToStringAnsi(ptr);
    }

    /// <summary>
    /// Disposes this instance synchronously.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        if (_handle != IntPtr.Zero)
        {
            // Finish the stream to clean up resources
            try
            {
                NativeBindings.qwen3_asr_stream_finish(_handle);
            }
            catch
            {
                // Ignore errors during cleanup
            }
            _handle = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes this instance asynchronously.
    /// </summary>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    /// <summary>
    /// Finalizer.
    /// </summary>
    ~StreamingTranscriber()
    {
        Dispose();
    }
}
