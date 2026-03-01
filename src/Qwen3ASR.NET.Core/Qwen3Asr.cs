using System.Runtime.InteropServices;
using System.Text.Json;
using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;
using Qwen3ASR.NET.Native;

namespace Qwen3ASR.NET;

/// <summary>
/// Main API for Qwen3-ASR speech recognition.
/// </summary>
public sealed class Qwen3Asr : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;
    private readonly string _modelPath;

    /// <summary>
    /// Gets the model path or HuggingFace model ID.
    /// </summary>
    public string ModelPath => _modelPath;

    /// <summary>
    /// Gets the device used for inference.
    /// </summary>
    public DeviceType Device { get; }

    /// <summary>
    /// Gets whether the model is loaded.
    /// </summary>
    public bool IsLoaded => _handle != IntPtr.Zero;

    private Qwen3Asr(IntPtr handle, string modelPath, DeviceType device)
    {
        _handle = handle;
        _modelPath = modelPath;
        Device = device;
    }

    /// <summary>
    /// Creates a new Qwen3Asr instance from a pretrained model.
    /// </summary>
    /// <param name="modelPath">The model path or HuggingFace model ID (e.g., "Qwen/Qwen3-ASR-0.6B").</param>
    /// <param name="device">The device to use for inference. Default is CPU.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A new Qwen3Asr instance.</returns>
    /// <exception cref="ArgumentNullException">Thrown when modelPath is null or empty.</exception>
    /// <exception cref="Qwen3AsrException">Thrown when model loading fails.</exception>
    public static async Task<Qwen3Asr> FromPretrainedAsync(
        string modelPath,
        DeviceType device = DeviceType.Cpu,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentNullException(nameof(modelPath));

        return await Task.Run(() =>
        {
            var options = new LoadOptions(modelPath) { Device = device };
            return Create(options);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Creates a new Qwen3Asr instance with the specified options.
    /// </summary>
    /// <param name="options">The load options.</param>
    /// <returns>A new Qwen3Asr instance.</returns>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="Qwen3AsrException">Thrown when model loading fails.</exception>
    public static Qwen3Asr Create(LoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (string.IsNullOrEmpty(options.ModelPath))
            throw new ArgumentException("ModelPath is required", nameof(options));

        var ffiOptions = new NativeBindings.LoadOptionsFFI
        {
            Device = options.Device,
            ModelPath = StringToPtr(options.ModelPath),
            Revision = StringToPtr(options.Revision),
            NumThreads = options.NumThreads
        };

        try
        {
            var handle = NativeBindings.qwen3_asr_create(ref ffiOptions, out var errorMsg);

            if (handle == IntPtr.Zero)
            {
                var error = PtrToString(errorMsg);
                NativeBindings.qwen3_asr_free_string(errorMsg);
                throw new Qwen3AsrException($"Failed to create ASR instance: {error}");
            }

            return new Qwen3Asr(handle, options.ModelPath, options.Device);
        }
        finally
        {
            FreeString(ffiOptions.ModelPath);
            FreeString(ffiOptions.Revision);
        }
    }

    /// <summary>
    /// Transcribes audio from a file.
    /// </summary>
    /// <param name="filePath">Path to the audio file.</param>
    /// <param name="language">Optional language code (e.g., "Japanese", "English"). If null, auto-detects.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The transcription result.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when this instance is disposed.</exception>
    /// <exception cref="Qwen3AsrException">Thrown when transcription fails.</exception>
    public async Task<TranscriptionResult> TranscribeFileAsync(
        string filePath,
        string? language = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentNullException(nameof(filePath));

        return await Task.Run(() =>
        {
            var filePathPtr = StringToPtr(filePath);
            var languagePtr = StringToPtr(language);

            try
            {
                var result = NativeBindings.qwen3_asr_transcribe_file(_handle, filePathPtr, languagePtr);
                return ProcessResult(result);
            }
            finally
            {
                FreeString(filePathPtr);
                FreeString(languagePtr);
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Transcribes audio samples.
    /// </summary>
    /// <param name="audioSamples">Audio samples as 32-bit floats (16kHz, mono).</param>
    /// <param name="language">Optional language code (e.g., "Japanese", "English"). If null, auto-detects.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The transcription result.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when this instance is disposed.</exception>
    /// <exception cref="Qwen3AsrException">Thrown when transcription fails.</exception>
    public async Task<TranscriptionResult> TranscribeAsync(
        float[] audioSamples,
        string? language = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(audioSamples);

        return await Task.Run(() =>
        {
            var languagePtr = StringToPtr(language);

            try
            {
                var result = NativeBindings.qwen3_asr_transcribe(
                    _handle,
                    audioSamples,
                    (UIntPtr)audioSamples.Length,
                    languagePtr);
                return ProcessResult(result);
            }
            finally
            {
                FreeString(languagePtr);
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Starts a new streaming transcription session.
    /// </summary>
    /// <param name="options">Stream options. If null, default options are used.</param>
    /// <returns>A new streaming transcriber.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when this instance is disposed.</exception>
    /// <exception cref="Qwen3AsrException">Thrown when starting the stream fails.</exception>
    public StreamingTranscriber StartStream(StreamOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        options ??= new StreamOptions();

        var ffiOptions = new NativeBindings.StreamOptionsFFI
        {
            Language = StringToPtr(options.Language),
            ChunkSizeSec = options.ChunkSizeSec,
            EnableTimestamps = options.EnableTimestamps,
            EnablePartialResults = options.EnablePartialResults
        };

        try
        {
            var streamHandle = NativeBindings.qwen3_asr_stream_start(_handle, ref ffiOptions);

            if (streamHandle == IntPtr.Zero)
            {
                throw new Qwen3AsrException("Failed to start streaming transcription");
            }

            return new StreamingTranscriber(streamHandle, options);
        }
        finally
        {
            FreeString(ffiOptions.Language);
        }
    }

    /// <summary>
    /// Gets the library version.
    /// </summary>
    /// <returns>The version string.</returns>
    public static string GetVersion()
    {
        var versionPtr = NativeBindings.qwen3_asr_version();
        var version = PtrToString(versionPtr);
        NativeBindings.qwen3_asr_free_string(versionPtr);
        return version ?? "unknown";
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

    private static IntPtr StringToPtr(string? str)
    {
        if (string.IsNullOrEmpty(str))
            return IntPtr.Zero;

        return Marshal.StringToHGlobalAnsi(str);
    }

    private static string? PtrToString(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero)
            return null;

        return Marshal.PtrToStringAnsi(ptr);
    }

    private static void FreeString(IntPtr ptr)
    {
        if (ptr != IntPtr.Zero)
            Marshal.FreeHGlobal(ptr);
    }

    /// <summary>
    /// Disposes this instance and releases all resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        if (_handle != IntPtr.Zero)
        {
            NativeBindings.qwen3_asr_destroy(_handle);
            _handle = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer.
    /// </summary>
    ~Qwen3Asr()
    {
        Dispose();
    }
}
