using System.Runtime.InteropServices;
using System.Text.Json;
using Qwen3ASR.NET.Audio;
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
    public static async Task<Qwen3Asr> FromPretrainedAsync(
        string modelPath,
        DeviceType device = DeviceType.Cpu,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentNullException(nameof(modelPath));

        return await Task.Run(() =>
        {
            var options = new LoadOptions { ModelPath = modelPath, Device = device };
            return Create(options);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Creates a new Qwen3Asr instance with the specified options.
    /// </summary>
    /// <param name="options">The load options.</param>
    /// <returns>A new Qwen3Asr instance.</returns>
    public static Qwen3Asr Create(LoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (string.IsNullOrEmpty(options.ModelPath))
            throw new ArgumentException("ModelPath is required", nameof(options));

        var modelPathPtr = StringToPtr(options.ModelPath);
        var deviceType = (NativeBindings.DeviceTypeFFI)options.Device;

        try
        {
            var handle = NativeBindings.qwen3_asr_load(modelPathPtr, deviceType, out var errorMsg);

            if (handle == IntPtr.Zero)
            {
                var error = PtrToString(errorMsg);
                NativeBindings.qwen3_asr_free_string(errorMsg);
                throw new Qwen3AsrException($"Failed to load model: {error}");
            }

            return new Qwen3Asr(handle, options.ModelPath, options.Device);
        }
        finally
        {
            FreeString(modelPathPtr);
        }
    }

    /// <summary>
    /// Transcribes audio from a WAV file.
    /// </summary>
    /// <param name="filePath">Path to the WAV file.</param>
    /// <param name="options">Transcription options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The transcription result.</returns>
    public async Task<TranscriptionResult> TranscribeFileAsync(
        string filePath,
        TranscriptionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentNullException(nameof(filePath));

        // Use native file reading
        return await Task.Run(() =>
        {
            var filePathPtr = StringToPtr(filePath);
            var opts = BuildTranscribeOptions(options);

            try
            {
                var result = NativeBindings.qwen3_asr_transcribe_file(_handle, filePathPtr, ref opts);
                return ProcessResult(result);
            }
            finally
            {
                FreeString(filePathPtr);
                FreeTranscribeOptions(ref opts);
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Transcribes audio samples (f32, mono).
    /// </summary>
    /// <param name="samples">Audio samples as 32-bit floats.</param>
    /// <param name="sampleRate">Sample rate of the audio. Will be resampled to 16kHz if needed.</param>
    /// <param name="options">Transcription options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The transcription result.</returns>
    public async Task<TranscriptionResult> TranscribeAsync(
        float[] samples,
        int sampleRate = 16000,
        TranscriptionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(samples);

        return await Task.Run(() =>
        {
            var opts = BuildTranscribeOptions(options);

            try
            {
                var result = NativeBindings.qwen3_asr_transcribe(
                    _handle,
                    samples,
                    (UIntPtr)samples.Length,
                    (uint)sampleRate,
                    ref opts);
                return ProcessResult(result);
            }
            finally
            {
                FreeTranscribeOptions(ref opts);
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Transcribes audio from a stream containing WAV data.
    /// </summary>
    /// <param name="stream">Stream containing WAV file data.</param>
    /// <param name="options">Transcription options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The transcription result.</returns>
    public async Task<TranscriptionResult> TranscribeAsync(
        Stream stream,
        TranscriptionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(stream);

        // Read and parse WAV in .NET (cross-platform)
        var wavData = await Task.Run(() => WavReader.ReadWav(stream), cancellationToken).ConfigureAwait(false);
        return await TranscribeAsync(wavData.Samples, (int)wavData.SampleRate, options, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Transcribes multiple audio files in batch.
    /// </summary>
    /// <param name="filePaths">Paths to the audio files.</param>
    /// <param name="options">Transcription options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of transcription results.</returns>
    public async Task<List<TranscriptionResult>> TranscribeBatchAsync(
        IEnumerable<string> filePaths,
        TranscriptionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(filePaths);

        var results = new List<TranscriptionResult>();

        foreach (var filePath in filePaths)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var result = await TranscribeFileAsync(filePath, options, cancellationToken).ConfigureAwait(false);
            results.Add(result);
        }

        return results;
    }

    /// <summary>
    /// Transcribes multiple audio samples in batch.
    /// </summary>
    /// <param name="samplesList">List of audio samples (each as 32-bit floats at 16kHz mono).</param>
    /// <param name="options">Transcription options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of transcription results.</returns>
    public async Task<List<TranscriptionResult>> TranscribeBatchAsync(
        IEnumerable<float[]> samplesList,
        TranscriptionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(samplesList);

        var results = new List<TranscriptionResult>();

        foreach (var samples in samplesList)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var result = await TranscribeAsync(samples, 16000, options, cancellationToken).ConfigureAwait(false);
            results.Add(result);
        }

        return results;
    }

    /// <summary>
    /// Transcribes audio from byte array containing WAV data.
    /// </summary>
    /// <param name="wavBytes">Byte array containing WAV file data.</param>
    /// <param name="options">Transcription options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The transcription result.</returns>
    public async Task<TranscriptionResult> TranscribeWavBytesAsync(
        byte[] wavBytes,
        TranscriptionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(wavBytes);

        // Parse WAV in .NET (cross-platform)
        var wavData = await Task.Run(() => WavReader.ReadWav(wavBytes), cancellationToken).ConfigureAwait(false);
        return await TranscribeAsync(wavData.Samples, (int)wavData.SampleRate, options, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Starts a new streaming transcription session.
    /// </summary>
    /// <param name="options">Stream options. If null, default options are used.</param>
    /// <returns>A new streaming transcriber.</returns>
    public StreamingTranscriber StartStream(StreamOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        options ??= StreamOptions.Default;
        return new StreamingTranscriber(this, options);
    }

    /// <summary>
    /// Transcribes an audio file with streaming callbacks.
    /// This allows processing large files with partial results.
    /// </summary>
    /// <param name="filePath">Path to the audio file (WAV format supported).</param>
    /// <param name="onPartialResult">Callback for partial transcription results.</param>
    /// <param name="options">Stream options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The final transcription result.</returns>
    public async Task<TranscriptionResult> TranscribeFileStreamAsync(
        string filePath,
        Action<TranscriptionResult>? onPartialResult = null,
        StreamOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentNullException(nameof(filePath));

        var wavData = await Task.Run(() => WavReader.ReadWav(filePath), cancellationToken).ConfigureAwait(false);
        return await TranscribeSamplesStreamAsync(wavData.Samples, onPartialResult, options, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Transcribes audio samples with streaming callbacks.
    /// This allows processing large audio with partial results.
    /// </summary>
    /// <param name="samples">Audio samples as 32-bit floats (16kHz, mono).</param>
    /// <param name="onPartialResult">Callback for partial transcription results.</param>
    /// <param name="options">Stream options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The final transcription result.</returns>
    public async Task<TranscriptionResult> TranscribeSamplesStreamAsync(
        float[] samples,
        Action<TranscriptionResult>? onPartialResult = null,
        StreamOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(samples);

        options ??= StreamOptions.Default;

        using var stream = StartStream(options);
        var chunkSizeSamples = (int)(options.ChunkSizeSec * 16000);

        for (int i = 0; i < samples.Length; i += chunkSizeSamples)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int remaining = Math.Min(chunkSizeSamples, samples.Length - i);
            var chunk = new float[remaining];
            Array.Copy(samples, i, chunk, 0, remaining);

            var partialResult = await stream.PushAsync(chunk, cancellationToken).ConfigureAwait(false);
            onPartialResult?.Invoke(partialResult);
        }

        return await stream.FinishAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Transcribes audio from a stream with streaming callbacks.
    /// </summary>
    /// <param name="stream">Stream containing WAV data.</param>
    /// <param name="onPartialResult">Callback for partial transcription results.</param>
    /// <param name="options">Stream options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The final transcription result.</returns>
    public async Task<TranscriptionResult> TranscribeStreamAsync(
        Stream stream,
        Action<TranscriptionResult>? onPartialResult = null,
        StreamOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(stream);

        var wavData = await Task.Run(() => WavReader.ReadWav(stream), cancellationToken).ConfigureAwait(false);
        return await TranscribeSamplesStreamAsync(wavData.Samples, onPartialResult, options, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the supported languages as an array.
    /// </summary>
    public async Task<string[]> GetSupportedLanguagesAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        return await Task.Run(() =>
        {
            var jsonPtr = NativeBindings.qwen3_asr_supported_languages(_handle);
            var json = PtrToString(jsonPtr);
            NativeBindings.qwen3_asr_free_string(jsonPtr);

            if (string.IsNullOrEmpty(json))
                return Array.Empty<string>();

            try
            {
                return JsonSerializer.Deserialize<string[]>(json) ?? Array.Empty<string>();
            }
            catch
            {
                return Array.Empty<string>();
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the library version.
    /// </summary>
    public static string GetVersion()
    {
        var versionPtr = NativeBindings.qwen3_asr_version();
        var version = PtrToString(versionPtr);
        NativeBindings.qwen3_asr_free_string(versionPtr);
        return version ?? "unknown";
    }

    // Internal method for streaming transcription
    internal TranscriptionResult TranscribeInternal(float[] samples, TranscriptionOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var opts = BuildTranscribeOptions(options);

        try
        {
            var result = NativeBindings.qwen3_asr_transcribe(
                _handle,
                samples,
                (UIntPtr)samples.Length,
                16000,
                ref opts);
            return ProcessResult(result);
        }
        finally
        {
            FreeTranscribeOptions(ref opts);
        }
    }

    private static NativeBindings.TranscribeOptionsFFI BuildTranscribeOptions(TranscriptionOptions? options)
    {
        options ??= TranscriptionOptions.Default;

        return new NativeBindings.TranscribeOptionsFFI
        {
            Context = StringToPtr(options.Context),
            Language = StringToPtr(options.Language.ToNativeString()),
            ReturnTimestamps = options.ReturnTimestamps,
            MaxNewTokens = options.MaxNewTokens,
            MaxBatchSize = options.MaxBatchSize,
            ChunkMaxSec = options.ChunkMaxSec ?? 0f,
            BucketByLength = options.BucketByLength
        };
    }

    private static void FreeTranscribeOptions(ref NativeBindings.TranscribeOptionsFFI opts)
    {
        FreeString(opts.Context);
        FreeString(opts.Language);
    }

    private static TranscriptionResult ProcessResult(NativeBindings.TranscriptionResultFFI ffiResult)
    {
        try
        {
            if (ffiResult.Code != NativeBindings.ResultCode.Success)
            {
                var error = PtrToString(ffiResult.Error) ?? "Unknown error";
                throw new Qwen3AsrException(error, ToErrorCode(ffiResult.Code));
            }

            var json = PtrToString(ffiResult.Json);
            if (!string.IsNullOrEmpty(json))
            {
                try
                {
                    var result = JsonSerializer.Deserialize<TranscriptionResult>(json);
                    if (result != null)
                        return result;
                }
                catch (JsonException)
                {
                    // Fall through
                }
            }

            return new TranscriptionResult { Text = string.Empty };
        }
        finally
        {
            NativeBindings.qwen3_asr_free_result(ref ffiResult);
        }
    }

    private static ErrorCode ToErrorCode(NativeBindings.ResultCode code) => code switch
    {
        NativeBindings.ResultCode.Success => ErrorCode.Success,
        NativeBindings.ResultCode.InvalidHandle => ErrorCode.InvalidHandle,
        NativeBindings.ResultCode.InvalidParameter => ErrorCode.InvalidParameter,
        NativeBindings.ResultCode.ModelNotLoaded => ErrorCode.ModelNotLoaded,
        NativeBindings.ResultCode.InferenceError => ErrorCode.InferenceError,
        NativeBindings.ResultCode.MemoryError => ErrorCode.MemoryError,
        _ => ErrorCode.UnknownError
    };

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

        return Marshal.PtrToStringUTF8(ptr);
    }

    private static void FreeString(IntPtr ptr)
    {
        if (ptr != IntPtr.Zero)
            Marshal.FreeHGlobal(ptr);
    }

    /// <summary>
    /// Releases all resources used by the Qwen3Asr instance.
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
    /// Finalizer to ensure unmanaged resources are released.
    /// </summary>
    ~Qwen3Asr()
    {
        Dispose();
    }
}
