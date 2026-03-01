using System.Runtime.InteropServices;
using Qwen3ASR.NET.Enums;

namespace Qwen3ASR.NET.Native;

/// <summary>
/// Native FFI bindings for qwen3_asr_ffi library.
/// </summary>
internal static class NativeBindings
{
    private const string DllName = "qwen3_asr_ffi";

    #region FFI Structures

    /// <summary>
    /// Result codes for FFI functions.
    /// </summary>
    public enum ResultCode
    {
        Success = 0,
        InvalidHandle = 1,
        InvalidParameter = 2,
        ModelNotLoaded = 3,
        InferenceError = 4,
        MemoryError = 5,
        StreamError = 6,
        UnknownError = 99
    }

    /// <summary>
    /// Options for loading a model (FFI representation).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct LoadOptionsFFI
    {
        public DeviceType Device;
        public IntPtr ModelPath;
        public IntPtr Revision;
        public int NumThreads;
    }

    /// <summary>
    /// Options for streaming transcription (FFI representation).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct StreamOptionsFFI
    {
        public IntPtr Language;
        public float ChunkSizeSec;
        [MarshalAs(UnmanagedType.U1)]
        public bool EnableTimestamps;
        [MarshalAs(UnmanagedType.U1)]
        public bool EnablePartialResults;
    }

    /// <summary>
    /// Transcription result (FFI representation).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct TranscriptionResultFFI
    {
        public IntPtr Text;
        public IntPtr JsonResult;
        public ResultCode Code;
        public IntPtr ErrorMessage;
    }

    #endregion

    #region FFI Functions

    /// <summary>
    /// Create a new ASR instance.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr qwen3_asr_create(
        ref LoadOptionsFFI options,
        out IntPtr errorMessage);

    /// <summary>
    /// Transcribe audio data (offline/batch mode).
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern TranscriptionResultFFI qwen3_asr_transcribe(
        IntPtr handle,
        [In] float[] audioData,
        UIntPtr len,
        IntPtr language);

    /// <summary>
    /// Transcribe audio from a file.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern TranscriptionResultFFI qwen3_asr_transcribe_file(
        IntPtr handle,
        IntPtr filePath,
        IntPtr language);

    /// <summary>
    /// Start a streaming transcription session.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr qwen3_asr_stream_start(
        IntPtr handle,
        ref StreamOptionsFFI options);

    /// <summary>
    /// Push audio chunk to streaming session.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern TranscriptionResultFFI qwen3_asr_stream_push(
        IntPtr streamHandle,
        [In] float[] audioData,
        UIntPtr len);

    /// <summary>
    /// Get partial result from streaming session.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern TranscriptionResultFFI qwen3_asr_stream_get_partial(
        IntPtr streamHandle);

    /// <summary>
    /// Finish streaming session and get final result.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern TranscriptionResultFFI qwen3_asr_stream_finish(
        IntPtr streamHandle);

    /// <summary>
    /// Free a transcription result.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void qwen3_asr_free_result(ref TranscriptionResultFFI result);

    /// <summary>
    /// Free a string returned by FFI.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void qwen3_asr_free_string(IntPtr ptr);

    /// <summary>
    /// Destroy an ASR instance.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void qwen3_asr_destroy(IntPtr handle);

    /// <summary>
    /// Get the library version string.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr qwen3_asr_version();

    #endregion
}
