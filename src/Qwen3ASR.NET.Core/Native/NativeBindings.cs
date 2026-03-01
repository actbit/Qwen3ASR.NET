using System.Runtime.InteropServices;

namespace Qwen3ASR.NET.Native;

/// <summary>
/// Native FFI bindings for qwen3_asr_ffi library.
/// </summary>
internal static class NativeBindings
{
    private const string DllName = "qwen3_asr_ffi";

    #region FFI Structures

    internal enum ResultCode
    {
        Success = 0,
        InvalidHandle = 1,
        InvalidParameter = 2,
        ModelNotLoaded = 3,
        InferenceError = 4,
        MemoryError = 5,
        UnknownError = 99
    }

    internal enum DeviceTypeFFI
    {
        Cpu = 0,
        Cuda = 1,
        Metal = 2
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct TranscribeOptionsFFI
    {
        public IntPtr Context;
        public IntPtr Language;
        [MarshalAs(UnmanagedType.U1)]
        public bool ReturnTimestamps;
        public int MaxNewTokens;
        public int MaxBatchSize;
        public float ChunkMaxSec;
        [MarshalAs(UnmanagedType.U1)]
        public bool BucketByLength;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct TranscriptionResultFFI
    {
        public IntPtr Json;
        public ResultCode Code;
        public IntPtr Error;
    }

    #endregion

    #region FFI Functions - Model Management

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr qwen3_asr_load(
        IntPtr modelPath,
        DeviceTypeFFI device,
        out IntPtr errorMsg);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void qwen3_asr_destroy(IntPtr handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.U1)]
    internal static extern bool qwen3_asr_is_loaded(IntPtr handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr qwen3_asr_supported_languages(IntPtr handle);

    #endregion

    #region FFI Functions - Transcription

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern TranscriptionResultFFI qwen3_asr_transcribe(
        IntPtr handle,
        [In] float[] samples,
        UIntPtr sampleCount,
        uint sampleRate,
        ref TranscribeOptionsFFI options);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern TranscriptionResultFFI qwen3_asr_transcribe_file(
        IntPtr handle,
        IntPtr filePath,
        ref TranscribeOptionsFFI options);

    #endregion

    #region FFI Functions - Memory Management

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void qwen3_asr_free_result(ref TranscriptionResultFFI result);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void qwen3_asr_free_string(IntPtr ptr);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr qwen3_asr_version();

    #endregion
}
