using Qwen3ASR.NET.Native;

namespace Qwen3ASR.NET;

/// <summary>
/// Exception thrown when a Qwen3-ASR operation fails.
/// </summary>
public class Qwen3AsrException : Exception
{
    /// <summary>
    /// Gets the error code associated with this exception.
    /// </summary>
    public NativeBindings.ResultCode ErrorCode { get; }

    /// <summary>
    /// Creates a new instance of Qwen3AsrException.
    /// </summary>
    /// <param name="message">The error message.</param>
    public Qwen3AsrException(string message)
        : base(message)
    {
        ErrorCode = NativeBindings.ResultCode.UnknownError;
    }

    /// <summary>
    /// Creates a new instance of Qwen3AsrException with an error code.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="errorCode">The error code.</param>
    public Qwen3AsrException(string message, NativeBindings.ResultCode errorCode)
        : base(message)
    {
        ErrorCode = errorCode;
    }

    /// <summary>
    /// Creates a new instance of Qwen3AsrException with an inner exception.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public Qwen3AsrException(string message, Exception innerException)
        : base(message, innerException)
    {
        ErrorCode = NativeBindings.ResultCode.UnknownError;
    }
}
