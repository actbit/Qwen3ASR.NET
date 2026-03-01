namespace Qwen3ASR.NET;

/// <summary>
/// Error codes for Qwen3-ASR operations.
/// </summary>
public enum ErrorCode
{
    /// <summary>Operation successful</summary>
    Success = 0,
    /// <summary>Invalid handle</summary>
    InvalidHandle = 1,
    /// <summary>Invalid parameter</summary>
    InvalidParameter = 2,
    /// <summary>Model not loaded</summary>
    ModelNotLoaded = 3,
    /// <summary>Inference error</summary>
    InferenceError = 4,
    /// <summary>Memory allocation error</summary>
    MemoryError = 5,
    /// <summary>Streaming session error</summary>
    StreamError = 6,
    /// <summary>Unknown error</summary>
    UnknownError = 99
}

/// <summary>
/// Exception thrown when a Qwen3-ASR operation fails.
/// </summary>
public class Qwen3AsrException : Exception
{
    /// <summary>
    /// Gets the error code associated with this exception.
    /// </summary>
    public ErrorCode ErrorCode { get; }

    /// <summary>
    /// Creates a new instance of Qwen3AsrException.
    /// </summary>
    /// <param name="message">The error message.</param>
    public Qwen3AsrException(string message)
        : base(message)
    {
        ErrorCode = ErrorCode.UnknownError;
    }

    /// <summary>
    /// Creates a new instance of Qwen3AsrException with an error code.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="errorCode">The error code.</param>
    public Qwen3AsrException(string message, ErrorCode errorCode)
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
        ErrorCode = ErrorCode.UnknownError;
    }
}
