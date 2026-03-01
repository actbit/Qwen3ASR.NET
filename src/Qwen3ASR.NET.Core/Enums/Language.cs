namespace Qwen3ASR.NET.Enums;

/// <summary>
/// Supported languages for transcription.
/// </summary>
public enum Language
{
    /// <summary>
    /// Auto-detect language
    /// </summary>
    Auto = 0,

    /// <summary>
    /// Japanese
    /// </summary>
    Japanese,

    /// <summary>
    /// English
    /// </summary>
    English,

    /// <summary>
    /// Chinese (Mandarin)
    /// </summary>
    Chinese,

    /// <summary>
    /// Korean
    /// </summary>
    Korean,

    /// <summary>
    /// French
    /// </summary>
    French,

    /// <summary>
    /// German
    /// </summary>
    German,

    /// <summary>
    /// Spanish
    /// </summary>
    Spanish,

    /// <summary>
    /// Russian
    /// </summary>
    Russian,

    /// <summary>
    /// Portuguese
    /// </summary>
    Portuguese,

    /// <summary>
    /// Italian
    /// </summary>
    Italian,

    /// <summary>
    /// Dutch
    /// </summary>
    Dutch,

    /// <summary>
    /// Arabic
    /// </summary>
    Arabic,

    /// <summary>
    /// Hindi
    /// </summary>
    Hindi,

    /// <summary>
    /// Thai
    /// </summary>
    Thai,

    /// <summary>
    /// Vietnamese
    /// </summary>
    Vietnamese,

    /// <summary>
    /// Indonesian
    /// </summary>
    Indonesian,

    /// <summary>
    /// Turkish
    /// </summary>
    Turkish,

    /// <summary>
    /// Polish
    /// </summary>
    Polish,

    /// <summary>
    /// Swedish
    /// </summary>
    Swedish,

    /// <summary>
    /// Czech
    /// </summary>
    Czech
}

/// <summary>
/// Extension methods for Language enum.
/// </summary>
internal static class LanguageExtensions
{
    /// <summary>
    /// Converts Language enum to the string format expected by the native library.
    /// </summary>
    public static string? ToNativeString(this Language language) => language switch
    {
        Language.Auto => null,
        Language.Japanese => "Japanese",
        Language.English => "English",
        Language.Chinese => "Chinese",
        Language.Korean => "Korean",
        Language.French => "French",
        Language.German => "German",
        Language.Spanish => "Spanish",
        Language.Russian => "Russian",
        Language.Portuguese => "Portuguese",
        Language.Italian => "Italian",
        Language.Dutch => "Dutch",
        Language.Arabic => "Arabic",
        Language.Hindi => "Hindi",
        Language.Thai => "Thai",
        Language.Vietnamese => "Vietnamese",
        Language.Indonesian => "Indonesian",
        Language.Turkish => "Turkish",
        Language.Polish => "Polish",
        Language.Swedish => "Swedish",
        Language.Czech => "Czech",
        _ => null
    };
}
