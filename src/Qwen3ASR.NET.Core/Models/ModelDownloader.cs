using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace Qwen3ASR.NET.Models;

/// <summary>
/// HuggingFace model downloader with caching support.
/// </summary>
public static class ModelDownloader
{
    private static readonly HttpClient HttpClient = new(new HttpClientHandler
    {
        AllowAutoRedirect = true,
        MaxAutomaticRedirections = 10,
        AutomaticDecompression = System.Net.DecompressionMethods.GZip | System.Net.DecompressionMethods.Deflate
    });
    private const string HuggingFaceApiBase = "https://huggingface.co/api";

    /// <summary>
    /// Default ASR model ID.
    /// </summary>
    public const string DefaultAsrModel = "Qwen/Qwen3-ASR-0.6B";

    /// <summary>
    /// Default forced aligner model ID for timestamp prediction.
    /// </summary>
    public const string DefaultForcedAlignerModel = "Qwen/Qwen3-ForcedAligner-0.6B";

    /// <summary>
    /// Default cache directory for downloaded models.
    /// </summary>
    public static string DefaultCacheDirectory =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "qwen3asr");

    // Required file patterns for Qwen3-ASR models
    private static readonly string[] RequiredFilePatterns = new[]
    {
        "^config\\.json$",
        "^model\\.safetensors$",
        "^model\\.safetensors\\.index\\.json$",  // Sharded model index
        "^model-\\d+-of-\\d+\\.safetensors$",   // Sharded model files
        "^tokenizer_config\\.json$",
        "^vocab\\.json$",
        "^merges\\.txt$",
        "^preprocessor_config\\.json$",
        "^generation_config\\.json$",
        "^tokenizer\\.json$"  // Some models use this instead
    };

    /// <summary>
    /// Downloads a model from HuggingFace if not already cached.
    /// </summary>
    /// <param name="modelId">HuggingFace model ID (e.g., "Qwen/Qwen2-Audio-7B-Instruct").</param>
    /// <param name="cacheDirectory">Optional custom cache directory.</param>
    /// <param name="progress">Optional progress callback (0.0 - 1.0).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Path to the cached model directory.</returns>
    public static async Task<string> DownloadModelAsync(
        string modelId,
        string? cacheDirectory = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelId);

        cacheDirectory ??= DefaultCacheDirectory;
        var modelDir = Path.Combine(cacheDirectory, modelId.Replace('/', Path.DirectorySeparatorChar));

        // Check if already cached
        if (IsModelCached(modelDir))
        {
            progress?.Report(1.0);
            return modelDir;
        }

        // Create directory
        Directory.CreateDirectory(modelDir);

        try
        {
            // Get model info with file list from HuggingFace API
            var modelInfo = await GetModelInfoAsync(modelId, cancellationToken);
            var availableFiles = modelInfo.Siblings?.Select(s => s.Filename).Where(f => f != null).ToList()
                ?? new List<string?>();

            // Determine which files to download based on patterns
            var filesToDownload = GetRequiredFiles(availableFiles!);
            var totalFiles = filesToDownload.Count;
            var completedFiles = 0;

            foreach (var file in filesToDownload)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var filePath = Path.Combine(modelDir, file);
                if (!File.Exists(filePath))
                {
                    await DownloadFileAsync(modelId, file, filePath, cancellationToken);
                }

                completedFiles++;
                progress?.Report((double)completedFiles / totalFiles);
            }

            return modelDir;
        }
        catch
        {
            // Clean up on failure
            if (Directory.Exists(modelDir))
            {
                try { Directory.Delete(modelDir, recursive: true); } catch { }
            }
            throw;
        }
    }

    /// <summary>
    /// Gets the list of required files from available files based on patterns.
    /// </summary>
    private static List<string> GetRequiredFiles(List<string> availableFiles)
    {
        var result = new List<string>();
        var compiledPatterns = RequiredFilePatterns.Select(p => new Regex(p, RegexOptions.Compiled)).ToList();

        foreach (var pattern in compiledPatterns)
        {
            var matches = availableFiles.Where(f => pattern.IsMatch(f)).ToList();
            result.AddRange(matches);
        }

        // Remove duplicates and return
        return result.Distinct().ToList();
    }

    /// <summary>
    /// Checks if a model is already cached locally.
    /// </summary>
    /// <param name="modelId">HuggingFace model ID.</param>
    /// <param name="cacheDirectory">Optional custom cache directory.</param>
    /// <returns>True if the model is cached and complete.</returns>
    public static bool IsModelCached(string modelId, string? cacheDirectory = null)
    {
        cacheDirectory ??= DefaultCacheDirectory;
        var modelDir = Path.Combine(cacheDirectory, modelId.Replace('/', Path.DirectorySeparatorChar));
        return IsModelCached(modelDir);
    }

    /// <summary>
    /// Gets the local path for a cached model.
    /// </summary>
    /// <param name="modelId">HuggingFace model ID.</param>
    /// <param name="cacheDirectory">Optional custom cache directory.</param>
    /// <returns>Path to the cached model directory, or null if not cached.</returns>
    public static string? GetCachedModelPath(string modelId, string? cacheDirectory = null)
    {
        cacheDirectory ??= DefaultCacheDirectory;
        var modelDir = Path.Combine(cacheDirectory, modelId.Replace('/', Path.DirectorySeparatorChar));
        return IsModelCached(modelDir) ? modelDir : null;
    }

    /// <summary>
    /// Clears the model cache.
    /// </summary>
    /// <param name="cacheDirectory">Optional custom cache directory.</param>
    public static void ClearCache(string? cacheDirectory = null)
    {
        cacheDirectory ??= DefaultCacheDirectory;
        if (Directory.Exists(cacheDirectory))
        {
            Directory.Delete(cacheDirectory, recursive: true);
        }
    }

    private static bool IsModelCached(string modelDir)
    {
        if (!Directory.Exists(modelDir))
            return false;

        var files = Directory.GetFiles(modelDir).Select(Path.GetFileName).ToList();

        // Check for essential files
        bool hasConfig = files.Any(f => f == "config.json");
        bool hasModel = files.Any(f => f == "model.safetensors" || (f != null && f.StartsWith("model-") && f.EndsWith(".safetensors")));
        bool hasTokenizer = files.Any(f => f == "vocab.json" || f == "tokenizer.json");

        return hasConfig && hasModel && hasTokenizer;
    }

    private static async Task<ModelInfo> GetModelInfoAsync(string modelId, CancellationToken cancellationToken)
    {
        var url = $"{HuggingFaceApiBase}/models/{modelId}";
        var response = await HttpClient.GetAsync(url, cancellationToken);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        return JsonSerializer.Deserialize<ModelInfo>(json) ?? throw new InvalidOperationException("Failed to parse model info");
    }

    private static async Task DownloadFileAsync(string modelId, string filename, string destinationPath, CancellationToken cancellationToken)
    {
        var url = $"https://huggingface.co/{modelId}/resolve/main/{filename}";

        using var response = await HttpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        await using var fileStream = File.Create(destinationPath);
        await response.Content.CopyToAsync(fileStream, cancellationToken);
    }

    private class ModelInfo
    {
        [JsonPropertyName("modelId")]
        public string? ModelId { get; set; }

        [JsonPropertyName("siblings")]
        public List<ModelFile>? Siblings { get; set; }
    }

    private class ModelFile
    {
        [JsonPropertyName("rfilename")]
        public string? Filename { get; set; }
    }
}
