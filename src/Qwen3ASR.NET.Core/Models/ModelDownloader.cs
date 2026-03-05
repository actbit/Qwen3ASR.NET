using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Qwen3ASR.NET.Models;

/// <summary>
/// HuggingFace model downloader with caching support.
/// </summary>
public static class ModelDownloader
{
    private static readonly HttpClient HttpClient = new();
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
            // Get model info
            var modelInfo = await GetModelInfoAsync(modelId, cancellationToken);

            // Download required files (Qwen3-ASR uses vocab.json and merges.txt instead of tokenizer.json)
            var requiredFiles = new[] { "config.json", "model.safetensors", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json", "generation_config.json" };
            var totalFiles = requiredFiles.Length;
            var completedFiles = 0;

            foreach (var file in requiredFiles)
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

        // Check for essential files (Qwen3-ASR specific)
        var requiredFiles = new[] { "config.json", "model.safetensors", "vocab.json", "merges.txt" };
        return requiredFiles.All(f => File.Exists(Path.Combine(modelDir, f)));
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
