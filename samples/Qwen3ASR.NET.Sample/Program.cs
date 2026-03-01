using Qwen3ASR.NET;
using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;

namespace Qwen3ASR.NET.Sample;

class Program
{
    static async Task<int> Main(string[] args)
    {
        Console.WriteLine("Qwen3ASR.NET Sample Application");
        Console.WriteLine($"Library Version: {Qwen3Asr.GetVersion()}");
        Console.WriteLine();

        if (args.Length == 0)
        {
            Console.WriteLine("Usage: Qwen3ASR.NET.Sample <audio-file> [model-path] [language]");
            Console.WriteLine();
            Console.WriteLine("Arguments:");
            Console.WriteLine("  audio-file  - Path to the audio file to transcribe");
            Console.WriteLine("  model-path  - Optional: Model path or HuggingFace model ID (default: Qwen/Qwen3-ASR-0.6B)");
            Console.WriteLine("  language    - Optional: Language code (e.g., Japanese, English, Chinese)");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  Qwen3ASR.NET.Sample audio.wav");
            Console.WriteLine("  Qwen3ASR.NET.Sample audio.wav Qwen/Qwen3-ASR-0.6B Japanese");
            return 1;
        }

        var audioFile = args[0];
        var modelPath = args.Length > 1 ? args[1] : "Qwen/Qwen3-ASR-0.6B";
        var language = args.Length > 2 ? args[2] : null;

        if (!File.Exists(audioFile))
        {
            Console.WriteLine($"Error: Audio file not found: {audioFile}");
            return 1;
        }

        try
        {
            Console.WriteLine($"Loading model: {modelPath}");
            Console.WriteLine("This may take a while for the first run...");

            using var asr = await Qwen3Asr.FromPretrainedAsync(modelPath, DeviceType.Cpu);
            Console.WriteLine("Model loaded successfully!");
            Console.WriteLine();

            Console.WriteLine($"Transcribing: {audioFile}");
            if (!string.IsNullOrEmpty(language))
            {
                Console.WriteLine($"Language: {language}");
            }
            Console.WriteLine();

            var result = await asr.TranscribeFileAsync(audioFile, language);

            Console.WriteLine("Transcription Result:");
            Console.WriteLine("-------------------");
            Console.WriteLine(result.Text);
            Console.WriteLine();

            if (result.Timestamps != null && result.Timestamps.Count > 0)
            {
                Console.WriteLine("Timestamps:");
                foreach (var ts in result.Timestamps)
                {
                    Console.WriteLine($"  [{ts.Start:F2}s - {ts.End:F2}s] {ts.Text}");
                }
            }

            // Streaming transcription example
            Console.WriteLine();
            Console.WriteLine("Streaming Transcription Example:");
            Console.WriteLine("--------------------------------");

            await using var stream = asr.StartStream(new StreamOptions
            {
                Language = language,
                ChunkSizeSec = 0.5f,
                EnableTimestamps = true
            });

            // In a real application, you would push audio chunks as they arrive
            // For demonstration, we'll just show the API usage
            Console.WriteLine("(Streaming would process audio chunks in real-time)");

            var finalResult = await stream.FinishAsync();
            Console.WriteLine($"Final result: {finalResult.Text}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }

        return 0;
    }
}
