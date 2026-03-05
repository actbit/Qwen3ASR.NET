using System.Text;
using Pv;
using Qwen3ASR.NET;
using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;

namespace Qwen3ASR.NET.Realtime;

class Program
{
    static async Task<int> Main(string[] args)
    {
        // Set console encoding to UTF-8 for proper Japanese display
        Console.OutputEncoding = Encoding.UTF8;

        Console.WriteLine("=== Qwen3ASR.NET Real-time Transcription ===");
        Console.WriteLine($"Library Version: {Qwen3Asr.GetVersion()}");
        Console.WriteLine();

        // Parse arguments
        var modelPath = "Qwen/Qwen3-ASR-0.6B";
        string? forcedAlignerPath = null;
        Language? language = null;
        int? deviceIndex = null;
        bool useVad = true;  // VAD enabled by default
        float vadThreshold = 0.01f;
        DeviceType deviceType = DeviceType.Cpu;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "-m":
                case "--model":
                    modelPath = args[++i];
                    break;
                case "-a":
                case "--aligner":
                    forcedAlignerPath = args[++i];
                    break;
                case "-l":
                case "--language":
                    language = ParseLanguage(args[++i]);
                    break;
                case "-d":
                case "--device":
                    deviceIndex = int.Parse(args[++i]);
                    break;
                case "--no-vad":
                    useVad = false;
                    break;
                case "--vad-threshold":
                    vadThreshold = float.Parse(args[++i]);
                    break;
                case "--gpu":
                case "--cuda":
                    deviceType = DeviceType.Cuda;
                    break;
                case "--metal":
                    deviceType = DeviceType.Metal;
                    break;
                case "-h":
                case "--help":
                    PrintHelp();
                    return 0;
            }
        }

        try
        {
            // Load model
            Console.WriteLine($"Loading model: {modelPath}");
            if (!string.IsNullOrEmpty(forcedAlignerPath))
            {
                Console.WriteLine($"Forced Aligner: {forcedAlignerPath}");
            }
            Console.WriteLine($"Device: {deviceType}");
            Console.WriteLine("This may take a while for the first run (downloading model)...");
            Console.WriteLine();

            using var asr = await Qwen3Asr.FromPretrainedAsync(modelPath, forcedAlignerPath, deviceType);
            Console.WriteLine("Model loaded successfully!");
            Console.WriteLine();

            // List available audio devices
            ListAudioDevices();

            // Start real-time transcription
            await RunRealtimeTranscription(asr, language, deviceIndex, useVad, vadThreshold);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            return 1;
        }

        return 0;
    }

    static void PrintHelp()
    {
        Console.WriteLine("Usage: Qwen3ASR.NET.Realtime [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -m, --model <path>      Model path or HuggingFace ID (default: Qwen/Qwen3-ASR-0.6B)");
        Console.WriteLine("  -a, --aligner <path>    Forced aligner model for timestamps (e.g., Qwen/Qwen3-ForcedAligner-0.6B)");
        Console.WriteLine("  -l, --language <lang>   Language code (Japanese, English, Chinese, etc.)");
        Console.WriteLine("  -d, --device <index>    Audio input device index");
        Console.WriteLine("  --gpu, --cuda           Use CUDA GPU for inference (requires NVIDIA GPU)");
        Console.WriteLine("  --metal                 Use Metal GPU for inference (macOS only)");
        Console.WriteLine("  --no-vad                Disable Voice Activity Detection");
        Console.WriteLine("  --vad-threshold <val>   VAD energy threshold (default: 0.01)");
        Console.WriteLine("  -h, --help              Show this help message");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  Qwen3ASR.NET.Realtime");
        Console.WriteLine("  Qwen3ASR.NET.Realtime -l Japanese");
        Console.WriteLine("  Qwen3ASR.NET.Realtime -l English -d 1");
        Console.WriteLine("  Qwen3ASR.NET.Realtime --gpu");
        Console.WriteLine("  Qwen3ASR.NET.Realtime -a Qwen/Qwen3-ForcedAligner-0.6B");
        Console.WriteLine("  Qwen3ASR.NET.Realtime --no-vad");
    }

    static Language? ParseLanguage(string lang)
    {
        return lang.ToLowerInvariant() switch
        {
            "ja" or "japanese" => Language.Japanese,
            "en" or "english" => Language.English,
            "zh" or "chinese" => Language.Chinese,
            "ko" or "korean" => Language.Korean,
            "fr" or "french" => Language.French,
            "de" or "german" => Language.German,
            "es" or "spanish" => Language.Spanish,
            _ => Language.Auto
        };
    }

    static void ListAudioDevices()
    {
        Console.WriteLine("Available Audio Input Devices:");
        Console.WriteLine("-----------------------------");

        var devices = PvRecorder.GetAvailableDevices();
        for (int i = 0; i < devices.Length; i++)
        {
            Console.WriteLine($"  [{i}] {devices[i]}");
        }

        if (devices.Length == 0)
        {
            Console.WriteLine("  No audio input devices found!");
        }

        Console.WriteLine();
    }

    static async Task RunRealtimeTranscription(Qwen3Asr asr, Language? language, int? deviceIndex, bool useVad, float vadThreshold)
    {
        var cancellationTokenSource = new CancellationTokenSource();

        Console.WriteLine("Starting real-time transcription...");
        Console.WriteLine($"Language: {language?.ToString() ?? "Auto"}");
        Console.WriteLine($"VAD: {(useVad ? $"Enabled (threshold: {vadThreshold})" : "Disabled")}");
        Console.WriteLine();
        Console.WriteLine("Press ENTER to stop recording...");
        Console.WriteLine("-------------------------------------------");
        Console.WriteLine();

        // Audio parameters
        const int sampleRate = 16000;
        const int frameLength = 512; // PvRecorder frame size

        // Get device index
        int selectedDevice = deviceIndex ?? -1; // -1 = default device

        var devices = PvRecorder.GetAvailableDevices();
        if (selectedDevice >= 0 && selectedDevice < devices.Length)
        {
            Console.WriteLine($"Using device [{selectedDevice}]");
        }
        else
        {
            Console.WriteLine("Using default audio device");
        }
        Console.WriteLine();

        // Create VAD filter
        VadFilter? vad = useVad ? new VadFilter(sampleRate, vadThreshold, 300) : null;

        // Create PvRecorder
        using var recorder = PvRecorder.Create(frameLength: frameLength, deviceIndex: selectedDevice);

        // Start streaming transcription
        await using var stream = asr.StartStream(new StreamOptions
        {
            Language = language ?? Language.Auto,
            ChunkSizeSec = 1.0f,
            UnfixedChunkNum = 2,
            MaxNewTokens = 256
        });

        var audioBuffer = new List<float>();
        var bufferLock = new object();
        var lastPartialText = "";

        // Start recording
        recorder.Start();
        Console.WriteLine("Recording started...");
        Console.WriteLine();

        // Recording task
        var recordingTask = Task.Run(() =>
        {
            while (!cancellationTokenSource.Token.IsCancellationRequested)
            {
                try
                {
                    // Read frame (blocking)
                    short[] frame = recorder.Read();

                    // Convert short[] to float[] (normalize to -1.0 to 1.0)
                    float[] floatFrame = new float[frame.Length];
                    for (int i = 0; i < frame.Length; i++)
                    {
                        floatFrame[i] = frame[i] / 32768f;
                    }

                    // Apply VAD if enabled
                    if (vad != null)
                    {
                        vad.Process(floatFrame);
                        // VAD buffers speech internally
                    }
                    else
                    {
                        lock (bufferLock)
                        {
                            audioBuffer.AddRange(floatFrame);
                        }
                    }
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    // Ignore read errors during shutdown
                    break;
                }
            }
        }, cancellationTokenSource.Token);

        // Processing loop
        var processingTask = Task.Run(async () =>
        {
            // Faster real-time response with smaller chunks
            var chunkSize = (int)(sampleRate * 0.3f); // 0.3 second chunks
            var processInterval = TimeSpan.FromMilliseconds(100); // Check every 100ms

            while (!cancellationTokenSource.Token.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(processInterval, cancellationTokenSource.Token);

                    float[]? chunkToProcess = null;

                    if (vad != null)
                    {
                        // Get buffered audio (includes silence as zeros)
                        // Use effective buffer count to skip if all silence
                        var bufferedSamples = vad.GetEffectiveBufferCount();

                        // Send when we have enough samples with actual speech
                        if (bufferedSamples >= chunkSize)
                        {
                            // Get up to chunkSize samples
                            var allSamples = vad.PeekBuffer();
                            if (allSamples != null && allSamples.Length > 0)
                            {
                                var takeCount = Math.Min(chunkSize, allSamples.Length);
                                chunkToProcess = new float[takeCount];
                                Array.Copy(allSamples, chunkToProcess, takeCount);
                                vad.ConsumeSamples(takeCount);
                            }
                        }
                    }
                    else
                    {
                        lock (bufferLock)
                        {
                            if (audioBuffer.Count >= chunkSize)
                            {
                                chunkToProcess = audioBuffer.GetRange(0, chunkSize).ToArray();
                                audioBuffer.RemoveRange(0, chunkSize);
                            }
                        }
                    }

                    if (chunkToProcess == null || chunkToProcess.Length == 0)
                        continue;

                    // Push audio to streaming transcriber
                    var partial = await stream.PushAsync(chunkToProcess);

                    // Display partial result if changed
                    if (partial.Text != lastPartialText && !string.IsNullOrEmpty(partial.Text))
                    {
                        // Clear previous line and write new text
                        Console.Write($"\r{new string(' ', lastPartialText.Length)}\r{partial.Text}");
                        lastPartialText = partial.Text;
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }, cancellationTokenSource.Token);

        // Wait for user to press Enter
        Console.ReadLine();

        // Stop recording
        cancellationTokenSource.Cancel();
        recorder.Stop();

        try
        {
            await Task.WhenAll(recordingTask, processingTask);
        }
        catch (OperationCanceledException)
        {
            // Expected
        }

        // Get final result
        Console.WriteLine();
        Console.WriteLine();
        Console.WriteLine("Finalizing transcription...");

        // Flush remaining buffer before finishing
        if (vad != null)
        {
            Console.WriteLine($"VAD Stats: Silence = {vad.SilenceDuration.TotalSeconds:F1}s / Total = {vad.TotalDuration.TotalSeconds:F1}s");

            // Send any remaining buffered audio
            var remainingSamples = vad.Flush();
            if (remainingSamples != null && remainingSamples.Length > 0)
            {
                Console.WriteLine($"Flushing remaining {remainingSamples.Length / (float)sampleRate:F2}s of audio...");
                await stream.PushAsync(remainingSamples);
            }
        }
        else
        {
            float[]? remainingSamples = null;
            lock (bufferLock)
            {
                if (audioBuffer.Count > 0)
                {
                    remainingSamples = audioBuffer.ToArray();
                    audioBuffer.Clear();
                }
            }
            if (remainingSamples != null && remainingSamples.Length > 0)
            {
                Console.WriteLine($"Flushing remaining {remainingSamples.Length / (float)sampleRate:F2}s of audio...");
                await stream.PushAsync(remainingSamples);
            }
        }

        var final = await stream.FinishAsync();

        Console.WriteLine("-------------------------------------------");
        Console.WriteLine("Final Result:");
        Console.WriteLine(final.Text);
        Console.WriteLine("-------------------------------------------");

        if (final.Timestamps != null && final.Timestamps.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Timestamps:");
            foreach (var ts in final.Timestamps)
            {
                Console.WriteLine($"  [{ts.Start:F2}s - {ts.End:F2}s] {ts.Text}");
            }
        }
    }
}
