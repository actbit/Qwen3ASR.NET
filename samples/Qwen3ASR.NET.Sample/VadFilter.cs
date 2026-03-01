namespace Qwen3ASR.NET.Sample;

/// <summary>
/// Simple energy-based Voice Activity Detection (VAD)
/// </summary>
public class VadFilter
{
    private readonly float _energyThreshold;
    private readonly int _silenceDurationMs;
    private readonly int _sampleRate;

    private int _silenceSampleCount;
    private int _silenceThresholdSamples;
    private readonly List<float> _speechBuffer = new();

    /// <summary>
    /// Gets the total duration of audio processed.
    /// </summary>
    public TimeSpan TotalDuration { get; private set; }

    /// <summary>
    /// Gets the duration of silence filtered out.
    /// </summary>
    public TimeSpan SilenceDuration { get; private set; }

    /// <summary>
    /// Creates a new VAD filter.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate.</param>
    /// <param name="energyThreshold">Energy threshold (0.0-1.0). Default: 0.01</param>
    /// <param name="silenceDurationMs">Silence duration in ms to filter out. Default: 300ms</param>
    public VadFilter(int sampleRate, float energyThreshold = 0.01f, int silenceDurationMs = 300)
    {
        _sampleRate = sampleRate;
        _energyThreshold = energyThreshold;
        _silenceDurationMs = silenceDurationMs;
        _silenceThresholdSamples = (int)(sampleRate * silenceDurationMs / 1000.0);
    }

    /// <summary>
    /// Processes audio samples and returns only speech segments.
    /// </summary>
    /// <param name="samples">Input audio samples.</param>
    /// <returns>Audio samples with silence removed, or null if all silence.</returns>
    public float[]? Process(float[] samples)
    {
        TotalDuration += TimeSpan.FromSeconds((double)samples.Length / _sampleRate);

        // Calculate RMS energy
        float energy = CalculateEnergy(samples);

        if (energy < _energyThreshold)
        {
            // Silence detected
            _silenceSampleCount += samples.Length;

            // If we have accumulated silence, don't return anything
            SilenceDuration += TimeSpan.FromSeconds((double)samples.Length / _sampleRate);
            return null;
        }
        else
        {
            // Speech detected
            _silenceSampleCount = 0;

            // Add to speech buffer
            lock (_speechBuffer)
            {
                _speechBuffer.AddRange(samples);
            }

            return null; // We accumulate and return on Flush
        }
    }

    /// <summary>
    /// Flushes any remaining buffered audio.
    /// </summary>
    /// <returns>Buffered audio samples, or null if empty.</returns>
    public float[]? Flush()
    {
        lock (_speechBuffer)
        {
            if (_speechBuffer.Count == 0)
                return null;

            var result = _speechBuffer.ToArray();
            _speechBuffer.Clear();
            return result;
        }
    }

    /// <summary>
    /// Gets the current buffer length in samples.
    /// </summary>
    public int BufferedSampleCount
    {
        get
        {
            lock (_speechBuffer)
            {
                return _speechBuffer.Count;
            }
        }
    }

    /// <summary>
    /// Resets the VAD filter state.
    /// </summary>
    public void Reset()
    {
        lock (_speechBuffer)
        {
            _speechBuffer.Clear();
            _silenceSampleCount = 0;
        }
        TotalDuration = TimeSpan.Zero;
        SilenceDuration = TimeSpan.Zero;
    }

    private static float CalculateEnergy(float[] samples)
    {
        if (samples.Length == 0)
            return 0f;

        float sumSquares = 0f;
        for (int i = 0; i < samples.Length; i++)
        {
            sumSquares += samples[i] * samples[i];
        }

        return MathF.Sqrt(sumSquares / samples.Length);
    }
}
