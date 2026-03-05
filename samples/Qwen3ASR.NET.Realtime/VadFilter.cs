namespace Qwen3ASR.NET.Realtime;

/// <summary>
/// Voice Activity Detection (VAD) with silence padding
/// Keeps the audio stream continuous by including silence periods
/// </summary>
public class VadFilter
{
    private readonly float _energyThreshold;
    private readonly int _sampleRate;

    private readonly List<float> _audioBuffer = new();
    private readonly object _bufferLock = new();

    /// <summary>
    /// Gets the total duration of audio processed.
    /// </summary>
    public TimeSpan TotalDuration { get; private set; }

    /// <summary>
    /// Gets the duration of silence detected.
    /// </summary>
    public TimeSpan SilenceDuration { get; private set; }

    /// <summary>
    /// Creates a new VAD filter.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate.</param>
    /// <param name="energyThreshold">Energy threshold (0.0-1.0). Default: 0.01</param>
    /// <param name="silenceDurationMs">Ignored (kept for API compatibility).</param>
    public VadFilter(int sampleRate, float energyThreshold = 0.01f, int silenceDurationMs = 300)
    {
        _sampleRate = sampleRate;
        _energyThreshold = energyThreshold;
    }

    /// <summary>
    /// Processes audio samples - always adds to buffer (silence as zeros, speech as-is).
    /// </summary>
    /// <param name="samples">Input audio samples.</param>
    /// <returns>Always null (use Flush to get buffered audio).</returns>
    public float[]? Process(float[] samples)
    {
        TotalDuration += TimeSpan.FromSeconds((double)samples.Length / _sampleRate);

        // Calculate RMS energy
        float energy = CalculateEnergy(samples);
        bool isSilence = energy < _energyThreshold;

        if (isSilence)
        {
            SilenceDuration += TimeSpan.FromSeconds((double)samples.Length / _sampleRate);
        }

        // Always add to buffer: silence as zeros, speech as-is
        lock (_bufferLock)
        {
            if (isSilence)
            {
                // Add silence as zeros to maintain timing
                _audioBuffer.AddRange(samples);
            }
            else
            {
                // Add actual speech
                _audioBuffer.AddRange(samples);
            }
        }

        return null; // Use Flush to get buffered audio
    }

    /// <summary>
    /// Flushes buffered audio and clears the buffer.
    /// </summary>
    /// <returns>Buffered audio samples, or null if empty or all silence.</returns>
    public float[]? Flush()
    {
        lock (_bufferLock)
        {
            if (_audioBuffer.Count == 0)
                return null;

            var result = _audioBuffer.ToArray();
            _audioBuffer.Clear();

            // Return null if all silence (all zeros)
            if (IsAllSilence(result))
                return null;

            return result;
        }
    }

    /// <summary>
    /// Gets buffered audio without clearing.
    /// </summary>
    /// <returns>Buffered audio samples, or null if empty or all silence.</returns>
    public float[]? PeekBuffer()
    {
        lock (_bufferLock)
        {
            if (_audioBuffer.Count == 0)
                return null;

            var result = _audioBuffer.ToArray();

            // Return null if all silence
            if (IsAllSilence(result))
                return null;

            return result;
        }
    }

    /// <summary>
    /// Gets the actual buffer count (ignoring trailing silence).
    /// </summary>
    public int GetEffectiveBufferCount()
    {
        lock (_bufferLock)
        {
            if (_audioBuffer.Count == 0)
                return 0;

            // Check if there's any non-silence in the buffer
            foreach (var sample in _audioBuffer)
            {
                if (Math.Abs(sample) > 0.0001f)
                    return _audioBuffer.Count;
            }

            return 0; // All silence
        }
    }

    /// <summary>
    /// Clears specified number of samples from the beginning of the buffer.
    /// </summary>
    public void ConsumeSamples(int count)
    {
        lock (_bufferLock)
        {
            if (count >= _audioBuffer.Count)
            {
                _audioBuffer.Clear();
            }
            else
            {
                _audioBuffer.RemoveRange(0, count);
            }
        }
    }

    /// <summary>
    /// Checks if all samples are effectively silence (near zero).
    /// </summary>
    private static bool IsAllSilence(float[] samples)
    {
        const float threshold = 0.0001f;
        foreach (var sample in samples)
        {
            if (Math.Abs(sample) > threshold)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Gets the current buffer length in samples.
    /// </summary>
    public int BufferedSampleCount
    {
        get
        {
            lock (_bufferLock)
            {
                return _audioBuffer.Count;
            }
        }
    }

    /// <summary>
    /// Resets the VAD filter state.
    /// </summary>
    public void Reset()
    {
        lock (_bufferLock)
        {
            _audioBuffer.Clear();
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
