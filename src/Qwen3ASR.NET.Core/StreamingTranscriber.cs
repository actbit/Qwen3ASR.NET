using System.Text;
using Qwen3ASR.NET.Enums;
using Qwen3ASR.NET.Models;

namespace Qwen3ASR.NET;

/// <summary>
/// Streaming transcriber for real-time speech recognition.
/// Implements buffering, chunking, and rolling context logic in .NET.
/// </summary>
public sealed class StreamingTranscriber : IAsyncDisposable, IDisposable
{
    private const int SampleRate = 16000;

    private readonly Qwen3Asr _asr;
    private readonly StreamOptions _options;
    private readonly List<float> _audioBuffer;
    private readonly List<TranscriptSegment> _segments;
    private readonly List<TranscriptSegment> _unfixedSegments;
    private readonly int _chunkSizeSamples;
    private readonly int _audioWindowSamples;
    private readonly TranscriptionOptions _transcribeOptions;

    private bool _disposed;
    private bool _finished;
    private int _totalSamplesProcessed;
    private string _rollingContext;

    /// <summary>
    /// Gets whether the stream is still active.
    /// </summary>
    public bool IsActive => !_disposed && !_finished;

    /// <summary>
    /// Gets the number of segments processed.
    /// </summary>
    public int SegmentCount => _segments.Count;

    /// <summary>
    /// Gets the total duration processed in seconds.
    /// </summary>
    public float Duration => (float)_totalSamplesProcessed / SampleRate;

    internal StreamingTranscriber(Qwen3Asr asr, StreamOptions options)
    {
        _asr = asr ?? throw new ArgumentNullException(nameof(asr));
        _options = options;
        _audioBuffer = new List<float>();
        _segments = new List<TranscriptSegment>();
        _unfixedSegments = new List<TranscriptSegment>();
        _chunkSizeSamples = (int)(options.ChunkSizeSec * SampleRate);
        _audioWindowSamples = options.AudioWindowSec.HasValue
            ? (int)(options.AudioWindowSec.Value * SampleRate)
            : 0; // 0 means no window
        _rollingContext = options.Context ?? string.Empty;

        _transcribeOptions = new TranscriptionOptions
        {
            Language = options.Language,
            Context = _rollingContext,
            ReturnTimestamps = false,
            MaxNewTokens = options.MaxNewTokens
        };
    }

    /// <summary>
    /// Pushes audio samples to the stream.
    /// </summary>
    /// <param name="audioSamples">Audio samples as 32-bit floats (16kHz, mono).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The partial transcription result.</returns>
    public async Task<TranscriptionResult> PushAsync(
        float[] audioSamples,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_finished)
            throw new InvalidOperationException("Stream is already finished");

        ArgumentNullException.ThrowIfNull(audioSamples);

        if (audioSamples.Length == 0)
            return BuildPartialResult();

        // Add to audio buffer
        _audioBuffer.AddRange(audioSamples);

        // Apply audio window if configured (rolling context)
        if (_audioWindowSamples > 0 && _audioBuffer.Count > _audioWindowSamples)
        {
            int samplesToRemove = _audioBuffer.Count - _audioWindowSamples;
            _audioBuffer.RemoveRange(0, samplesToRemove);
        }

        // Process complete chunks
        while (_audioBuffer.Count >= _chunkSizeSamples)
        {
            var chunk = _audioBuffer.Take(_chunkSizeSamples).ToArray();
            _audioBuffer.RemoveRange(0, _chunkSizeSamples);

            await ProcessChunkAsync(chunk, cancellationToken).ConfigureAwait(false);
        }

        return BuildPartialResult();
    }

    /// <summary>
    /// Gets the current partial result without pushing more audio.
    /// </summary>
    /// <returns>The partial transcription result.</returns>
    public TranscriptionResult GetPartialResult()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return BuildPartialResult();
    }

    /// <summary>
    /// Finishes the stream and returns the final transcription result.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The final transcription result.</returns>
    public async Task<TranscriptionResult> FinishAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_finished)
            throw new InvalidOperationException("Stream is already finished");

        _finished = true;

        // Process remaining buffer
        if (_audioBuffer.Count > 0)
        {
            var remainingChunk = _audioBuffer.ToArray();
            _audioBuffer.Clear();
            await ProcessChunkAsync(remainingChunk, cancellationToken).ConfigureAwait(false);
        }

        // Fix all remaining unfixed segments
        foreach (var segment in _unfixedSegments)
        {
            segment.IsFixed = true;
            _segments.Add(segment);
        }
        _unfixedSegments.Clear();

        return BuildFinalResult();
    }

    private async Task ProcessChunkAsync(float[] chunk, CancellationToken cancellationToken)
    {
        var startTime = (float)_totalSamplesProcessed / SampleRate;
        var endTime = startTime + (float)chunk.Length / SampleRate;

        try
        {
            // Update context with rolling text window
            UpdateRollingContext();

            // Transcribe the chunk
            var result = await _asr.TranscribeAsync(chunk, SampleRate, _transcribeOptions, cancellationToken)
                .ConfigureAwait(false);

            _totalSamplesProcessed += chunk.Length;

            if (!string.IsNullOrEmpty(result.Text))
            {
                var newSegment = new TranscriptSegment
                {
                    Text = result.Text,
                    StartTime = startTime,
                    EndTime = endTime,
                    IsFixed = false
                };

                // Add to unfixed segments
                _unfixedSegments.Add(newSegment);

                // Manage unfixed segments - fix old ones based on UnfixedChunkNum
                while (_unfixedSegments.Count > _options.UnfixedChunkNum)
                {
                    var segmentToFix = _unfixedSegments[0];
                    _unfixedSegments.RemoveAt(0);
                    segmentToFix.IsFixed = true;
                    _segments.Add(segmentToFix);
                }

                // Update rolling context for next chunk
                UpdateRollingContext();
            }
        }
        catch (Qwen3AsrException)
        {
            // Re-throw transcription errors
            throw;
        }
    }

    private void UpdateRollingContext()
    {
        // Build context from fixed segments and apply text window
        var allText = new StringBuilder();

        // Add original context if present
        if (!string.IsNullOrEmpty(_options.Context))
        {
            allText.Append(_options.Context);
            allText.Append(' ');
        }

        // Add fixed segments
        foreach (var segment in _segments)
        {
            allText.Append(segment.Text);
            allText.Append(' ');
        }

        // Add unfixed segments
        foreach (var segment in _unfixedSegments)
        {
            allText.Append(segment.Text);
            allText.Append(' ');
        }

        var fullText = allText.ToString().Trim();

        // Apply text window (token-based approximation: ~4 chars per token)
        if (_options.TextWindowTokens.HasValue && _options.TextWindowTokens.Value > 0)
        {
            int maxChars = _options.TextWindowTokens.Value * 4;
            if (fullText.Length > maxChars)
            {
                fullText = fullText[^maxChars..];
            }
        }

        _rollingContext = fullText;
        _transcribeOptions.Context = _rollingContext;
    }

    private TranscriptionResult BuildPartialResult()
    {
        // Combine fixed and unfixed segments
        var allSegments = _segments.Concat(_unfixedSegments).ToList();
        var text = string.Join(" ", allSegments.Select(s => s.Text));

        return new TranscriptionResult
        {
            Text = text,
            Language = _options.Language.ToNativeString(),
            IsPartial = true
        };
    }

    private TranscriptionResult BuildFinalResult()
    {
        var text = string.Join(" ", _segments.Select(s => s.Text));

        var timestamps = _segments.Select(s => new Timestamp
        {
            Start = s.StartTime,
            End = s.EndTime,
            Text = s.Text
        }).ToList();

        return new TranscriptionResult
        {
            Text = text,
            Language = _options.Language.ToNativeString(),
            Timestamps = timestamps,
            IsPartial = false
        };
    }

    /// <summary>
    /// Releases all resources used by the StreamingTranscriber.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        _audioBuffer.Clear();
        _segments.Clear();
        _unfixedSegments.Clear();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Asynchronously releases all resources used by the StreamingTranscriber.
    /// </summary>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    /// <summary>
    /// Finalizer to ensure resources are released.
    /// </summary>
    ~StreamingTranscriber()
    {
        Dispose();
    }

    private class TranscriptSegment
    {
        public string Text { get; init; } = string.Empty;
        public float StartTime { get; init; }
        public float EndTime { get; init; }
        public bool IsFixed { get; set; }
    }
}
