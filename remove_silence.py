# Dependencies: pip install miniaudio numpy lameenc soundfile
# No ffmpeg required.

import argparse
import os
import numpy as np
import miniaudio
import lameenc


def detect_nonsilent(mono, sample_rate, min_silence_ms=700, silence_thresh_db=-50):
    """Return list of (start, end) sample index pairs for non-silent regions."""
    silence_thresh_amp = 10 ** (silence_thresh_db / 20.0)
    min_silence_samples = int(sample_rate * min_silence_ms / 1000)

    # RMS over 20ms windows
    frame_len = int(sample_rate * 0.02)
    n_frames = len(mono) // frame_len
    rms = np.array([
        np.sqrt(np.mean(mono[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    is_sound = rms > silence_thresh_amp

    # Find transitions
    transitions = np.diff(is_sound.astype(int))
    sound_starts = np.where(transitions == 1)[0] + 1
    sound_ends = np.where(transitions == -1)[0] + 1

    if is_sound[0]:
        sound_starts = np.concatenate([[0], sound_starts])
    if is_sound[-1]:
        sound_ends = np.concatenate([sound_ends, [n_frames]])

    # Convert frame indices to sample indices, merging gaps smaller than min_silence_samples
    regions = []
    for s, e in zip(sound_starts, sound_ends):
        start_sample = s * frame_len
        end_sample = min(e * frame_len, len(mono))
        if regions and (start_sample - regions[-1][1]) < min_silence_samples:
            regions[-1] = (regions[-1][0], end_sample)
        else:
            regions.append((start_sample, end_sample))

    return regions


def remove_silence(input_path, output_path, min_silence_ms=700, silence_thresh_db=-50):
    print(f"Loading: {input_path}")
    decoded = miniaudio.mp3_read_file_f32(input_path)
    sample_rate = decoded.sample_rate
    nchannels = decoded.nchannels
    samples = np.array(decoded.samples, dtype=np.float32).reshape(-1, nchannels)
    mono = samples.mean(axis=1)

    duration = len(mono) / sample_rate
    print(f"Duration: {duration:.1f}s  |  {nchannels}ch  |  {sample_rate}Hz")

    print(f"Detecting silence (threshold={silence_thresh_db}dBFS, min_gap={min_silence_ms}ms)...")
    regions = detect_nonsilent(mono, sample_rate, min_silence_ms, silence_thresh_db)

    if not regions:
        print("No audio detected above the silence threshold. Try raising --threshold (e.g. -40).")
        return

    print(f"Found {len(regions)} non-silent segment(s). Joining...")
    kept_chunks = [samples[s:e] for s, e in regions]
    cleaned = np.concatenate(kept_chunks, axis=0)

    cleaned_duration = len(cleaned) / sample_rate
    removed = duration - cleaned_duration
    print(f"Removed {removed:.1f}s of silence  →  Output duration: {cleaned_duration:.1f}s")

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp3":
        _export_mp3(cleaned, sample_rate, nchannels, output_path)
    else:
        import soundfile as sf
        sf.write(output_path, cleaned, sample_rate)

    print(f"Saved to: {output_path}")


def _export_mp3(samples, sample_rate, nchannels, output_path):
    # Convert float32 [-1, 1] to int16
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(nchannels)
    encoder.set_quality(2)  # 2 = high quality

    # Feed in chunks to avoid memory spikes
    chunk_size = sample_rate * 10  # 10s chunks
    mp3_data = b""
    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i:i + chunk_size]
        mp3_data += encoder.encode(chunk.tobytes())
    mp3_data += encoder.flush()

    with open(output_path, "wb") as f:
        f.write(mp3_data)


def main():
    parser = argparse.ArgumentParser(
        description="Remove silence gaps from an audio mix file. No ffmpeg required."
    )
    parser.add_argument("input", help="Path to the input audio file (.mp3 or .wav)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: <input>_cleaned.<ext>)",
    )
    parser.add_argument(
        "--min-silence",
        type=int,
        default=700,
        help="Minimum silence gap in ms to remove (default: 700). Lower for shorter gaps.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=-50,
        help="Silence threshold in dBFS (default: -50). Raise toward 0 if gaps aren't detected.",
    )
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_cleaned{ext}"

    remove_silence(
        input_path=args.input,
        output_path=args.output,
        min_silence_ms=args.min_silence,
        silence_thresh_db=args.threshold,
    )


if __name__ == "__main__":
    main()
