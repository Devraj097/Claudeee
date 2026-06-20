# Requires: pip install pydub
# Requires: ffmpeg installed on your system (https://ffmpeg.org/download.html)
#   macOS:   brew install ffmpeg
#   Ubuntu:  sudo apt install ffmpeg
#   Windows: download from ffmpeg.org and add to PATH

import argparse
import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def remove_silence(input_path, output_path, min_silence_ms=700, silence_thresh_db=-50):
    print(f"Loading: {input_path}")
    audio = AudioSegment.from_file(input_path)

    print(f"Detecting silence (threshold={silence_thresh_db}dBFS, min_duration={min_silence_ms}ms)...")
    non_silent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
    )

    if not non_silent_ranges:
        print("No audio detected above the silence threshold. Try raising --threshold.")
        return

    print(f"Found {len(non_silent_ranges)} non-silent segment(s). Joining...")
    cleaned = AudioSegment.empty()
    for start, end in non_silent_ranges:
        cleaned += audio[start:end]

    ext = output_path.rsplit(".", 1)[-1].lower()
    cleaned.export(output_path, format=ext)
    print(f"Done! Saved to: {output_path}")
    print(f"Original duration: {len(audio)/1000:.1f}s  →  Cleaned duration: {len(cleaned)/1000:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Remove silence gaps from an audio mix file."
    )
    parser.add_argument("input", help="Path to the input audio file (e.g. mymix.mp3)")
    parser.add_argument(
        "--output",
        help="Path for the output file (default: <input>_cleaned.<ext>)",
        default=None,
    )
    parser.add_argument(
        "--min-silence",
        type=int,
        default=700,
        help="Minimum silence length in milliseconds to remove (default: 700). "
             "Lower this if short gaps aren't being caught.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=-50,
        help="Silence threshold in dBFS (default: -50). "
             "Raise toward 0 (e.g. -45) if gaps aren't being detected.",
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
