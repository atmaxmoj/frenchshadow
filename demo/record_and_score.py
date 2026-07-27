"""CLI demo: record a sentence from the microphone and print the analysis."""

from __future__ import annotations

import argparse
import json

import sys
import os

# Allow importing src when running from the demo directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import analyze
from src.articulatory import attach_tips
from src.audio import record_audio
from src.models import transcribe

DEFAULT_SENTENCE = "The quick brown fox jumps over the lazy dog."


def main():
    parser = argparse.ArgumentParser(description="Record and score pronunciation")
    parser.add_argument("--text", default=DEFAULT_SENTENCE, help="target sentence")
    parser.add_argument("--duration", type=float, default=5.0, help="recording seconds")
    parser.add_argument("--save", default=None, help="optional path to save WAV")
    args = parser.parse_args()

    print(f"Target: {args.text}")
    print(f"Recording {args.duration}s...")
    audio = record_audio(duration=args.duration)

    if args.save:
        from src.audio import save_audio
        save_audio(args.save, audio)
        print(f"Saved to {args.save}")

    print("Transcribing...")
    result = transcribe(audio)
    print("Learner IPA:", result["raw"])

    print("Analyzing...")
    analysis = analyze(args.text, result["tokens"])
    attach_tips(analysis)

    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
