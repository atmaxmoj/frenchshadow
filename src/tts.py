"""Text-to-speech for shadow-reader.

Prefers `edge-tts` (Microsoft Azure neural voices, same high-quality voices used
by vocab-app) and falls back to macOS `say` for offline use. Generated audio is
cached on disk so repeated sentences do not re-synthesize.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).with_suffix("").parent.parent / "audio_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Microsoft Azure neural voices used by vocab-app's prebuild_tts.py.
EDGE_VOICES = {
    "en-us": "en-US-AriaNeural",
    "en-gb": "en-GB-SoniaNeural",
    "fr-fr": "fr-FR-DeniseNeural",
    "fr-ca": "fr-CA-SylvieNeural",
    "es-es": "es-ES-ElviraNeural",
    "de-de": "de-DE-KatjaNeural",
}

# macOS `say` voices for offline fallback.
SAY_VOICES = {
    "en-us": "Samantha",
    "en-gb": "Daniel",
    "fr-fr": "Thomas",
    "fr-ca": "Amelie",
    "es-es": "Monica",
    "de-de": "Anna",
}


def _cache_key(text: str, voice: str) -> str:
    return hashlib.sha256(f"{voice}:{text}".encode("utf-8")).hexdigest()


def _edge_tts(text: str, voice: str, out_path: Path) -> bool:
    if not shutil.which("edge-tts"):
        return False
    try:
        subprocess.run(
            [
                "edge-tts",
                "--voice",
                voice,
                "--text",
                text,
                "--write-media",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except Exception as exc:
        logger.warning("edge-tts failed: %s", exc)
        return False


def _say_tts(text: str, voice: str, out_path: Path) -> bool:
    if not shutil.which("say"):
        return False
    # `say` always writes AIFF-C; we use a temp .aiff and convert below.
    aiff_path = out_path.with_suffix(".aiff")
    try:
        subprocess.run(
            ["say", "-v", voice, text, "-o", str(aiff_path)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(aiff_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except Exception as exc:
        logger.warning("say failed: %s", exc)
        return False
    finally:
        aiff_path.unlink(missing_ok=True)


def _convert_to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dst),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )


def synthesize(text: str, language: str = "en-us") -> bytes:
    """Return WAV audio bytes for *text* in *language*.

    Raises:
        RuntimeError: if no TTS backend succeeds.
    """
    if not text or not text.strip():
        raise ValueError("empty text")

    lang = language.lower()
    edge_voice = EDGE_VOICES.get(lang, EDGE_VOICES["en-us"])
    cache_key = _cache_key(text, edge_voice)
    wav_path = CACHE_DIR / f"{cache_key}.wav"

    if wav_path.exists():
        return wav_path.read_bytes()

    mp3_path = CACHE_DIR / f"{cache_key}.mp3"
    if mp3_path.exists():
        _convert_to_wav(mp3_path, wav_path)
        return wav_path.read_bytes()

    if _edge_tts(text, edge_voice, mp3_path):
        _convert_to_wav(mp3_path, wav_path)
        return wav_path.read_bytes()

    say_voice = SAY_VOICES.get(lang, SAY_VOICES["en-us"])
    if _say_tts(text, say_voice, wav_path):
        return wav_path.read_bytes()

    raise RuntimeError("no TTS backend available")
