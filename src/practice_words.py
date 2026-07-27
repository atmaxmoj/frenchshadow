"""Phone-specific practice words for articulatory feedback.

Keeps per-language example-word lists so that tips can suggest words that
actually contain the target sound instead of a generic list.
"""

from __future__ import annotations

# English example words keyed by IPA phone.
_ENGLISH: dict[str, list[str]] = {
    "p": ["pig", "pen", "happy", "top"],
    "b": ["big", "bed", "rabbit", "about"],
    "t": ["time", "top", "butter", "cat"],
    "d": ["dog", "bed", "ladder", "road"],
    "k": ["cat", "key", "talking", "back"],
    "g": ["go", "good", "bigger", "dog"],
    "ɡ": ["go", "good", "bigger", "dog"],
    "f": ["fish", "fine", "after", "life"],
    "v": ["very", "video", "have", "love"],
    "θ": ["think", "three", "bath", "mouth"],
    "ð": ["this", "that", "mother", "brother"],
    "s": ["sun", "see", "bus", "house"],
    "z": ["zoo", "zero", "easy", "dogs"],
    "ʃ": ["ship", "she", "show", "washing"],
    "ʒ": ["measure", "vision", "garage", "beige"],
    "tʃ": ["chair", "child", "teacher", "church"],
    "dʒ": ["juice", "job", "age", "bridge"],
    "m": ["moon", "mother", "time", "room"],
    "n": ["no", "nine", "sunny", "pen"],
    "ŋ": ["sing", "long", "thank", "finger"],
    "l": ["light", "love", "hello", "school"],
    "r": ["red", "right", "sorry", "very"],
    "ɹ": ["red", "right", "sorry", "very"],
    "j": ["yes", "you", "beyond", "onion"],
    "w": ["water", "we", "want", "always"],
    "h": ["happy", "house", "ahead", "behind"],
    "æ": ["cat", "bad", "happy", "family"],
    "ɛ": ["bed", "head", "said", "many"],
    "ɪ": ["ship", "sit", "milk", "busy"],
    "i": ["see", "eat", "tree", "happy"],
    "ʌ": ["cup", "love", "mother", "country"],
    "ɑ": ["father", "hot", "car", "start"],
    "ɔ": ["dog", "caught", "law", "saw"],
    "ʊ": ["good", "book", "look", "could"],
    "u": ["food", "moon", "blue", "do"],
    "ə": ["about", "sofa", "banana", "system"],
    "ɜ": ["bird", "word", "nurse", "curve"],
    "ɚ": ["better", "teacher", "player", "corner"],
}

# French example words keyed by IPA phone.
_FRENCH: dict[str, list[str]] = {
    "p": ["petit", "pain", "parole", "apporter"],
    "b": ["bon", "beau", "table", "aboutir"],
    "t": ["temps", "table", "petit", "entendre"],
    "d": ["dans", "deux", "madame", "rendez"],
    "k": ["qui", "quand", "car", "école"],
    "g": ["grand", "gare", "garçon", "langue"],
    "f": ["faire", "femme", "enfant", "vie"],
    "v": ["vous", "vivre", "neuf", "vie"],
    "s": ["sous", "si", "maison", "passe"],
    "z": ["zéro", "raison", "chose", "vraiment"],
    "ʃ": ["chat", "cher", "machine", "chaque"],
    "ʒ": ["je", "jour", "rouge", "magie"],
    "m": ["mais", "madame", "maintenant", "même"],
    "n": ["non", "nous", "vin", "bonne"],
    "ɲ": ["montagne", "agneau", "camping", "vigne"],
    "ŋ": ["parking", "camping", "shopping", "standing"],
    "l": ["le", "la", "elle", "parler"],
    "ʁ": ["rouge", "parler", "merci", "très"],
    "r": ["rouge", "parler", "merci", "très"],
    "j": ["yeux", "hier", "payer", "lion"],
    "ɥ": ["nuit", "lui", "fruit", "pluie"],
    "w": ["oui", "louer", "fouet", "wallon"],
    "a": ["la", "pas", "chat", "ma"],
    "ɑ": ["pâte", "âne", "glas", "tasse"],
    "e": ["été", "chez", "aller", "ne"],
    "ɛ": ["faire", "mer", "père", "elle"],
    "i": ["si", "vie", "ici", "petite"],
    "o": ["beau", "eau", "mot", "chose"],
    "ɔ": ["porte", "bol", "homme", "pomme"],
    "u": ["vous", "tout", "rouge", "doux"],
    "y": ["tu", "rue", "sûr", "pur"],
    "ø": ["peu", "feu", "sœur", "heureux"],
    "œ": ["sœur", "peur", "heure", "cœur"],
    "ə": ["le", "petit", "je", "ne"],
}


def for_phone(phone: str, language: str = "en-us") -> list[str]:
    """Return up to four practice words containing *phone*.

    Falls back to English if the requested language is not supported.
    """
    canonical = phone.strip() if phone else ""
    lang = "fr" if language.lower().startswith("fr") else "en"
    lexicon = _FRENCH if lang == "fr" else _ENGLISH
    return lexicon.get(canonical, [])


def practice_phrase(phone: str, language: str = "en-us", default: str = "listen and repeat") -> str:
    """Return a comma-separated practice phrase for *phone*."""
    words = for_phone(phone, language)
    if words:
        return ", ".join(words[:4])
    return default
