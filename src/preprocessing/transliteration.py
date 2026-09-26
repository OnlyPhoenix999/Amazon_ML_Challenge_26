from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# Unicode ranges for the scripts we care about
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "oriya": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}


SCRIPT_TO_SCHEME = {
    "devanagari": sanscript.DEVANAGARI,
    "bengali": sanscript.BENGALI,
    "gurmukhi": sanscript.GURMUKHI,
    "gujarati": sanscript.GUJARATI,
    "oriya": sanscript.ORIYA,
    "tamil": sanscript.TAMIL,
    "telugu": sanscript.TELUGU,
    "kannada": sanscript.KANNADA,
    "malayalam": sanscript.MALAYALAM,
}


def detect_script(text: str) -> str | None:
    """Detect the dominant Indic script in the text."""

    counts = {script: 0 for script in SCRIPT_RANGES}

    for char in text:
        code = ord(char)

        for script, (start, end) in SCRIPT_RANGES.items():
            if start <= code <= end:
                counts[script] += 1
                break

    detected = max(counts, key=counts.get)

    if counts[detected] == 0:
        return None

    return detected


def transliterate_text(text: object) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    script = detect_script(text)

    # Latin / unknown text → keep it as-is
    if script is None:
        return text.lower()

    scheme = SCRIPT_TO_SCHEME[script]

    try:
        result = transliterate(
            text,
            scheme,
            sanscript.ITRANS
        )
    except Exception:
        # Never destroy the original information
        return text.lower()

    return result.lower()


# Temporary test
if __name__ == "__main__":
    samples = [
        "महाकाल ट्रेडर्स",
        "ગુજરાતી વેપાર",
        "తెలుగు వ్యాపారం",
        "தமிழ் நிறுவனம்",
        "ಕನ್ನಡ ವ್ಯಾಪಾರ",
        "বাংলা ব্যবসা",
        "ਪੰਜਾਬੀ ਵਪਾਰ",
        "ABC Traders Pvt Ltd",
    ]

    print("\n--- Transliteration Test ---\n")

    for text in samples:
        print(f"{text} → {transliterate_text(text)}")