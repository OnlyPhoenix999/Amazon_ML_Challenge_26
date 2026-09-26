from collections import Counter
import math


def char_ngrams(text: str, n: int = 3) -> Counter:
    text = str(text).lower().strip()

    if len(text) < n:
        return Counter()

    return Counter(
        text[i:i + n]
        for i in range(len(text) - n + 1)
    )


def cosine_similarity(counter1: Counter, counter2: Counter) -> float:
    if not counter1 or not counter2:
        return 0.0

    common = set(counter1) & set(counter2)

    dot_product = sum(
        counter1[x] * counter2[x]
        for x in common
    )

    magnitude1 = math.sqrt(
        sum(value * value for value in counter1.values())
    )

    magnitude2 = math.sqrt(
        sum(value * value for value in counter2.values())
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def char_ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    grams1 = char_ngrams(text1, n)
    grams2 = char_ngrams(text2, n)

    return cosine_similarity(grams1, grams2)


def build_ngram_features(
    name1: str,
    name2: str,
    transliterated_name1: str,
    transliterated_name2: str,
) -> dict:

    features = {}

    # Original names
    for n in (3, 4, 5):
        features[f"name_char_{n}gram"] = char_ngram_similarity(
            name1,
            name2,
            n
        )

    # Transliterated names
    for n in (3, 4, 5):
        features[f"translit_char_{n}gram"] = char_ngram_similarity(
            transliterated_name1,
            transliterated_name2,
            n
        )

    return features

if __name__ == "__main__":

    pairs = [
        (
            "महाकाल ट्रेडर्स",
            "Mahakaal Traders",
            "mahakala tredarsa",
            "mahakaal traders",
        ),
        (
            "ABC Enterprises",
            "ABC Enterprise",
            "abc enterprises",
            "abc enterprise",
        ),
        (
            "Pinnacle Healthcare",
            "Pinnacle Health Care",
            "pinnacle healthcare",
            "pinnacle health care",
        ),
    ]

    print("\n--- Character N-gram Test ---\n")

    for name1, name2, translit1, translit2 in pairs:

        features = build_ngram_features(
            name1,
            name2,
            translit1,
            translit2
        )

        print(f"{name1} ↔ {name2}")
        print(f"Translit: {translit1} ↔ {translit2}")
        print(features)
        print()  