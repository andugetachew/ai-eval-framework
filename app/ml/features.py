def word_overlap_ratio(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def length_ratio(a: str, b: str) -> float:
    len_a, len_b = len(a.split()), len(b.split())
    if max(len_a, len_b) == 0:
        return 0.0
    return min(len_a, len_b) / max(len_a, len_b)