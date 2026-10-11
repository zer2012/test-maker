def clamp(value, low, high):
    """Ramène value dans l'intervalle [low, high]."""
    if low > high:
        raise ValueError("low must be <= high")
    if value < low:
        return low
    if value > high:
        return high
    return value


def is_leap_year(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0


def word_count(text):
    counts = {}
    for word in text.lower().split():
        word = word.strip(".,;:!?")
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts
