import pytest

from solution import clamp, is_leap_year, word_count


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
    assert clamp(0, 0, 10) == 0
    with pytest.raises(ValueError):
        clamp(1, 10, 0)


def test_is_leap_year():
    assert is_leap_year(2000)
    assert not is_leap_year(1900)
    assert is_leap_year(2024)
    assert not is_leap_year(2023)


def test_word_count():
    assert word_count("A a, b!") == {"a": 2, "b": 1}
    assert word_count("") == {}
