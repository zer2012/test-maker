from solution import clamp


def test_wrong_expectation():
    assert clamp(-1, 0, 10) == -1
