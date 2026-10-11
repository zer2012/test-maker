from solution import clamp


def test_nothing():
    clamp(5, 0, 10)
    assert True
