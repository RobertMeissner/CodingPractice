import pytest

from src.problems.anagram_calculator import anagram_calculator


@pytest.mark.parametrize("text, count", [("mom", 2), ("", 0), ("eee", 3), ("Aabb", 2)])
def test_anagram_calculator(text, count):
    assert anagram_calculator(text) == count
