import pytest

from src.frequency_sort import frequency_sort


@pytest.mark.parametrize("text, result", [("tree", "eert"), ("", ""), ("eee", "eee"), ("Aabb", "bbAa")])
def test_frequency_sort(text: str, result: str) -> None:
    assert frequency_sort(text) == result
