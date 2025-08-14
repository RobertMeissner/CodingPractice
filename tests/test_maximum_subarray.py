import pytest

from src.maximum_subarray import maximum_subarray


@pytest.mark.parametrize(
    "arr, expected",
    [([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6), ([], 0), ([1], 1), ([-1], -1)],
)
def test_maximum_subarray(arr: list[int], expected: int) -> None:
    result = maximum_subarray(arr)
    assert result == expected
