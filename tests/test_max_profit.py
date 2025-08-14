import pytest

from src.max_profit import max_profit


@pytest.mark.parametrize("arr, profit", [([7, 1, 5, 3, 6, 4], 5), ([], 0), ([1], 0)])
def test_max_profit(arr, profit):
    assert max_profit(arr) == profit
