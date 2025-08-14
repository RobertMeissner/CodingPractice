from src.arraysorting import sort_array


def test_sort_array() -> None:
    array = [4, 3, 2, 1]
    expected = 2
    swaps = sort_array(array)
    assert expected == swaps
