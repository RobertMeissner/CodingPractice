def sort_array(arr: list[int]) -> int:
    print("sorting")
    _swaps = 0
    for index in range(len(arr)):
        while arr[index] != index + 1:
            correct_position = arr[index] - 1
            arr[index], arr[correct_position] = arr[correct_position], arr[index]
            _swaps += 1
    print(arr)
    return _swaps


if __name__ == "__main__":
    array = [4, 3, 2, 1]
    expected = 2
    swaps = sort_array(array)
    assert expected == swaps
