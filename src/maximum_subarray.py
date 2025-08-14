def maximum_subarray(nums: list[int]) -> int:
    # sliding window

    # brute force, i from the left, j from the right, two for loops, calculate sum, e.g., with a hash, sum as key
    # hash not needed, only highest sum

    # O(n**2)
    if not nums:
        return 0
    length = len(nums)

    BRUTE = False
    if BRUTE:
        maximum = nums[0]  # educated guess
        for index_left in range(length):
            for index_right in range(1, length - index_left + 1):
                current_sum = sum(nums[index_left:-index_right])
                print(current_sum, index_left, index_right, nums[index_left:-index_right])
                maximum = max(maximum, current_sum)

        print(maximum)
    # Kadane's algorithm

    if not nums:
        return 0
    length = len(nums)
    maximum = max_so_far = 0  # minimum 0, for Kadane float("-inf")

    for index in range(1, length):
        maximum = max(nums[index], maximum + nums[index])
        max_so_far = max(max_so_far, maximum)

    return max_so_far
