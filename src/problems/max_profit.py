def max_profit(prices: list[int]) -> int:
    # first find a minimum, then after that a maximum -> difference is profit.
    # brute: take element index, maximum after that -> if larger than element, profit -> find biggest profit

    profit = 0
    if prices:
        min_price = prices[0]

        for price in prices[1:]:
            # faster, if more explicit. Max and min can add overhead
            profit = max(profit, price - min_price)
            min_price = min(min_price, price)
    return profit

    # for index, element in enumerate(prices):
    #    remainder = prices[index + 1 :]
    #    print(remainder, element)
    #    if remainder:
    #        print(max(remainder))
    #        profit = max(profit, max(remainder) - element)
    # return profit
