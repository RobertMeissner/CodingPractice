def frequency_sort(s: str) -> str:
    """


    :param s: input string, e.g, "tree"
    :return: string of character occurrence in decreased frequency, e.g. "etr". If multiple answers, just any (e.g. tr or rt does not matter)
    """

    counter = {char: s.count(char) for char in set(s)}
    return "".join(k * v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True))
