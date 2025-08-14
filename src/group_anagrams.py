def groupAnagrams(input: list[str]) -> list[list[str]]:
    hash: dict[str, list[str]] = {}

    for element in input:
        sorted_element = "".join(sorted(element.lower()))

        hash[sorted_element] = hash.get(sorted_element, []) + [element]
        # if sorted_element in hash.keys():
        #     hash[sorted_element].append(element)
        # else:
        #     hash.update({sorted_element: [element]})

    return list(hash.values())
