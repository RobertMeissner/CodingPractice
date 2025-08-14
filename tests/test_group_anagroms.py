from src.group_anagrams import groupAnagrams


def test_group_anagrams() -> None:
    anagrams = ["eat", "tea", "tan", "ate", "nat", "bat"]

    result = groupAnagrams(anagrams)
    print(result)
    assert len(result) == 3
