# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import random

import pytest
from max.nn.kv_cache.paged_cache.simple_trie import SimpleTrie


@pytest.mark.asyncio
async def test_simple_trie() -> None:
    trie = SimpleTrie()
    trie.insert(["i", "like", "to", "eat", "pie"])
    assert trie.pretty_format() == [
        "i",
        "--like",
        "----to",
        "------eat",
        "--------pie",
        "----------*",
    ]

    # inserting same thing is a noop
    trie.insert(["i", "like", "to", "eat", "pie"])
    assert trie.pretty_format() == [
        "i",
        "--like",
        "----to",
        "------eat",
        "--------pie",
        "----------*",
    ]

    assert ["i", "like", "to", "eat", "pie"] in trie
    assert ["i", "like", "to"] not in trie
    assert ["we", "like", "to", "eat", "pie"] not in trie
    assert ["i", "like", "to", "eat", "pie", "everyday"] not in trie
    assert ["i", "like", "to", "eat", "pears"] not in trie

    trie.insert(["i", "like", "to", "bake", "pie"])
    assert trie.pretty_format() == [
        "i",
        "--like",
        "----to",
        "------eat",
        "--------pie",
        "----------*",
        "------bake",
        "--------pie",
        "----------*",
    ]
    assert ["i", "like", "to", "bake", "pie"] in trie
    assert ["i", "like", "to", "eat", "pie"] in trie

    del trie["i", "like", "to", "eat", "pie"]
    assert trie.pretty_format() == [
        "i",
        "--like",
        "----to",
        "------bake",
        "--------pie",
        "----------*",
    ]
    assert ["i", "like", "to", "eat", "pie"] not in trie
    assert ["i", "like", "to", "bake", "pie"] in trie

    trie.insert(["i", "like", "to", "cook", "pizza"])
    assert trie.pretty_format() == [
        "i",
        "--like",
        "----to",
        "------bake",
        "--------pie",
        "----------*",
        "------cook",
        "--------pizza",
        "----------*",
    ]

    res = trie.find_string_with_largest_common_prefix(
        ["i", "like", "to", "cook", "eggs"]
    )
    assert res is not None
    s, prefix_len = res
    assert s == ["i", "like", "to", "cook", "pizza"]
    assert prefix_len == 4

    del trie["i", "like", "to", "cook", "pizza"]
    assert trie.pretty_format() == [
        "i",
        "--like",
        "----to",
        "------bake",
        "--------pie",
        "----------*",
    ]

    res = trie.find_string_with_largest_common_prefix(
        ["i", "like", "to", "cook", "eggs"]
    )
    assert res is not None
    s, prefix_len = res
    assert s == ["i", "like", "to", "bake", "pie"]
    assert prefix_len == 3

    trie.insert(["i", "like", "to", "bake", "pie", "everyday"])
    assert trie.pretty_format() == [
        "i",
        "--like",
        "----to",
        "------bake",
        "--------pie",
        "----------*",
        "----------everyday",
        "------------*",
    ]

    trie.insert([])
    assert trie.pretty_format() == [
        "*",
        "i",
        "--like",
        "----to",
        "------bake",
        "--------pie",
        "----------*",
        "----------everyday",
        "------------*",
    ]


@pytest.mark.asyncio
async def test_simple_trie_random() -> None:
    def gen_random_word() -> str:
        word_length = random.randint(1, 3)
        return "".join(random.choices(["a", "b"], k=word_length))

    def gen_random_sentence() -> tuple[str, ...]:
        sentence_length = random.randint(1, 10)
        return tuple(gen_random_word() for _ in range(sentence_length))

    deleted_sentences = set()
    current_sentences = set()
    trie = SimpleTrie()

    for _ in range(10000):
        # insert new sentence into trie
        new_sentence = gen_random_sentence()
        current_sentences.add(new_sentence)
        trie.insert(new_sentence)

        # get random sentence from my_set
        sentence = random.choice(list(current_sentences))
        assert sentence in trie

        # delete random sentence from my_set with probability 0.5
        if random.random() < 0.5:
            del trie[sentence]
            assert sentence not in trie
            deleted_sentences.add(sentence)
            current_sentences.remove(sentence)

        # get random deleted sentence
        if deleted_sentences:
            deleted_sentence = random.choice(list(deleted_sentences))
            # if it is not in current_sentences, it should not be in trie
            if deleted_sentence not in current_sentences:
                assert deleted_sentence not in trie
