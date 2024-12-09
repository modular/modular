# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.pipelines.kv_cache.radix_trie import RadixTrie


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_insert() -> None:
    trie = RadixTrie()
    trie.insert(
        ["i", "like", "to", "eat", "pie"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"],
    )
    assert trie.pretty_format() == [
        "['i', 'like', 'to', 'eat', 'pie']",
    ]

    trie.insert(
        ["i", "like", "to", "eat", "pizza"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 5"],
    )
    assert trie.pretty_format() == [
        "['i', 'like', 'to', 'eat']",
        "--['pie']",
        "--['pizza']",
    ]

    trie.insert(
        ["i", "like", "to", "dance"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 6", "BLOCK 7"],
    )
    assert trie.pretty_format() == [
        "['i', 'like', 'to']",
        "--['eat']",
        "----['pie']",
        "----['pizza']",
        "--['dance']",
    ]

    trie.insert(
        ["we", "like", "to", "dance"],
        ["BLOCK 8", "BLOCK 8", "BLOCK 9", "BLOCK 10"],
    )
    assert trie.pretty_format() == [
        "['i', 'like', 'to']",
        "--['eat']",
        "----['pie']",
        "----['pizza']",
        "--['dance']",
        "['we', 'like', 'to', 'dance']",
    ]

    trie.insert(
        ["we", "like", "to", "frolic"],
        ["BLOCK 8", "BLOCK 8", "BLOCK 9", "BLOCK 11"],
    )
    assert trie.pretty_format() == [
        "['i', 'like', 'to']",
        "--['eat']",
        "----['pie']",
        "----['pizza']",
        "--['dance']",
        "['we', 'like', 'to']",
        "--['dance']",
        "--['frolic']",
    ]

    # no change when inserting same sequence
    trie.insert(
        ["we", "like", "to", "frolic"],
        ["BLOCK 8", "BLOCK 8", "BLOCK 9", "BLOCK 11"],
    )
    assert trie.pretty_format() == [
        "['i', 'like', 'to']",
        "--['eat']",
        "----['pie']",
        "----['pizza']",
        "--['dance']",
        "['we', 'like', 'to']",
        "--['dance']",
        "--['frolic']",
    ]


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_match_prefix_simple() -> None:
    trie = RadixTrie()
    trie.insert(
        ["i", "like", "to", "eat", "pie"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"],
    )
    _, blocks = trie.match_prefix(["i"])
    assert blocks == ["BLOCK 0"]
    _, blocks = trie.match_prefix(["i", "like"])
    assert blocks == ["BLOCK 0", "BLOCK 1"]
    _, blocks = trie.match_prefix(["i", "like", "to", "eat", "pie"])
    assert blocks == ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"]
    _, blocks = trie.match_prefix(["i", "like", "to", "eat", "pie", "everyday"])
    assert blocks == ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"]
    _, blocks = trie.match_prefix(["we"])
    assert blocks == []


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_match_prefix_complex() -> None:
    trie = RadixTrie()
    trie.insert(
        ["i", "like", "to", "eat", "pie"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"],
    )
    trie.insert(
        ["i", "like", "to", "eat", "pizza"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 5"],
    )
    trie.insert(
        ["i", "like", "to", "dance"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 6", "BLOCK 7"],
    )
    trie.insert(
        ["we", "like", "to", "dance"],
        ["BLOCK 8", "BLOCK 8", "BLOCK 9", "BLOCK 10"],
    )
    trie.insert(
        ["we", "like", "to", "frolic"],
        ["BLOCK 8", "BLOCK 8", "BLOCK 9", "BLOCK 11"],
    )
    _, blocks = trie.match_prefix(["i", "like", "to", "eat", "dominos"])
    assert blocks == ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3"]
    _, blocks = trie.match_prefix(["we"])
    assert blocks == ["BLOCK 8"]
    _, blocks = trie.match_prefix(["we", "love", "chicken", "nuggets"])
    assert blocks == ["BLOCK 8"]
    _, blocks = trie.match_prefix(["we", "like", "to", "dance", "daily"])
    assert blocks == ["BLOCK 8", "BLOCK 8", "BLOCK 9", "BLOCK 10"]


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_insert_at_node() -> None:
    trie = RadixTrie()
    trie.insert(
        ["i", "like", "to", "eat", "pie"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"],
    )

    node, blocks = trie.match_prefix(["i", "like", "to", "eat", "dominos"])
    assert blocks == ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3"]
    trie.insert(["dominos"], ["BLOCK 5"], node=node)
    assert trie.pretty_format() == [
        "['i', 'like', 'to', 'eat']",
        "--['pie']",
        "--['dominos']",
    ]

    node, blocks = trie.match_prefix(["we", "love", "burgers"])
    assert blocks == []
    trie.insert(
        ["we", "love", "burgers"], ["BLOCK 6", "BLOCK 7", "BLOCK 8"], node=node
    )
    assert trie.pretty_format() == [
        "['i', 'like', 'to', 'eat']",
        "--['pie']",
        "--['dominos']",
        "['we', 'love', 'burgers']",
    ]

    node, blocks = trie.match_prefix(["we", "love", "cheese", "burgers"])
    assert blocks == ["BLOCK 6", "BLOCK 7"]
    trie.insert(["cheese", "burgers"], ["BLOCK 9", "BLOCK 10"], node=node)
    assert trie.pretty_format() == [
        "['i', 'like', 'to', 'eat']",
        "--['pie']",
        "--['dominos']",
        "['we', 'love']",
        "--['burgers']",
        "--['cheese', 'burgers']",
    ]


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_raises() -> None:
    trie = RadixTrie()
    with pytest.raises(ValueError):
        trie.insert(["i", "like", "dogs"], ["BLOCK 0"])
    with pytest.raises(ValueError):
        trie.insert([], [])
    with pytest.raises(ValueError):
        trie.match_prefix([])
