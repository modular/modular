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
async def test_kv_cache_radix_trie_insert_at_split_node() -> None:
    trie = RadixTrie()
    trie.insert(
        ["i", "like", "to", "eat", "pie"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"],
    )

    node, _ = trie.match_prefix(["i", "like"])

    trie.insert(
        ["i", "like", "to", "eat", "pizza"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 5"],
    )

    trie.insert(
        ["dancing", "all", "night"],
        ["BLOCK 6", "BLOCK 7", "BLOCK 8"],
        node=node,
    )

    assert trie.pretty_format() == [
        "['i', 'like']",
        "--['to', 'eat']",
        "----['pie']",
        "----['pizza']",
        "--['dancing', 'all', 'night']",
    ]


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_eviction() -> None:
    trie = RadixTrie()

    seq0 = 0
    seq1 = 1

    node0 = trie.insert(
        [
            "i",
            "like",
            "to",
        ],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2"],
    )
    trie.mark_in_use_by(node0, seq0)
    node0 = trie.insert(["eat"], ["BLOCK 3"], node=node0)
    trie.mark_in_use_by(node0, seq0)
    node0 = trie.insert(["pie"], ["BLOCK 4"], node=node0)
    trie.mark_in_use_by(node0, seq0)

    node1 = trie.insert(
        [
            "i",
            "like",
            "to",
            "dance",
            "all",
            "night",
        ],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 5", "BLOCK 6", "BLOCK 7"],
    )
    trie.mark_in_use_by(node1, seq1)

    evicted_blocks = trie.evict_blocks(desired_num_evicted=999)
    assert len(evicted_blocks) == 0
    assert trie.pretty_format() == [
        "['i', 'like', 'to']",
        "--['eat']",
        "----['pie']",
        "--['dance', 'all', 'night']",
    ]

    trie.mark_not_in_use_by(node1, seq1)
    evicted_blocks = trie.evict_blocks(desired_num_evicted=1)
    assert set(evicted_blocks) == set(["BLOCK 7"])
    assert trie.pretty_format() == [
        "['i', 'like', 'to']",
        "--['eat']",
        "----['pie']",
        "--['dance', 'all']",
    ]

    evicted_blocks = trie.evict_blocks(desired_num_evicted=999)
    assert set(evicted_blocks) == set(["BLOCK 6", "BLOCK 5"])
    assert trie.pretty_format() == [
        "['i', 'like', 'to']",
        "--['eat']",
        "----['pie']",
    ]

    evicted_blocks = trie.evict_blocks(desired_num_evicted=999)
    assert len(evicted_blocks) == 0
    assert trie.pretty_format() == [
        "['i', 'like', 'to']",
        "--['eat']",
        "----['pie']",
    ]

    trie.mark_not_in_use_by(node0, seq0)
    evicted_blocks = trie.evict_blocks(desired_num_evicted=999)
    assert len(evicted_blocks) == 5
    assert trie.pretty_format() == []


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_eviction_lru() -> None:
    trie = RadixTrie()

    seq0 = 0
    seq1 = 1
    seq2 = 2

    node0 = trie.insert(
        ["i", "like", "to", "sleep", "snugly"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"],
    )
    node1 = trie.insert(
        ["i", "like", "to", "sleep", "tight"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 5"],
    )
    node2 = trie.insert(
        ["i", "like", "to", "eat", "hamburgers"],
        ["BLOCK 0", "BLOCK 1", "BLOCK 2", "BLOCK 6", "BLOCK 7"],
    )

    trie.mark_in_use_by(node0, seq0)
    trie.mark_in_use_by(node1, seq1)
    trie.mark_in_use_by(node2, seq2)
    trie.mark_not_in_use_by(node1, seq1)
    trie.mark_not_in_use_by(node2, seq2)

    # LRU: seq0 < seq1 < seq2

    # As seq0 is still in use, it cannot be evicted.
    # Thus, we free the next LRU leaf, which is the token "tight" of seq1 with block 5
    evicted_blocks = trie.evict_blocks(desired_num_evicted=1)
    assert set(evicted_blocks) == set(["BLOCK 5"])

    evicted_blocks = trie.evict_blocks(desired_num_evicted=2)
    assert set(evicted_blocks) == set(["BLOCK 6", "BLOCK 7"])
    evicted_blocks = trie.evict_blocks(desired_num_evicted=999)
    assert set(evicted_blocks) == set()


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_raises() -> None:
    trie = RadixTrie()
    with pytest.raises(ValueError):
        trie.insert(["i", "like", "dogs"], ["BLOCK 0"])
    with pytest.raises(ValueError):
        trie.insert([], [])
    with pytest.raises(ValueError):
        trie.match_prefix([])


@pytest.mark.asyncio
async def test_kv_cache_radix_trie_with_page_size_gt_1() -> None:
    trie = RadixTrie(page_size=2)
    trie.insert(["i", "like", "tasty", "food"], ["BLOCK 0", "BLOCK 1"])
    assert trie.pretty_format() == [
        "['i', 'like', 'tasty', 'food']",
    ]

    _, blocks = trie.match_prefix(["i"])
    assert blocks == []

    _, blocks = trie.match_prefix(["i", "like"])
    assert blocks == ["BLOCK 0"]

    # AIPIPE-323: We should also return first token of BLOCK 1 in future
    _, blocks = trie.match_prefix(["i", "like", "tasty"])
    assert blocks == ["BLOCK 0"]

    _, blocks = trie.match_prefix(["i", "like", "tasty", "food"])
    assert blocks == ["BLOCK 0", "BLOCK 1"]

    # AIPIPE-323: We should also return first token of BLOCK 1 in future
    _, blocks = trie.match_prefix(["i", "like", "tasty", "pizza"])
    assert blocks == ["BLOCK 0"]

    _, blocks = trie.match_prefix(["we", "like", "tasty", "food"])
    assert blocks == []

    _, blocks = trie.match_prefix(["i", "like", "yummy", "food"])
    assert blocks == ["BLOCK 0"]

    node = trie.insert(
        ["i", "like", "tasty", "oranges"], ["BLOCK 0", "BLOCK 2"]
    )
    assert trie.pretty_format(print_blocks=True) == [
        "['i', 'like'] : ['BLOCK 0']",
        "--['tasty', 'food'] : ['BLOCK 1']",
        "--['tasty', 'oranges'] : ['BLOCK 2']",
    ]
    trie.mark_in_use_by(node, 0)

    assert trie.evict_blocks(desired_num_evicted=999) == ["BLOCK 1"]
    assert trie.pretty_format(print_blocks=True) == [
        "['i', 'like'] : ['BLOCK 0']",
        "--['tasty', 'oranges'] : ['BLOCK 2']",
    ]
