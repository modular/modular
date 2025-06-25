# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
from max._core_mojo import block_hasher


def test_block_hasher():
    block_size = 128
    num_tokens = 3000
    tokens = np.arange(num_tokens)

    hashes = block_hasher(tokens, block_size, hash("None"))

    assert isinstance(hashes, list)
    assert isinstance(hashes[0], int)
    assert len(hashes) == num_tokens // block_size

    # It is very unlikely (but not impossible) that a valid hasher will return 0
    # Usually a 0 is indicative of a bug of some sort.
    assert 0 not in hashes

    # It is unlikely (but not impossible) that a hasher will return the same value twice.
    # Usually a duplicate is indicative of a bug of some sort.
    seen = set()
    for h in hashes:
        assert h not in seen
        seen.add(h)
