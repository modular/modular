# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import numpy as np
import pytest
from max.nn.kv_cache.paged_cache import block_utils
from max.nn.kv_cache.paged_cache.block_utils import (
    hash_block_tokens,
    hash_request_tokens,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_mojo_hasher", [True, False])
@pytest.mark.parametrize("block_size", [1, 2, 4, 64, 128, 256, 1024])
@pytest.mark.parametrize("prompt_len", [16, 65536])
async def test_basic(
    use_mojo_hasher: bool, block_size: int, prompt_len: int
) -> None:
    # Set the global variable to toggle mojo block hashing.
    block_utils.ENABLE_MOJO_BLOCK_HASHER = use_mojo_hasher

    prompt = np.arange(prompt_len, dtype=np.int64)
    block_hashes = hash_request_tokens(prompt, block_size)
    assert len(block_hashes) == prompt_len // block_size

    # Check that they form a chain
    for i in range(len(block_hashes)):
        block_hash = block_hashes[i]
        block_token_ids = prompt[i * block_size : (i + 1) * block_size]
        expected_hash = block_hash.value
        actual_hash = hash_block_tokens(
            block_token_ids,
            block_hash.parent_hash_value,
        ).value
        assert expected_hash == actual_hash

    hash_vals = [block_hash.value for block_hash in block_hashes]

    # Check that the hash values are non-zero.
    # Technically a 0 hash is possible, but it's extremely unlikely and usually
    # indicates a bug in the hasher.
    assert 0 not in hash_vals

    # Check that the hash values are unique.
    assert len(set(hash_vals)) == len(hash_vals)
