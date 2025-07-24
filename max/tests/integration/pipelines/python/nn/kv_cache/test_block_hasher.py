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


def check_for_collisions(
    prompt_1: np.ndarray, prompt_2: np.ndarray, block_size: int
) -> None:
    block_hashes_1 = hash_request_tokens(prompt_1, block_size, hash("None"))
    block_hashes_2 = hash_request_tokens(prompt_2, block_size, hash("None"))

    hash_vals_1 = [block_hash.value for block_hash in block_hashes_1]
    hash_vals_2 = [block_hash.value for block_hash in block_hashes_2]

    for i, x in enumerate(hash_vals_1):
        if x in hash_vals_2:
            j = hash_vals_2.index(x)
            raise ValueError(
                f"Collision found at idx={i} and idx={j} with hash value {x}"
            )

    # There should be no duplicate hashes.
    assert len(hash_vals_1) + len(hash_vals_2) == len(
        set(hash_vals_1) | set(hash_vals_2)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_mojo_hasher", [True, False])
async def test_collision(use_mojo_hasher: bool) -> None:
    block_utils.ENABLE_MOJO_BLOCK_HASHER = use_mojo_hasher
    block_size = 1

    prompt_1 = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 0])
    prompt_2 = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0])

    check_for_collisions(prompt_1, prompt_2, block_size)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_mojo_hasher", [False, True])
@pytest.mark.parametrize("block_size", [1, 128])
async def test_collision_random(use_mojo_hasher: bool, block_size: int) -> None:
    block_utils.ENABLE_MOJO_BLOCK_HASHER = use_mojo_hasher
    # Picking too large of number of iterations can cause test to timeout.
    iterations = 10

    for _ in range(iterations):
        # Generate arbitrary suffix
        random_prompt = np.random.randint(0, 2, size=8192)

        # Append either 0 or 1 to the beginning of the prompt so they result in
        # unique hashes,
        prompt_1 = np.concatenate([np.array([0]), random_prompt])
        prompt_2 = np.concatenate([np.array([1]), random_prompt])

        # Check for collisions
        check_for_collisions(prompt_1, prompt_2, block_size)
